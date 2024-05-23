//! Address utilities.

use crate::obj::{PHDR_OBJECT_CACHE, SELF_BIN_PATH};
use cached::proc_macro::cached;
use libc::{c_void, dlsym, Dl_info, RTLD_DEFAULT};
use std::{
    error::Error,
    ffi::{CStr, CString},
    mem::MaybeUninit,
    path::{Path, PathBuf},
};

/// A Rust wrapper around `libc::Dl_info` using FFI types.
///
/// The strings inside are handed out by the loader and can (for now, since we don't support
/// dlclose) be considered of static lifetime. This makes the struct thread safe, and thus
/// cacheable using `#[cached]`.
#[derive(Debug, Clone)]
pub struct DLInfo {
    dli_fname: Option<&'static CStr>,
    dli_fbase: usize,
    dli_sname: Option<&'static CStr>,
    dli_saddr: usize,
}

impl From<Dl_info> for DLInfo {
    fn from(dli: Dl_info) -> Self {
        let dli_fname = if !dli.dli_fname.is_null() {
            Some(unsafe { CStr::from_ptr(dli.dli_fname) })
        } else {
            None
        };

        let dli_sname = if !dli.dli_sname.is_null() {
            Some(unsafe { CStr::from_ptr(dli.dli_sname) })
        } else {
            None
        };

        Self {
            dli_fname,
            dli_fbase: dli.dli_fbase as usize,
            dli_sname,
            dli_saddr: dli.dli_saddr as usize,
        }
    }
}

impl DLInfo {
    pub fn dli_fname(&self) -> Option<&'static CStr> {
        self.dli_fname
    }
    pub fn dli_fbase(&self) -> usize {
        self.dli_fbase
    }
    pub fn dli_sname(&self) -> Option<&'static CStr> {
        self.dli_sname
    }
    pub fn dli_saddr(&self) -> usize {
        self.dli_saddr
    }
}

/// Wraps `libc::dlinfo`.
///
/// Returns `Err` if the underlying call to `libc::dlddr` fails.
///
/// FIXME: This cache should be invalidated (in part, if possible) when a object is loaded or
/// unloaded from the address space.
///
/// FIXME: Consider using a LRU cache to limit memory consumption. The cached crate can do this for
/// us if we can give it a suitable cache size.
///
/// FIXME: This cache is cloning. Performance could probably be improved more.
#[cached]
pub fn dladdr(vaddr: usize) -> Option<DLInfo> {
    let mut info = MaybeUninit::<Dl_info>::uninit();
    if unsafe { libc::dladdr(vaddr as *const c_void, info.as_mut_ptr()) } != 0 {
        Some(unsafe { info.assume_init() }.into())
    } else {
        None
    }
}

/// Given a virtual address, returns a pair indicating the object in which the address originated
/// and the byte offset.
#[cached]
pub fn vaddr_to_obj_and_off(vaddr: usize) -> Option<(PathBuf, u64)> {
    // Find the object file from which the virtual address was loaded.
    let info = dladdr(vaddr).unwrap();
    let containing_obj = PathBuf::from(info.dli_fname.unwrap().to_str().unwrap());

    // Find the corresponding byte offset of the virtual address in the object.
    for obj in PHDR_OBJECT_CACHE.iter() {
        let obj_name = obj.name();
        let obj_name: &Path = if unsafe { *obj_name.as_ptr() } == 0 {
            SELF_BIN_PATH.as_path()
        } else {
            Path::new(obj_name.to_str().unwrap())
        };
        if obj_name != containing_obj {
            continue;
        }
        return Some((containing_obj, u64::try_from(vaddr).unwrap() - obj.addr()));
    }
    None
}

/// Find the virtual address of the offset `off` in the object `containing_obj`.
///
/// Returns `OK(virtual_address)` if a virtual address is found for the object in question,
/// otherwise returns `None`.
///
/// This is fragile and should be avoided if possible. In order for a hit, `containing_obj` must be
/// in the same form as it appears in the program header table. This function makes no attempt to
/// canonicalise equivalent, but different (in terms of string equality) object paths.
pub fn off_to_vaddr(containing_obj: &Path, off: u64) -> Option<usize> {
    for obj in PHDR_OBJECT_CACHE.iter() {
        if Path::new(obj.name().to_str().unwrap()) != containing_obj {
            continue;
        }
        return Some(usize::try_from(off + obj.addr()).unwrap());
    }
    None // Not found.
}

/// Given a virtual address in the current address space, (if possible) determine the name of the
/// symbol this belongs to, and the path to the object from which it came.
///
/// On success returns `Ok` with a `SymbolInObject`, or on failure `Err(())`.
///
/// This function uses `dladdr()` internally, and thus inherits the same symbol visibility rules
/// used there. For example, this function will not find unexported symbols.
pub fn vaddr_to_sym_and_obj(vaddr: usize) -> Option<DLInfo> {
    // `dladdr()` returns success if at least the virtual address could be mapped to an object
    // file, but here it is crucial that we can also find the symbol that the address belongs to.
    match dladdr(vaddr) {
        Some(x) if x.dli_sname().is_some() => Some(x),
        Some(_) | None => None,
    }
}

/// A thin wrapper around `dlsym()` for mapping symbol names to virtual addresses.
///
/// FIXME: Look for raw uses of `dlsym()` throughout our code base and replace them with a call to
/// this wrapper. Related: https://github.com/ykjit/yk/issues/835
pub fn symbol_to_ptr(name: &str) -> Result<*const (), Box<dyn Error>> {
    let s = CString::new(name).unwrap();
    let p = unsafe { dlsym(RTLD_DEFAULT, s.as_ptr()) };
    if !p.is_null() {
        Ok(p as *const _)
    } else {
        Err(format!("dlsym(\"{name}\") returned NULL").into())
    }
}

#[cfg(test)]
mod tests {
    use super::{off_to_vaddr, vaddr_to_obj_and_off, vaddr_to_sym_and_obj, MaybeUninit};
    use crate::obj::PHDR_MAIN_OBJ;
    use libc::{dlsym, Dl_info};
    use std::{ffi::CString, ptr};

    #[test]
    fn map_libc() {
        let func = CString::new("getuid").unwrap();
        let vaddr = unsafe { dlsym(ptr::null_mut(), func.as_ptr()) };
        assert_ne!(vaddr, ptr::null_mut());
        assert!(vaddr_to_obj_and_off(vaddr as usize).is_some());
    }

    #[test]
    #[no_mangle]
    fn map_so() {
        let vaddr = vaddr_to_obj_and_off as *const u8;
        assert_ne!(vaddr, ptr::null_mut());
        assert!(vaddr_to_obj_and_off(vaddr as usize).is_some());
    }

    /// Check that converting a virtual address (from a shared object) to a file offset and back to
    /// a virtual address correctly round-trips.
    #[test]
    fn round_trip_so() {
        let func = CString::new("getuid").unwrap();
        let func_vaddr = unsafe { dlsym(ptr::null_mut(), func.as_ptr()) };
        let mut dlinfo = MaybeUninit::<Dl_info>::uninit();
        assert_ne!(unsafe { libc::dladdr(func_vaddr, dlinfo.as_mut_ptr()) }, 0);
        let dlinfo = unsafe { dlinfo.assume_init() };
        assert_eq!(func_vaddr, dlinfo.dli_saddr);

        let (obj, off) = vaddr_to_obj_and_off(func_vaddr as usize).unwrap();
        assert_eq!(off_to_vaddr(&obj, off).unwrap(), func_vaddr as usize);
    }

    /// Check that converting a virtual address (from the main object) to a file offset and back to
    /// a virtual address correctly round-trips.
    #[no_mangle]
    #[test]
    fn round_trip_main() {
        let func_vaddr = round_trip_main as *const fn();
        let (_obj, off) = vaddr_to_obj_and_off(func_vaddr as usize).unwrap();
        assert_eq!(
            off_to_vaddr(&PHDR_MAIN_OBJ, off).unwrap(),
            func_vaddr as usize
        );
    }

    #[test]
    fn vaddr_to_sym() {
        // To test this we need an exported symbol with a predictable (i.e. unmangled) name.
        use libc::fflush;
        let func_vaddr = fflush as *const fn();
        let sio = vaddr_to_sym_and_obj(func_vaddr as usize).unwrap();
        assert!(matches!(
            sio.dli_sname().unwrap().to_str().unwrap(),
            "fflush" | "_IO_fflush"
        ));
    }

    #[test]
    fn vaddr_to_sym_and_obj_cant_find_obj() {
        let func_vaddr = 1; // Obscure address unlikely to be in any loaded object.
        assert!(vaddr_to_sym_and_obj(func_vaddr as usize).is_none());
    }

    #[test]
    fn vaddr_to_sym_and_obj_cant_find_sym() {
        // Address valid, but symbol not exported (test bin not built with `-Wl,--export-dynamic`).
        let func_vaddr = vaddr_to_sym_and_obj_cant_find_sym as *const fn();
        assert!(vaddr_to_sym_and_obj(func_vaddr as usize).is_none());
    }
}
