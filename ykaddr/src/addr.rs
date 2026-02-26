//! Address utilities.

use crate::obj::{PHDR_OBJECT_CACHE, SELF_BIN_PATH};
use cached::proc_macro::cached;
use libc::{Dl_info, RTLD_DEFAULT, c_void, dlsym};
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
/// and the byte offset into that object.
#[cached]
pub fn vaddr_to_obj_and_off(vaddr: usize) -> Option<(PathBuf, u64)> {
    for obj in PHDR_OBJECT_CACHE.iter() {
        for seg in obj.phdrs() {
            let seg_start_vaddr = obj.addr() + seg.vaddr();
            let seg_vaddr_rng = seg_start_vaddr..(seg_start_vaddr + seg.memsz());
            if seg_vaddr_rng.contains(&u64::try_from(vaddr).unwrap()) {
                let off_from_seg_start = vaddr - usize::try_from(seg_start_vaddr).unwrap();
                let off = usize::try_from(seg.offset()).unwrap() + off_from_seg_start;
                let obj_name: PathBuf = if obj.name().is_empty() {
                    // Some systems use the empty string to denote the main executable object.
                    SELF_BIN_PATH.clone()
                } else {
                    PathBuf::from(obj.name().to_str().unwrap())
                };
                return Some((obj_name, u64::try_from(off).unwrap()));
            }
        }
    }
    // Didn't find a on object containing that address.
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
        for seg in obj.phdrs() {
            let seg_off_range = seg.offset()..(seg.offset() + seg.filesz());
            if seg_off_range.contains(&off) {
                let off_from_seg_start = off - seg.offset();
                return Some(
                    usize::try_from(obj.addr() + seg.vaddr() + off_from_seg_start).unwrap(),
                );
            }
        }
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
    use super::{MaybeUninit, off_to_vaddr, vaddr_to_obj_and_off, vaddr_to_sym_and_obj};
    use crate::obj::PHDR_MAIN_OBJ;
    use libc::{Dl_info, dlsym};
    use std::{ffi::CString, ptr};

    #[test]
    fn vaddr_to_obj_and_off_syscall() {
        let func = CString::new("getuid").unwrap();
        let vaddr = unsafe { dlsym(ptr::null_mut(), func.as_ptr()) };
        assert_ne!(vaddr, ptr::null_mut());
        let (_obj, off) = vaddr_to_obj_and_off(vaddr as usize).unwrap();
        assert!(off < u64::try_from(vaddr as usize).unwrap());
    }

    #[test]
    fn vaddr_to_obj_and_off_libc() {
        let func = CString::new("strdup").unwrap();
        let vaddr = unsafe { dlsym(ptr::null_mut(), func.as_ptr()) };
        assert_ne!(vaddr, ptr::null_mut());
        let (_obj, off) = vaddr_to_obj_and_off(vaddr as usize).unwrap();
        assert!(off < u64::try_from(vaddr as usize).unwrap());
    }

    #[test]
    fn vaddr_to_obj_and_off_main_exe() {
        let vaddr = vaddr_to_obj_and_off_main_exe as *const () as usize;
        let (obj, off) = vaddr_to_obj_and_off(vaddr).unwrap();
        // because the loader will load the object a +ve offset from the start of the address space.
        assert!(off < u64::try_from(vaddr).unwrap());
        assert!(
            obj.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("ykaddr-")
        );
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
        assert_ne!(off, u64::try_from(func_vaddr as usize).unwrap());
        assert_eq!(off_to_vaddr(&obj, off).unwrap(), func_vaddr as usize);
    }

    /// Check that converting a virtual address (from the main object) to a file offset and back to
    /// a virtual address correctly round-trips.
    #[test]
    fn round_trip_main() {
        let func_vaddr = round_trip_main as *const fn();
        let (_obj, off) = vaddr_to_obj_and_off(func_vaddr as usize).unwrap();
        assert_ne!(off, u64::try_from(func_vaddr as usize).unwrap());
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
