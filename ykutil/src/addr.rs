//! Address utilities.

use crate::obj::SELF_BIN_PATH;
use libc::{c_void, dladdr, Dl_info};
use phdrs::objects;
use std::mem::MaybeUninit;
use std::{
    convert::TryFrom,
    ffi::CStr,
    path::{Path, PathBuf},
    ptr::null,
};

/// Given a virtual address, returns a pair indicating the object in which the address originated
/// and the byte offset.
pub fn code_vaddr_to_off(vaddr: usize) -> Option<(PathBuf, u64)> {
    // Find the object file from which the virtual address was loaded.
    let mut info = MaybeUninit::<Dl_info>::uninit();
    if unsafe { dladdr(vaddr as *const c_void, info.as_mut_ptr()) } == 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    let containing_obj = PathBuf::from(unsafe { CStr::from_ptr(info.dli_fname) }.to_str().unwrap());

    // Find the corresponding byte offset of the virtual address in the object.
    for obj in &objects() {
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

/// Find the virtual address of the offset `off` in the binary's "main object".
pub fn off_to_vaddr_main_obj(off: u64) -> Option<usize> {
    // Linux uses the empty string to denote the main object.
    #[cfg(target_os = "linux")]
    return off_to_vaddr(&PathBuf::new(), off);
    #[cfg(not(target_os = "linux"))]
    return off_to_vaddr(&env::current_exe().unwrap(), off);
}

/// Find the virtual address of the offset `off` in the object `containing_obj`.
pub fn off_to_vaddr(containing_obj: &Path, off: u64) -> Option<usize> {
    for obj in &objects() {
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
/// On success returns `Ok((symbol_name, object_path))`, or on failure `Err(())`.
///
/// This function uses `dladdr()` internally, and thus inherits the same symbol visibility rules
/// used there. For example, this function will not find unexported symbols.
pub fn vaddr_to_sym_and_obj(vaddr: usize) -> Result<(&'static CStr, &'static CStr), ()> {
    let mut info: MaybeUninit<Dl_info> = MaybeUninit::uninit();
    if unsafe { dladdr(vaddr as *const c_void, info.as_mut_ptr()) } == 0 {
        return Err(());
    }
    let info = unsafe { info.assume_init() };
    // `dladdr()` returns success if at leaset the virtual address could be mapped to an object
    // file, but here it is crucial that we can also find the symbol that the address belongs to.
    if info.dli_sname == null() {
        return Err(());
    }
    Ok((unsafe { CStr::from_ptr(info.dli_sname) }, unsafe {
        CStr::from_ptr(info.dli_fname)
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        code_vaddr_to_off, off_to_vaddr, off_to_vaddr_main_obj, vaddr_to_sym_and_obj, MaybeUninit,
    };
    use libc::{dladdr, dlsym, Dl_info};
    use std::{ffi::CString, path::PathBuf, ptr};

    #[test]
    fn map_libc() {
        let func = CString::new("getuid").unwrap();
        let vaddr = unsafe { dlsym(ptr::null_mut(), func.as_ptr() as *const i8) };
        assert_ne!(vaddr, ptr::null_mut());
        assert!(code_vaddr_to_off(vaddr as usize).is_some());
    }

    #[test]
    #[no_mangle]
    fn map_so() {
        let vaddr = code_vaddr_to_off as *const u8;
        assert_ne!(vaddr, ptr::null_mut());
        assert!(code_vaddr_to_off(vaddr as usize).is_some());
    }

    /// Check that converting a virtual address (from a shared object) to a file offset and back to
    /// a virtual address correctly round-trips.
    #[test]
    fn round_trip_so() {
        let func = CString::new("getuid").unwrap();
        let func_vaddr = unsafe { dlsym(ptr::null_mut(), func.as_ptr() as *const i8) };
        let mut dlinfo = MaybeUninit::<Dl_info>::uninit();
        assert_ne!(unsafe { dladdr(func_vaddr, dlinfo.as_mut_ptr()) }, 0);
        let dlinfo = unsafe { dlinfo.assume_init() };
        assert_eq!(func_vaddr, dlinfo.dli_saddr);

        let (obj, off) = code_vaddr_to_off(func_vaddr as usize).unwrap();
        assert_eq!(off_to_vaddr(&obj, off).unwrap(), func_vaddr as usize);
    }

    /// Check that converting a virtual address (from the main object) to a file offset and back to
    /// a virtual address correctly round-trips.
    #[no_mangle]
    #[test]
    fn round_trip_main() {
        let func_vaddr = round_trip_main as *const fn();
        let (_, off) = code_vaddr_to_off(func_vaddr as usize).unwrap();
        assert_eq!(off_to_vaddr_main_obj(off).unwrap(), func_vaddr as usize);
    }

    #[test]
    fn vaddr_to_sym_and_obj_found() {
        // To test this we need an exported symbol with a predictable (i.e. unmangled) name.
        use libc::fflush;
        let func_vaddr = fflush as *const fn();
        let (sym, obj) = vaddr_to_sym_and_obj(func_vaddr as usize).unwrap();
        assert_eq!(sym.to_str().unwrap(), "fflush");
        let obj_path = PathBuf::from(obj.to_str().unwrap());
        assert!(obj_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("libc.so."));
    }

    #[test]
    fn vaddr_to_sym_and_obj_cant_find_obj() {
        let func_vaddr = 1; // Obscure address unlikely to be in any loaded object.
        assert!(vaddr_to_sym_and_obj(func_vaddr as usize).is_err());
    }

    #[test]
    fn vaddr_to_sym_and_obj_cant_find_sym() {
        // Address valid, but symbol not exported (test bin not built with `-Wl,--export-dynamic`).
        let func_vaddr = vaddr_to_sym_and_obj_cant_find_sym as *const fn();
        assert!(vaddr_to_sym_and_obj(func_vaddr as usize).is_err());
    }
}
