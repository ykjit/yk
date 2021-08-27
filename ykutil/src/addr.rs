//! Address utilities.

use libc::{c_void, dladdr, Dl_info};
use phdrs::objects;
use std::{
    convert::TryFrom,
    env,
    ffi::CStr,
    fs,
    path::{Path, PathBuf},
    ptr::{null, null_mut},
};

// Make the path `p` a canonical absolute path.
fn canonicalise_path(p: &CStr) -> PathBuf {
    // On linux the empty object path means the "main binary".
    #[cfg(target_os = "linux")]
    if p.to_str().unwrap() == "" {
        return PathBuf::new();
    }
    let p_path = Path::new(p.to_str().unwrap());
    if p.to_str().unwrap() == "linux-vdso.so.1" {
        // The VDSO isn't a real file that can be canonicalised.
        p_path.to_owned()
    } else {
        fs::canonicalize(p_path).unwrap()
    }
}

/// Given a virtual address, returns a pair indicating the object in which the address originated
/// and the byte offset.
pub fn code_vaddr_to_off(vaddr: usize) -> Option<(PathBuf, u64)> {
    // Find the object file from which the virtual address was loaded.
    let mut info: Dl_info = Dl_info {
        dli_fname: null(),
        dli_fbase: null_mut(),
        dli_sname: null(),
        dli_saddr: null_mut(),
    };
    if unsafe { dladdr(vaddr as *const c_void, &mut info as *mut Dl_info) } == 0 {
        return None;
    }
    let containing_obj = canonicalise_path(unsafe { CStr::from_ptr(info.dli_fname) });

    // Find the corresponding byte offset of the virtual address in the object.
    for obj in &objects() {
        let obj_name = obj.name();
        let obj_name: PathBuf = if unsafe { *obj_name.as_ptr() } == 0 {
            // On some systems, the empty string indicates the main binary.
            let exe = env::current_exe().unwrap();
            debug_assert_eq!(&fs::canonicalize(&exe).unwrap(), &exe);
            exe
        } else {
            canonicalise_path(obj_name)
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
        if canonicalise_path(obj.name()) != containing_obj {
            continue;
        }
        return Some(usize::try_from(off + obj.addr()).unwrap());
    }
    None // Not found.
}

#[cfg(test)]
mod tests {
    use super::{code_vaddr_to_off, off_to_vaddr, off_to_vaddr_main_obj};
    use libc::{dladdr, dlsym, Dl_info};
    use std::{ffi::CString, ptr};

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
        let mut dlinfo: Dl_info = Dl_info {
            dli_fname: ptr::null(),
            dli_fbase: ptr::null_mut(),
            dli_sname: ptr::null(),
            dli_saddr: ptr::null_mut(),
        };
        assert_ne!(
            unsafe { dladdr(func_vaddr, &mut dlinfo as *mut Dl_info) },
            0
        );
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
}
