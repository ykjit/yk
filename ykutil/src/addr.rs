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

#[cfg(test)]
mod tests {
    use super::code_vaddr_to_off;
    use libc::dlsym;
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
}
