//! Address utilities.

use libc::{c_void, dladdr, Dl_info};
use once_cell::sync::Lazy;
use phdrs::objects;
use std::{
    convert::TryFrom,
    env,
    ffi::{CStr, CString},
    ptr::{null, null_mut},
};

static CUR_EXE_C: Lazy<CString> =
    Lazy::new(|| CString::new(env::current_exe().unwrap().to_str().unwrap()).unwrap());

/// Given a virtual address, returns a pair indicating the object in which the address originated
/// and the byte offset.
pub fn code_vaddr_to_off(vaddr: usize) -> Option<(&'static CStr, u64)> {
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
    let containing_obj = unsafe { CStr::from_ptr(info.dli_fname) };

    // Find the corresponding byte offset of the virtual address in the object.
    for obj in &objects() {
        let mut obj_name = obj.name();
        let obj_name_p = obj_name.as_ptr();
        if unsafe { *obj_name_p } == 0 {
            // On some systems, the empty string indicates the main binary.
            obj_name = CUR_EXE_C.as_c_str();
        }
        if obj_name != containing_obj {
            continue;
        }
        return Some((containing_obj, u64::try_from(vaddr).unwrap() - obj.addr()));
    }
    None
}
