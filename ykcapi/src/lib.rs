//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

#![allow(clippy::missing_safety_doc)]

use std::{
    ffi::{c_char, c_void, CString},
    ptr,
};
use ykrt::{HotThreshold, Location, MT};

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn yk_mt_new(err_msg: *mut *const c_char) -> *mut MT {
    match MT::new() {
        Ok(mt) => Box::into_raw(Box::new(mt)),
        Err(e) => {
            if err_msg.is_null() {
                panic!("{}", e);
            }
            let s = CString::new(e.to_string()).unwrap();
            let b = s.to_bytes_with_nul();
            let buf = unsafe { libc::malloc(b.len()) as *mut i8 };
            unsafe {
                buf.copy_from(b.as_ptr() as *const i8, b.len());
            }
            unsafe { *err_msg = buf };
            ptr::null_mut()
        }
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn yk_mt_drop(mt: *mut MT) {
    unsafe { Box::from_raw(mt) };
}

// The "dummy control point" that is replaced in an LLVM pass.
#[no_mangle]
pub extern "C" fn yk_mt_control_point(_mt: *mut MT, _loc: *mut Location) {
    // Intentionally empty.
}

// The "real" control point, that is called once the interpreter has been patched by ykllvm.
// Returns the address of a reconstructed stack or null if there wasn't a guard failure.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn __ykrt_control_point(
    mt: *mut MT,
    loc: *mut Location,
    ctrlp_vars: *mut c_void,
    // Frame address of caller.
    frameaddr: *mut c_void,
) -> *const c_void {
    debug_assert!(!ctrlp_vars.is_null());
    if !loc.is_null() {
        let mt = unsafe { &*mt };
        let loc = unsafe { &*loc };
        return mt.control_point(loc, ctrlp_vars, frameaddr);
    }
    std::ptr::null()
}

#[no_mangle]
pub extern "C" fn yk_mt_hot_threshold_set(mt: &MT, hot_threshold: HotThreshold) {
    mt.set_hot_threshold(hot_threshold);
}

#[no_mangle]
pub extern "C" fn yk_location_new() -> Location {
    Location::new()
}

#[no_mangle]
pub extern "C" fn yk_location_drop(loc: Location) {
    drop(loc)
}
