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
    mem::forget,
    ptr,
    sync::Arc,
};
use ykrt::{HotThreshold, Location, MT};

#[no_mangle]
pub unsafe extern "C" fn yk_mt_new(err_msg: *mut *const c_char) -> *const MT {
    match MT::new() {
        Ok(mt) => Arc::into_raw(mt),
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
pub extern "C" fn yk_mt_drop(mt: *const MT) {
    let _mt = unsafe { Arc::from_raw(mt) };
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
    mt: *const MT,
    loc: *mut Location,
    ctrlp_vars: *mut c_void,
    // Frame address of caller.
    frameaddr: *mut c_void,
    // Stackmap id for the control point.
    smid: u64,
) {
    debug_assert!(!ctrlp_vars.is_null());
    if !loc.is_null() {
        let mt = unsafe { &*mt };
        let loc = unsafe { &*loc };
        let arc = unsafe { Arc::from_raw(mt) };
        arc.control_point(loc, ctrlp_vars, frameaddr, smid);
        forget(arc);
    }
}

#[no_mangle]
pub unsafe extern "C" fn yk_mt_hot_threshold_set(mt: *const MT, hot_threshold: HotThreshold) {
    let arc = unsafe { Arc::from_raw(mt) };
    arc.set_hot_threshold(hot_threshold);
    forget(arc);
}

#[no_mangle]
pub unsafe extern "C" fn yk_mt_sidetrace_threshold_set(mt: *const MT, hot_threshold: HotThreshold) {
    let arc = unsafe { Arc::from_raw(mt) };
    arc.set_sidetrace_threshold(hot_threshold);
    forget(arc);
}

#[no_mangle]
pub extern "C" fn yk_location_new() -> Location {
    Location::new()
}

#[no_mangle]
pub extern "C" fn yk_location_drop(loc: Location) {
    drop(loc)
}
