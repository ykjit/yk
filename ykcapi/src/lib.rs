//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

#![allow(clippy::missing_safety_doc)]
#![feature(naked_functions)]

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

// The new control point called after the interpreter has been patched by ykllvm.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
pub extern "C" fn __ykrt_control_point(
    mt: *const MT,
    loc: *mut Location,
    ctrlp_vars: *mut c_void,
    // Frame address of caller.
    frameaddr: *mut c_void,
    // Stackmap id for the control point.
    smid: u64,
) {
    // FIXME: We can possibly avoid the below (and this entire function) by patching the return
    // address of the control point on the stack to point to the compiled trace. This means we
    // still run the epilogue of the control point function call, which automatically restores the
    // callee-saved registers for us (so we don't have to do it here).
    unsafe {
        std::arch::asm!(
            // Push callee-saved registers to the stack as these may contain trace inputs (live
            // variables) referenced by the control point's stackmap.
            "push rbx",
            "push rdi",
            "push rsi",
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            "call __ykrt_control_point_real",
            // Restore the previously pushed registers.
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop rsi",
            "pop rdi",
            "pop rbx",
            "ret",
            options(noreturn)
        );
    }
}

// The actual control point, after we have pushed the callee-saved registers.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn __ykrt_control_point_real(
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
