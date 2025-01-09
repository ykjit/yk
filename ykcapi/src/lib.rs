//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

// FIXME: This crate was designed to contain the entire public C API surface of Yk. Over time C API
// functions have leaked elsewhere. For example yk_debug_str() and yk_promote_*() are defined
// elsewhere. We should either move all the C API back into this file, or maybe move all of the C
// API into (e.g.) `ykrt::api::c` (and make ykrt a cdylib). The former means you have to `pub`
// stuff in `ykrt`, so perhaps the latter?

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

/// Shutdown this MT instance. Will panic if an error is detected when doing so.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn yk_mt_shutdown(mt: *const MT) {
    unsafe { Arc::from_raw(mt) }.shutdown();
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
    // Stackmap id for the control point.
    smid: u64,
) {
    // FIXME: We can possibly avoid the below (and this entire function) by patching the return
    // address of the control point on the stack to point to the compiled trace. This means we
    // still run the epilogue of the control point function call, which automatically restores the
    // callee-saved registers for us (so we don't have to do it here).
    unsafe {
        std::arch::naked_asm!(
            // Push all registers to the stack as these may contain trace inputs (live
            // variables) referenced by the control point's stackmap.
            //
            // We don't need to push and restore `rdx` since `smid` can never be a live value and
            // thus won't be tracked by the stackmap.
            //
            // FIXME: In the future we want the control point to return naturally into the compiled
            // trace (at the moment we just rip out the control point's stack), which means we then
            // no longer need to recover callee-saved registers as the control point will do this
            // for us.
            "push rax",
            "push rcx",
            "push rbx",
            "push rdi",
            "push rsi",
            "push r8",
            "push r9",
            "push r10",
            "push r11",
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            // Pass the interpreter frame's base pointer via the 4th argument register.
            "mov rcx, rbp",
            "call __ykrt_control_point_real",
            // Restore the previously pushed registers.
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop r11",
            "pop r10",
            "pop r9",
            "pop r8",
            "pop rsi",
            "pop rdi",
            "pop rbx",
            "pop rcx",
            "pop rax",
            "ret",
        );
    }
}

// The actual control point, after we have pushed the callee-saved registers.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn __ykrt_control_point_real(
    mt: *const MT,
    loc: *mut Location,
    // Stackmap id for the control point.
    smid: u64,
    // Frame address of caller.
    frameaddr: *mut c_void,
) {
    if !loc.is_null() {
        let mt = unsafe { &*mt };
        let loc = unsafe { &*loc };
        let arc = unsafe { Arc::from_raw(mt) };
        arc.control_point(loc, frameaddr, smid);
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
