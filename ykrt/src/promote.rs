//! Trace promotion: promote values to constants when recording and compiling a trace.
//!
//! In C, these are exposed to the user via the `yk_promote` value which automatically picks the
//! right method in this module to call.

use crate::mt::MTThread;
use std::ffi::{c_int, c_longlong, c_uint};

/// Promote a `c_int` during trace recording.
#[no_mangle]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub extern "C" fn __yk_promote_c_int(val: c_int) -> c_int {
    MTThread::with(|mtt| {
        // We ignore the return value as we can't really cancel tracing from this function.
        mtt.promote_i32(val);
    });
    val
}

/// Promote a `c_uint` during trace recording.
#[no_mangle]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub extern "C" fn __yk_promote_c_unsigned_int(val: c_uint) -> c_uint {
    MTThread::with(|mtt| {
        // We ignore the return value as we can't really cancel tracing from this function.
        mtt.promote_u32(val);
    });
    val
}

/// Promote a `usize` during trace recording.
#[no_mangle]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub extern "C" fn __yk_promote_c_long_long(val: c_longlong) -> c_longlong {
    MTThread::with(|mtt| {
        // We ignore the return value as we can't really cancel tracing from this function.
        mtt.promote_i64(val);
    });
    val
}

/// Promote a `usize` during trace recording.
#[no_mangle]
pub extern "C" fn __yk_promote_usize(val: usize) -> usize {
    MTThread::with(|mtt| {
        // We ignore the return value as we can't really cancel tracing from this function.
        mtt.promote_usize(val);
    });
    val
}
