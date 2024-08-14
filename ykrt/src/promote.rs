//! Trace promotion: promote values to constants when recording and compiling a trace.
//!
//! In C, these are exposed to the user via the `yk_promote` value which automatically picks the
//! right method in this module to call.

use crate::mt::MTThread;
use std::ffi::c_int;

/// Promote a `usize` during trace recording.
#[no_mangle]
pub extern "C" fn __yk_promote_c_int(val: c_int) -> c_int {
    MTThread::with(|mtt| {
        // We ignore the return value for `promote_usize` as we can't really cancel tracing from
        // this function.
        mtt.promote_i32(val);
    });
    val
}

/// Promote a `usize` during trace recording.
#[no_mangle]
pub extern "C" fn __yk_promote_usize(val: usize) -> usize {
    MTThread::with(|mtt| {
        // We ignore the return value for `promote_usize` as we can't really cancel tracing from
        // this function.
        mtt.promote_usize(val);
    });
    val
}
