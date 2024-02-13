//! Trace promotion: promote values to constants when recording and compiling a trace.

use crate::mt::THREAD_MTTHREAD;

/// Promote a value.
///
/// When encountered during trace recording, the returned value will be considered constant in the
/// resulting compiled trace.
///
/// The user sees this as `yk_promote` via a macro.
#[no_mangle]
pub extern "C" fn __yk_promote_usize(val: usize) {
    THREAD_MTTHREAD.with(|mtt| {
        if let Some(tt) = mtt.thread_tracer.borrow().as_ref() {
            // We ignore the return value for `promote_usize` as we can't really cancel tracing from
            // this function.
            tt.promote_usize(val);
        }
    });
}
