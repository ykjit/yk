//! Trace promotion: promote values to constants when recording and compiling a trace.

use crate::mt::MTThread;

/// Promote a value.
///
/// When encountered during trace recording, the returned value will be considered constant in the
/// resulting compiled trace.
///
/// The user sees this as `yk_promote` via a macro.
#[no_mangle]
pub extern "C" fn __yk_promote_usize(val: usize) -> usize {
    MTThread::with(|mtt| {
        // We ignore the return value for `promote_usize` as we can't really cancel tracing from
        // this function.
        mtt.promote_usize(val);
    });
    val
}
