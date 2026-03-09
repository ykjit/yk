use std::ffi::c_void;

/// Return `true` if the frame at address `current` must be a caller of `previous`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn is_caller_frame(cnd_caller: *const c_void, cnd_callee: *const c_void) -> bool {
    cnd_callee < cnd_caller
}
