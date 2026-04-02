use std::ffi::c_void;

/// Return `true` if the frame at address `cnd_callee` is definitely a callee of `cnd_caller`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn is_callee_frame(cnd_caller: *const c_void, cnd_callee: *const c_void) -> bool {
    cnd_callee > cnd_caller
}

/// Return `true` if the frame at address `cnd_caller` is definitely a caller of `cnd_callee`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn is_caller_frame(cnd_caller: *const c_void, cnd_callee: *const c_void) -> bool {
    cnd_callee < cnd_caller
}
