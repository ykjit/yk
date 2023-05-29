//! Run-time deoptimisation support: when a guard fails, this module restores the state necessary
//! to resume interpreter execution.

use std::{arch::asm, ffi::c_void};

/// The `__llvm__deoptimize()` function required by `llvm.experimental.deoptimize` intrinsic, that
/// we use for exiting to the stop-gap interpreter on guard failure.
#[cfg(target_arch = "x86_64")]
#[naked]
#[no_mangle]
pub extern "C" fn __llvm_deoptimize(
    stackmap: *const c_void,
    aotvals: *const c_void,
    frames: *const c_void,
    retval: *mut c_void,
) -> *const c_void {
    // Push all registers to the stack before they can be clobbered, so that we can find their
    // values after parsing in the stackmap. The order in which we push the registers is equivalent
    // to the Sys-V x86_64 ABI, which the stackmap format uses as well. This function has the
    // "naked" attribute to keep the optimiser from generating the function prologue which messes
    // with the RSP value of the previous stack frame (this value is often referenced by the
    // stackmap).
    unsafe {
        asm!(
            // Save registers that may be referenced by the stackmap to the stack before they get
            // overwritten, so that we read their values later during stopgapping.
            // FIXME: Add other registers that may be referenced by the stackmap.
            "push rsp",
            "push rbp",
            "push rdi",
            "push rsi",
            "push rbx",
            "push rcx",
            "push rdx",
            "push rax",
            // Now we need to call __ykrt_deopt. The arguments need to be in RDI, RSI, RDX,
            // RCX, R8, and R9. The first four arguments (stackmap
            // live variable map, frames, and return value pointer) are already where they
            // need to be as we are just forwarding them from the current function's
            // arguments. The remaining arguments (return address and current stack
            // pointer) need to be in R8 and R9. The return address was at [RSP] before
            // the above pushes, so to find it we need to offset 8 bytes per push.
            "mov r8, [rsp+64]",
            "mov r9, rsp",
            "sub rsp, 8", // Alignment
            "call __ykrt_deopt",
            "add rsp, 72",
            // FIXME: Don't rely on RBP being pushed. Use frame size retrieved from
            // stackmap instead.
            "mov rsp, rbp",
            "pop rbp",
            "ret",
            options(noreturn)
        )
    }
}

#[cfg(not(target_arch = "x86_64"))]
compile_error!("__llvm_deoptimize() not yet implemented for this platform");
