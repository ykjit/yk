//! Software tracing via ykrustc.

use self::trace_buffer::TraceBuffer;
use super::{SirTrace, ThreadTracer, ThreadTracerImpl};
use crate::{errors::InvalidTraceError, sir::SirLoc};
use std::cell::UnsafeCell;

/// Softare thread tracer.
struct SWTThreadTracer;

/// Stop tracing on the current thread.
impl ThreadTracerImpl for SWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<SirTrace, InvalidTraceError> {
        if unsafe { !__YK_SWT_ACTIVE } {
            println!("attempted to stop tracing when no tracer is active");
            return Err(InvalidTraceError::InternalError);
        }

        unsafe {
            __YK_SWT_ACTIVE = false;
        }

        let locs = TRACE_BUF.with(|trace_buf| {
            // When we make a SirTrace, we convert all of the locations from SwtLoc to SirLoc.
            trace_buf.get_sir_locs_and_clear()
        });

        Ok(SirTrace::new(locs))
    }
}

pub(super) fn start_tracing() -> ThreadTracer {
    TRACE_BUF.with(|trace_buf| {
        assert!(trace_buf.is_empty());
    });

    unsafe {
        __YK_SWT_ACTIVE = true;
    }

    ThreadTracer {
        t_impl: Box::new(SWTThreadTracer {}),
    }
}

// Rust translation of the C code removed in https://github.com/softdevteam/ykrustc/pull/121
#[repr(C)]
#[derive(Copy, Clone)]
struct SwtLoc {
    symbol_name: *const i8,
    bb_idx: u32,
}

thread_local! {
    /// The trace buffer.
    static TRACE_BUF: TraceBuffer = TraceBuffer::new();
}

/// true = we are tracing, false = we are not tracing or an error occurred.
#[thread_local]
#[no_mangle]
static mut __YK_SWT_ACTIVE: bool = false;

/// Record a location into the trace buffer if tracing is enabled on the current thread.
///
/// This function is separate from `__yk_swt_rec_loc_impl` to keep register spilling off the
/// fast path when tracing is disabled.
#[no_mangle]
//#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
unsafe extern "C" fn __yk_swt_rec_loc(symbol_name: *const i8, bb_idx: u32) {
    if !__YK_SWT_ACTIVE {
        return;
    }

    __yk_swt_rec_loc_impl(symbol_name, bb_idx);
}

// FIXME this doesn't export __yk_swt_rec_loc from the dylib
/*
// Cranelift produces suboptimal code for the Rust version of `__yk_swt_rec_loc` (redundant spills).
// When running on x86_64 and using ELF TLS we use our own handrolled implementation of
// `__yk_swt_rec_loc`.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
global_asm!(
    r#"
.intel_syntax noprefix
.globl __yk_swt_rec_loc
.type __yk_swt_rec_loc,@function
# This section is allocatable (mapped into memory) and executable
.section .text.__yk_swt_rec_loc,"ax",@progbits
__yk_swt_rec_loc:
    push rdi

# The following instructions must match these exact bytes. `__tls_get_addr` walks a linked list,
# which is relatively slow. For this reason most linkers will replace it with a direct `%fs:0`
# register access whenever possible, which is much faster. If this code block is even slightly
# changed, the linker will error when attempting to do this.
    data16 lea rdi, [rip+__YK_SWT_ACTIVE@tlsgd]
.byte 0x66 # data16
.byte 0x66 # data16
.byte 0x48 # rex
    call __tls_get_addr@plt

    movzx eax, BYTE PTR [rax]
    test eax, eax
    jne __yk_swt_rec_loc_active
    pop rdi
    ret
__yk_swt_rec_loc_active:
    pop rdi
    call __yk_swt_rec_loc_impl
    ret
.size __yk_swt_rec_loc, .-__yk_swt_rec_loc
.att_syntax prefix
"#
);
*/

/// This is outlined to make the common case of tracing being disabled faster.
#[no_mangle]
unsafe extern "C" fn __yk_swt_rec_loc_impl(symbol_name: *const i8, bb_idx: u32) {
    TRACE_BUF.with(|trace_buf| {
        trace_buf.push(SwtLoc {
            symbol_name,
            bb_idx,
        });
    });
}

mod trace_buffer {
    use super::*;

    /// A buffer containing [`SwtLoc`]s. All public methods only require an immutable reference,
    /// thereby allowing it to be stored inside a thread local without the overhead of a `RefCell`.
    pub(super) struct TraceBuffer(UnsafeCell<Vec<SwtLoc>>);

    impl TraceBuffer {
        pub(super) fn new() -> Self {
            TraceBuffer(UnsafeCell::new(Vec::with_capacity(1024)))
        }

        pub(super) fn is_empty(&self) -> bool {
            // SAFETY: The api of `TraceBuffer` prevents any mutable references for the duration of
            // this call.
            unsafe { (&*self.0.get()).is_empty() }
        }

        #[inline]
        pub(super) fn push(&self, loc: SwtLoc) {
            // SAFETY: The api of `TraceBuffer` prevents any other references for the duration of
            // this call.
            unsafe {
                (&mut *self.0.get()).push(loc);
            }
        }

        pub(super) fn get_sir_locs_and_clear(&self) -> Vec<SirLoc> {
            // SAFETY: The api of `TraceBuffer` prevents any other references for the duration of
            // this call.
            unsafe { &mut *self.0.get() }
                .drain(..)
                .map(|swt_loc| {
                    let symbol_name = unsafe { std::ffi::CStr::from_ptr(swt_loc.symbol_name) };
                    SirLoc {
                        symbol_name: symbol_name.to_str().unwrap(),
                        bb_idx: swt_loc.bb_idx,
                        addr: None,
                    }
                })
                .collect()
        }
    }
}
