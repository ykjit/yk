//! Software tracing via ykrustc.

use super::{SirTrace, ThreadTracer, ThreadTracerImpl};
use crate::{errors::InvalidTraceError, sir::SIR, SirLoc};
use libc;
use std::convert::TryFrom;
use ykpack::Local;

/// Softare thread tracer.
struct SWTThreadTracer;

impl ThreadTracerImpl for SWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<SirTrace, InvalidTraceError> {
        let mut len = 0;
        let buf = unsafe { yk_swt_stop_tracing_impl(&mut len) };
        if buf as usize == 0 {
            Err(InvalidTraceError::InternalError)
        } else {
            // When we make a SirTrace, we convert all of the locations from SwtLoc to SirLoc.
            let locs = (0..len)
                .map(|idx| {
                    let idx = isize::try_from(idx).unwrap();
                    let swt_loc = unsafe { &*buf.offset(idx) };
                    let symbol_name = unsafe { std::ffi::CStr::from_ptr(swt_loc.symbol_name) };
                    SirLoc {
                        symbol_name: symbol_name.to_str().unwrap(),
                        bb_idx: swt_loc.bb_idx,
                        addr: None
                    }
                })
                .collect();

            unsafe { libc::free(buf as *mut libc::c_void) };
            Ok(SirTrace::new(locs))
        }
    }
}

pub(crate) fn start_tracing() -> ThreadTracer {
    unsafe {
        yk_swt_start_tracing_impl();
    }
    ThreadTracer {
        t_impl: Box::new(SWTThreadTracer {})
    }
}

// Rust translation of the C code removed in https://github.com/softdevteam/ykrustc/pull/121
#[repr(C)]
#[derive(Copy, Clone)]
struct SwtLoc {
    symbol_name: *const i8,
    bb_idx: u32
}

const TL_TRACE_INIT_CAP: usize = 1024;
const TL_TRACE_REALLOC_CAP: usize = 1024;

/// The trace buffer.
///
/// `calloc` and `free` are directly used instead of `Vec` to avoid calling into traced functions
/// from `__yk_swt_rec_loc`. Otherwise there would be infinite recursion.
#[thread_local]
static mut TRACE_BUF: *mut SwtLoc = 0 as *mut SwtLoc;

/// The number of elements in the trace buffer.
#[thread_local]
static mut TRACE_BUF_LEN: usize = 0;

/// The allocation capacity of the trace buffer (in elements).
#[thread_local]
static mut TRACE_BUF_CAP: usize = 0;

/// true = we are tracing, false = we are not tracing or an error occurred.
#[thread_local]
#[no_mangle]
static mut __YK_SWT_ACTIVE: bool = false;

// =======================
// The following code may not call any non `#[do_not_trace]` function to prevent an infinite
// recursion of `__yk_swt_rec_loc` calling a function calling `__yk_swt_rec_loc`.

/// Start tracing on the current thread.
/// A new trace buffer is allocated and MIR locations will be written into it on
/// subsequent calls to `yk_swt_rec_loc`. If the current thread is already
/// tracing, calling this will lead to undefined behaviour.
#[do_not_trace]
unsafe fn yk_swt_start_tracing_impl() {
    TRACE_BUF = libc::calloc(TL_TRACE_INIT_CAP, std::mem::size_of::<SwtLoc>()) as *mut SwtLoc;
    if TRACE_BUF as usize == 0 {
        std::intrinsics::abort();
    }

    TRACE_BUF_CAP = TL_TRACE_INIT_CAP;
    __YK_SWT_ACTIVE = true;
}

/// Record a location into the trace buffer if tracing is enabled on the current thread.
///
/// This function is separate from `__yk_swt_rec_loc_impl` to keep register spilling off the
/// fast path when tracing is disabled.
#[do_not_trace]
#[no_mangle]
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
unsafe extern "C" fn __yk_swt_rec_loc(symbol_name: *const i8, bb_idx: u32) {
    if !__YK_SWT_ACTIVE {
        return;
    }

    __yk_swt_rec_loc_impl(symbol_name, bb_idx);
}

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
    call __tls_get_addr@plt-4

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

/// This is outlined to make the common case of tracing being disabled faster.
#[do_not_trace]
#[no_mangle]
unsafe extern "C" fn __yk_swt_rec_loc_impl(symbol_name: *const i8, bb_idx: u32) {
    // Check if we need more space and reallocate if necessary.
    if TRACE_BUF_LEN == TRACE_BUF_CAP {
        if TRACE_BUF_CAP >= std::isize::MAX as usize - TL_TRACE_REALLOC_CAP {
            // Trace capacity would overflow.
            __YK_SWT_ACTIVE = false;
            return;
        }
        let new_cap = TRACE_BUF_CAP + TL_TRACE_REALLOC_CAP;

        if new_cap > std::isize::MAX as usize / std::intrinsics::size_of::<SwtLoc>() {
            // New buffer size would overflow.
            __YK_SWT_ACTIVE = false;
            return;
        }
        let new_size = new_cap * std::intrinsics::size_of::<SwtLoc>();

        TRACE_BUF = libc::realloc(TRACE_BUF as *mut _, new_size as usize) as *mut SwtLoc;
        if TRACE_BUF as usize == 0 {
            __YK_SWT_ACTIVE = false;
            return;
        }

        TRACE_BUF_CAP = new_cap;
    }

    *((TRACE_BUF as usize + TRACE_BUF_LEN * std::intrinsics::size_of::<SwtLoc>()) as *mut SwtLoc) =
        SwtLoc {
            symbol_name,
            bb_idx
        };
    TRACE_BUF_LEN += 1;
}

/// Stop tracing on the current thread.
/// On success the trace buffer is returned and the number of locations it
/// holds is written to `*ret_trace_len`. It is the responsibility of the caller
/// to free the returned trace buffer. A NULL pointer is returned on error.
/// Calling this function when tracing was not started with
/// `yk_swt_start_tracing_impl()` results in undefined behaviour.
#[do_not_trace]
unsafe fn yk_swt_stop_tracing_impl(ret_trace_len: &mut usize) -> *mut SwtLoc {
    if !__YK_SWT_ACTIVE {
        libc::puts("no trace recorded\n\0" as *const str as *const i8);
        libc::free(TRACE_BUF as *mut _);
        TRACE_BUF = 0 as *mut SwtLoc;
        TRACE_BUF_LEN = 0;
        *ret_trace_len = 0;
        return 0 as *mut SwtLoc;
    }

    __YK_SWT_ACTIVE = false;

    // We hand ownership of the trace to the caller. The caller is responsible
    // for freeing the trace.
    let ret_trace = TRACE_BUF;
    *ret_trace_len = TRACE_BUF_LEN;

    // Now reset all off the recorder's state.
    TRACE_BUF = 0 as *mut SwtLoc;
    TRACE_BUF_LEN = 0;
    TRACE_BUF_CAP = 0;

    return ret_trace;
}
