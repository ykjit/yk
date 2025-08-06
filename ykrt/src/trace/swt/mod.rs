//! Software tracer.

use super::{
    AOTTraceIterator, AOTTraceIteratorError, TraceAction, TraceRecorder, TraceRecorderError, Tracer,
};
use crate::mt::MTThread;
use libc;
use std::{cell::RefCell, error::Error, sync::Arc};

#[derive(Debug, Eq, PartialEq, Clone)]
struct TracingBBlock {
    function_index: usize,
    block_index: usize,
}

enum SWTBuf {
    Blocks(Vec<TracingBBlock>),
    Invalid(String),
}

impl SWTBuf {
    fn is_empty(&self) -> bool {
        matches!(self, Self::Blocks(bs) if bs.is_empty())
    }
}

thread_local! {
    // Collection of traced basic blocks.
    static BASIC_BLOCKS: RefCell<SWTBuf> = const { RefCell::new(SWTBuf::Blocks(Vec::new())) };
}

/// Rust's `libc` crate does not expose any of the longjmp-related bits, so we have to define it
/// ourselves. The exact implementation is very platform specific.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
type JmpBuf = [u64; 8];

unsafe extern "C" {
    fn longjmp(env: *mut JmpBuf, val: libc::c_int) -> !;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __wrap_longjmp(env: *mut JmpBuf, val: libc::c_int) -> ! {
    {
        if MTThread::is_tracing() {
            BASIC_BLOCKS.with(|v| {
                let mut v = v.borrow_mut();
                match &mut *v {
                    vv @ SWTBuf::Blocks(_) => {
                        // Invalidate the SWT buffer.
                        *vv = SWTBuf::Invalid("traced longjmp".into());
                    }
                    SWTBuf::Invalid(_) => (), // Buffer already invalid.
                }
                drop(v);
            })
        }
    } // Extra scope to be really sure that everything inside is dropped.

    // In general you have to be careful when using setjmp/longjmp from Rust:
    // https://github.com/rust-lang/rfcs/issues/2625
    //
    // We don't call setjmp from Rust, so those concerns can be shelved.
    //
    // What remains is that, we have to ensure that as a result of longjmp, no Rust drop methods
    // could be skipped, and no borrow checking rules could be violated.
    //
    // Since this is the only Rust frame between now and the C code we will jump to, we only have
    // to reason about this wrapper function. Above I've explicitely dropped anything that needs
    // it, and ensured that no borrows are outstanding at the time of the longjmp.
    unsafe { longjmp(env, val) }
}

/// Inserts LLVM IR basicblock metadata into a thread-local BASIC_BLOCKS vector.
///
/// # Arguments
/// * `function_index` - The index of the function to which the basic block belongs.
/// * `block_index` - The index of the basic block within the function.
#[cfg(tracer_swt)]
#[unsafe(no_mangle)]
pub extern "C" fn __yk_trace_basicblock(function_index: usize, block_index: usize) {
    if MTThread::is_tracing() {
        BASIC_BLOCKS.with(|v| match &mut *v.borrow_mut() {
            SWTBuf::Blocks(bs) => bs.push(TracingBBlock {
                function_index,
                block_index,
            }),
            SWTBuf::Invalid(_) => (),
        })
    }
}

pub(crate) struct SWTracer {}

impl SWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(SWTracer {})
    }
}

impl Tracer for SWTracer {
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>> {
        debug_assert!(BASIC_BLOCKS.with(|bbs| bbs.borrow().is_empty()));
        Ok(Box::new(SWTTraceRecorder {}))
    }
}

#[derive(Debug)]
struct SWTTraceRecorder {}

impl TraceRecorder for SWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
        let bbs = BASIC_BLOCKS.with(|tb| tb.replace(SWTBuf::Blocks(Vec::new())));
        match bbs {
            SWTBuf::Blocks(bs) => {
                if bs.is_empty() {
                    // FIXME: who should handle an empty trace?
                    panic!();
                } else {
                    Ok(Box::new(SWTraceIterator::new(bs)))
                }
            }
            SWTBuf::Invalid(m) => Err(TraceRecorderError::TraceInvalid(m)),
        }
    }
}

struct SWTraceIterator {
    bbs: std::vec::IntoIter<TracingBBlock>,
}

impl SWTraceIterator {
    fn new(bbs: Vec<TracingBBlock>) -> SWTraceIterator {
        SWTraceIterator {
            bbs: bbs.into_iter(),
        }
    }
}

impl Iterator for SWTraceIterator {
    type Item = Result<TraceAction, AOTTraceIteratorError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.bbs.next().map(|tb| {
            Ok(TraceAction::MappedAOTBBlock {
                funcidx: tb.function_index,
                bbidx: tb.block_index,
            })
        })
    }
}

impl AOTTraceIterator for SWTraceIterator {}
