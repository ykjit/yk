//! Software tracer.

use super::{
    AOTTraceIterator, AOTTraceIteratorError, TraceAction, TraceRecorder, TraceRecorderError, Tracer,
};
use crate::mt::MTThread;
use std::{
    cell::UnsafeCell,
    error::Error,
    mem::{self, MaybeUninit},
    sync::Arc,
};

/// Traces with more than this many items will be turned into [TraceRecorderError::TraceTooLong].
static TRACE_TOO_LONG: usize = 15000;

#[derive(Debug, Eq, PartialEq, Clone)]
struct TracingBBlock {
    function_index: u16,
    block_index: u16,
}

thread_local! {
    // Collection of traced basic blocks. Because this is only accessed in this module, it's
    // relatively easy for us to reason about the safety of the [UnsafeCell].
    static BASIC_BLOCKS: UnsafeCell<BasicBlocks> = UnsafeCell::new(BasicBlocks::new());
}

/// Records the specified basic block into the software tracing buffer.
///
/// This must only be called if the current thread is tracing.
///
/// # Arguments
///
/// * `block_id` specifies the block to be recorded. The upper 16-bits are the function index, the
///   lower 16-bits are the basic block index.
#[cfg(tracer_swt)]
#[unsafe(no_mangle)]
pub extern "C" fn __yk_trace_basicblock(block_id: u32) {
    debug_assert!(MTThread::is_tracing());
    BASIC_BLOCKS.with(|bbs| {
        let bbs = unsafe { &mut *bbs.get() };
        bbs.push(block_id);
    })
}

pub(crate) struct SWTracer {}

impl SWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(SWTracer {})
    }
}

impl Tracer for SWTracer {
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>> {
        debug_assert!(BASIC_BLOCKS.with(|bbs| unsafe { &*bbs.get() }.is_empty()));
        Ok(Box::new(SWTTraceRecorder {}))
    }
}

#[derive(Debug)]
struct SWTTraceRecorder {}

impl TraceRecorder for SWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
        let bbs = BASIC_BLOCKS.with(|tb| {
            unsafe { &mut *tb.get() }.extract()
        });
        match bbs {
            Some(x) => {
                assert!(!x.is_empty()); // FIXME: who should handle an empty trace?
                Ok(Box::new(SWTraceIterator::new(x)))
            }
            None => Err(TraceRecorderError::TraceTooLong),
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
                funcidx: usize::from(tb.function_index),
                bbidx: usize::from(tb.block_index),
            })
        })
    }
}

impl AOTTraceIterator for SWTraceIterator {}

/// A thread-local buffer of basic blocks being gathered during tracing. Internally this reuses the
/// same buffer for each trace in a given thread, so no allocation is needed to start tracing,
/// no resizing happens during tracing, and at completion a single allocation of a "perfectly
/// sized" [Vec] can be made.
struct BasicBlocks {
    len: usize,
    /// The safety property we use on this allocation is that we never write elements when `len >=
    /// TRACE_TOO_LONG` (indeed, we cap `len` at `TRACE_TOO_LONG`).
    data: Box<[MaybeUninit<u32>]>,
}

impl BasicBlocks {
    fn new() -> Self {
        Self {
            len: 0,
            data: Box::new_uninit_slice(TRACE_TOO_LONG),
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push a [TracingBBlock] if there is capacity for it.
    fn push(&mut self, value: u32) {
        if self.len < TRACE_TOO_LONG {
            unsafe {
                self.data
                    .as_mut_ptr()
                    .add(self.len)
                    .write(MaybeUninit::new(value));
            }
            self.len += 1;
        }
    }

    /// Return the blocks recorded as a `Vec` or `None` if the trace was too long.
    fn extract(&mut self) -> Option<Vec<TracingBBlock>> {
        let len = mem::replace(&mut self.len, 0);
        if len < TRACE_TOO_LONG {
            let mut v = Vec::with_capacity(len);
            let slice =
                unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast::<u32>(), len) };
            v.extend(slice.iter().map(|block_id| {
                let block_id = *block_id;
                TracingBBlock {
                    function_index: u16::try_from(block_id >> 16).unwrap(),
                    block_index: u16::try_from(block_id & 0xffff).unwrap(),
                }
            }));
            Some(v)
        } else {
            None
        }
    }
}
