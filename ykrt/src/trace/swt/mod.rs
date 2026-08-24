//! Software tracer.

use super::{
    AOTTraceIterator, AOTTraceIteratorError, TraceAction, TraceRecorder, TraceRecorderError, Tracer,
};
use crate::mt::MTThread;
use std::{cell::UnsafeCell, error::Error, mem, sync::Arc};

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
    static BASIC_BLOCKS: UnsafeCell<Vec<TracingBBlock>> = const { UnsafeCell::new(vec![]) };
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
    BASIC_BLOCKS.with(|v| {
        let v = unsafe { &mut *v.get() };
        v.push(TracingBBlock {
            function_index: u16::try_from(block_id >> 16).unwrap(),
            block_index: u16::try_from(block_id & 0xffff).unwrap(),
        });
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
            let tb = unsafe { &mut *tb.get() };
            mem::replace(tb, Vec::new())
        });
        if bbs.is_empty() {
            // FIXME: who should handle an empty trace?
            panic!();
        } else {
            if bbs.len() > TRACE_TOO_LONG {
                Err(TraceRecorderError::TraceTooLong)
            } else {
                Ok(Box::new(SWTraceIterator::new(bbs)))
            }
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
