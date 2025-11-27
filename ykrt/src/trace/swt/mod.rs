//! Software tracer.

use super::{
    AOTTraceIterator, AOTTraceIteratorError, TraceAction, TraceRecorder, TraceRecorderError, Tracer,
};
use crate::mt::MTThread;
use std::{cell::RefCell, error::Error, sync::Arc};

#[derive(Debug, Eq, PartialEq, Clone)]
struct TracingBBlock {
    function_index: usize,
    block_index: usize,
}

thread_local! {
    // Collection of traced basic blocks.
    static BASIC_BLOCKS: RefCell<Vec<TracingBBlock>> = const { RefCell::new(vec![]) };
}

/// Records the specified basic block into the software tracing buffer.
///
/// This must only be called if the current thread is tracing.
///
/// # Arguments
/// * `function_index` - The index of the function to which the basic block belongs.
/// * `block_index` - The index of the basic block within the function.
#[cfg(tracer_swt)]
#[unsafe(no_mangle)]
pub extern "C" fn __yk_trace_basicblock(function_index: usize, block_index: usize) {
    debug_assert!(MTThread::is_tracing());
    BASIC_BLOCKS.with(|v| {
        v.borrow_mut().push(TracingBBlock {
            function_index,
            block_index,
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
        debug_assert!(BASIC_BLOCKS.with(|bbs| bbs.borrow().is_empty()));
        Ok(Box::new(SWTTraceRecorder {}))
    }
}

#[derive(Debug)]
struct SWTTraceRecorder {}

impl TraceRecorder for SWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
        let bbs = BASIC_BLOCKS.with(|tb| tb.replace(Vec::new()));
        if bbs.is_empty() {
            // FIXME: who should handle an empty trace?
            panic!();
        } else {
            Ok(Box::new(SWTraceIterator::new(bbs)))
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
