//! Software tracer.

use super::{
    AOTTraceIterator, AOTTraceIteratorError, TraceAction, TraceRecorder, TraceRecorderError, Tracer,
};
use std::{
    cell::UnsafeCell,
    error::Error,
    mem::{self, MaybeUninit},
    ptr,
    sync::Arc,
};

/// Traces with more than this many items will be turned into [TraceRecorderError::TraceTooLong].
static TRACE_TOO_LONG: usize = 15000;

// This is no_mangle because its name is relied upon by ykllvm (see `BasicBlockTracer.cpp`), and
// it needs to survive linking.
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
#[thread_local]
static mut __yk_trace_buffer: TraceBuffer = TraceBuffer {
    cursor: ptr::null_mut(),
    end: ptr::null_mut(),
};

thread_local! {
    // Collection of traced basic blocks. Because this is only accessed in this module, it's
    // relatively easy for us to reason about the safety of the [UnsafeCell].
    static BASIC_BLOCKS: UnsafeCell<BasicBlocks> = UnsafeCell::new(BasicBlocks::new());
}

#[derive(Debug, Eq, PartialEq, Clone)]
struct TracingBBlock {
    function_index: u16,
    block_index: u16,
}

pub(crate) struct SWTracer {}

impl SWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(SWTracer {})
    }
}

impl Tracer for SWTracer {
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>> {
        BASIC_BLOCKS.with(|bbs| {
            let bbs = unsafe { &mut *bbs.get() };
            let tb = &raw mut __yk_trace_buffer;
            unsafe {
                let start = bbs.data.as_mut_ptr().cast::<u32>();
                debug_assert!((*tb).cursor.is_null() || (*tb).cursor == start);
                (*tb).cursor = start;
                (*tb).end = start.add(TRACE_TOO_LONG);
            }
        });
        Ok(Box::new(SWTTraceRecorder {}))
    }
}

#[derive(Debug)]
struct SWTTraceRecorder {}

impl TraceRecorder for SWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
        let bbs = BASIC_BLOCKS.with(|bbs| unsafe { &*bbs.get() }.extract());
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

/// The struct shared with ykllvm in `BasicBlockTracer.h`.
#[repr(C)]
struct TraceBuffer {
    cursor: *mut u32,
    end: *mut u32,
}

/// A thread-local buffer of basic blocks being gathered during tracing. Internally this reuses the
/// same buffer for each trace in a given thread, so no allocation is needed to start tracing,
/// no resizing happens during tracing, and at completion a single allocation of a "perfectly
/// sized" [Vec] can be made.
struct BasicBlocks {
    /// The safety property we use on this allocation is that the trace-buffer TLS cursor never
    /// advances beyond the end of this allocation.
    data: Box<[MaybeUninit<u32>]>,
}

impl BasicBlocks {
    fn new() -> Self {
        Self {
            data: Box::new_uninit_slice(TRACE_TOO_LONG),
        }
    }

    /// Return the blocks recorded as a `Vec` or `None` if the trace was too long.
    fn extract(&self) -> Option<Vec<TracingBBlock>> {
        let tb = &raw mut __yk_trace_buffer;
        // We continually reuse the same allocation, so put `cursor` back to the start.
        let start = self.data.as_ptr().cast::<u32>().cast_mut();
        let cursor = unsafe { mem::replace(&mut (*tb).cursor, start) };
        let len = unsafe { cursor.offset_from(start) as usize };
        if cursor != unsafe { (*tb).end } {
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
