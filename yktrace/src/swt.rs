//! Software tracing via ykrustc.

use super::{SirTrace, ThreadTracer, ThreadTracerImpl};
use crate::{errors::InvalidTraceError, SirLoc};
use core::yk::{swt, SirLoc as CoreSirLoc};
use libc;
use std::convert::TryFrom;

/// A trace collected via software tracing.
/// Since the trace is a heap-allocated C buffer, we represent it as a pointer and a length.
#[derive(Debug)]
struct SWTSirTrace {
    locs: Vec<SirLoc>
}

impl SWTSirTrace {
    /// Create a SWTSirTrace from a raw buffer and (element) length.
    ///
    /// `buf` must have been allocated by malloc(3); this function will free(2) it.
    fn from_buf(buf: *const CoreSirLoc, len: usize) -> Self {
        // When we make a SWTSirTrace, we convert all of the locations from core::SirLoc up to
        // crate::SirLoc and store them in self so that we can hand out references via raw_loc().
        let locs = (0..len)
            .map(|idx| {
                let idx = isize::try_from(idx).unwrap();
                SirLoc::from(unsafe { &*buf.offset(idx) })
            })
            .collect();

        unsafe { libc::free(buf as *mut libc::c_void) };
        Self { locs }
    }
}

impl SirTrace for SWTSirTrace {
    fn raw_len(&self) -> usize {
        self.locs.len()
    }

    fn raw_loc(&self, idx: usize) -> &SirLoc {
        &self.locs[idx]
    }
}

/// Softare thread tracer.
struct SWTThreadTracer;

impl ThreadTracerImpl for SWTThreadTracer {
    #[trace_tail]
    fn stop_tracing(&mut self) -> Result<Box<dyn SirTrace>, InvalidTraceError> {
        match swt::stop_tracing() {
            None => Err(InvalidTraceError::InternalError),
            Some((buf, len)) => Ok(Box::new(SWTSirTrace::from_buf(buf, len)) as Box<dyn SirTrace>)
        }
    }
}

#[trace_head]
pub fn start_tracing() -> ThreadTracer {
    swt::start_tracing();
    ThreadTracer {
        t_impl: Box::new(SWTThreadTracer {})
    }
}

#[cfg(test)]
#[cfg(tracermode = "sw")]
mod tests {
    use crate::{test_helpers, TracingKind};

    const TRACING_KIND: TracingKind = TracingKind::SoftwareTracing;

    #[test]
    fn test_trace() {
        test_helpers::test_trace(TRACING_KIND);
    }

    #[test]
    fn test_trace_twice() {
        test_helpers::test_trace_twice(TRACING_KIND);
    }

    #[test]
    fn test_trace_concurrent() {
        test_helpers::test_trace_concurrent(TRACING_KIND);
    }

    #[test]
    #[should_panic]
    fn test_oob_trace_index() {
        test_helpers::test_oob_trace_index(TRACING_KIND);
    }

    #[test]
    fn test_in_bounds_trace_indices() {
        test_helpers::test_in_bounds_trace_indices(TRACING_KIND);
    }

    #[test]
    fn test_trace_iterator() {
        test_helpers::test_trace_iterator(TRACING_KIND);
    }
}
