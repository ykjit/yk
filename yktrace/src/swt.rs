// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Software tracing via ykrustc.

use super::{SirTrace, ThreadTracer, ThreadTracerImpl};
use crate::errors::InvalidTraceError;
use core::yk::{swt, SirLoc};
use libc;
use std::ops::Drop;

/// A trace collected via software tracing.
/// Since the trace is a heap-allocated C buffer, we represent it as a pointer and a length.
#[derive(Debug)]
struct SWTSirTrace {
    buf: *mut SirLoc,
    len: usize
}

impl SirTrace for SWTSirTrace {
    fn raw_len(&self) -> usize {
        self.len
    }

    fn raw_loc(&self, idx: usize) -> &SirLoc {
        assert!(idx < self.len, "out of bounds index");
        unsafe { &*self.buf.add(idx) }
    }
}

impl Drop for SWTSirTrace {
    fn drop(&mut self) {
        unsafe { libc::free(self.buf as *mut libc::c_void) };
    }
}

/// Softare thread tracer.
struct SWTThreadTracer;

impl ThreadTracerImpl for SWTThreadTracer {
    #[trace_tail]
    fn stop_tracing(&self) -> Result<Box<dyn SirTrace>, InvalidTraceError> {
        match swt::stop_tracing() {
            None => Err(InvalidTraceError::InternalError),
            Some((buf, len)) => Ok(Box::new(SWTSirTrace { buf, len }) as Box<dyn SirTrace>)
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
mod tests {
    use crate::{test_helpers, TracingKind};
    use core::yk::swt;

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
    fn test_trace_invalidated() {
        test_helpers::test_trace_invalidated(TRACING_KIND, swt::invalidate_trace);
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
