// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Software tracing via ykrustc.

use super::{MirTrace, ThreadTracerImpl, ThreadTracer};
use core::yk_swt::{self, MirLoc};
use libc;
use std::ops::Drop;

/// A trace collected via software tracing.
/// Since the trace is a heap-allocated C buffer, we represent it as a pointer and a length.
struct SWTMirTrace {
    buf: *mut MirLoc,
    len: usize
}

impl MirTrace for SWTMirTrace {
    fn len(&self) -> usize {
        self.len
    }

    fn loc(&self, idx: usize) -> &MirLoc {
        assert!(idx < self.len, "out of bounds index");
        unsafe { &*self.buf.add(idx) }
    }
}

impl Drop for SWTMirTrace {
    fn drop(&mut self) {
        unsafe { libc::free(self.buf as *mut libc::c_void) };
    }
}

/// Softare thread tracer.
struct SWTThreadTracer;

impl ThreadTracerImpl for SWTThreadTracer {
    fn stop_tracing(&self) -> Option<Box<dyn MirTrace>> {
        yk_swt::stop_tracing()
            .map(|(buf, len)| Box::new(SWTMirTrace { buf, len }) as Box<dyn MirTrace>)
    }
}

pub fn start_tracing() -> ThreadTracer {
    yk_swt::start_tracing();
    ThreadTracer{t_impl: Box::new(SWTThreadTracer {})}
}

#[cfg(test)]
mod tests {
    use crate::{test_helpers, TracingKind};
    use core::yk_swt;

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
        test_helpers::test_trace_invalidated(TRACING_KIND, yk_swt::invalidate_trace);
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
