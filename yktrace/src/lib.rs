// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(yk_swt)]
#![feature(test)]

extern crate test;

use core::yk_swt::MirLoc;
use std::iter::{IntoIterator, Iterator};

mod swt;

/// Generic representation of a trace of MIR block locations.
pub trait MirTrace {
    /// Returns the length of the trace, measured in MIR locations.
    fn len(&self) -> usize;
    /// Returns the MIR location and index `idx`.
    fn loc(&self, idx: usize) -> &MirLoc;
}

/// An iterator over a MIR trace.
pub struct MirTraceIntoIter<'a> {
    trace: &'a MirTrace,
    next_idx: usize
}

impl<'a> IntoIterator for &'a dyn MirTrace {
    type Item = &'a MirLoc;
    type IntoIter = MirTraceIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        MirTraceIntoIter {
            trace: self,
            next_idx: 0
        }
    }
}

impl<'a> Iterator for MirTraceIntoIter<'a> {
    type Item = &'a MirLoc;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx < self.trace.len() {
            let ret = self.trace.loc(self.next_idx);
            self.next_idx += 1;
            Some(ret)
        } else {
            None // No more locations.
        }
    }
}

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
pub enum TracingKind {
    /// Software tracing via ykrustc.
    SoftwareTracing,
    /// Hardware tracing via ykrustc + hwtracer.
    HardwareTracing
}

/// Represents a thread which is currently tracing.
pub trait ThreadTracer {
    /// Stops tracing on the current thread, returning the trace on success. `None` is returned on
    /// error or if the trace was invalidated.
    fn stop_tracing(self: Box<Self>) -> Option<Box<dyn MirTrace>>;
}

/// Start tracing on the current thread using the specified tracing kind.
/// If `None` is passed, then an appropriate tracing kind will be selected; by passing `Some(...)`,
/// a specific kind can be chosen. Any given thread can at most one active tracer; calling
/// `start_tracing()` on a thread where there is already an active tracer leads to undefined
/// behaviour.
pub fn start_tracing(kind: Option<TracingKind>) -> Box<dyn ThreadTracer> {
    match kind {
        None | Some(TracingKind::SoftwareTracing) => swt::start_tracing(),
        _ => unimplemented!("tracing kind not implemented")
    }
}

/// The bodies of tests that we want to run on all tracing kinds live in here.
#[cfg(test)]
mod test_helpers {
    use super::{start_tracing, TracingKind};
    use core;
    use std::thread;
    use test::black_box;

    // Some work to trace.
    fn work(loops: usize) -> usize {
        let mut res = 0;
        for i in 0..loops {
            if i % 2 == 0 {
                res += 5;
            } else {
                res += 10 / i;
            }
        }
        res
    }

    /// Test that basic tracing works.
    pub(crate) fn test_trace(kind: TracingKind) {
        let th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.stop_tracing().unwrap();
        assert!(trace.len() > 0);
    }

    /// Test that tracing twice sequentially in the same thread works.
    pub(crate) fn test_trace_twice(kind: TracingKind) {
        let th1 = start_tracing(Some(kind));
        black_box(work(100));
        let trace1 = th1.stop_tracing().unwrap();

        let th2 = start_tracing(Some(kind));
        black_box(work(1000));
        let trace2 = th2.stop_tracing().unwrap();

        assert!(trace1.len() < trace2.len());
    }

    /// Test that tracing in different threads works.
    pub(crate) fn test_trace_concurrent(kind: TracingKind) {
        let thr = thread::spawn(move || {
            let th1 = start_tracing(Some(kind));
            black_box(work(100));
            th1.stop_tracing().unwrap().len()
        });

        let th2 = start_tracing(Some(kind));
        black_box(work(1000));
        let len2 = th2.stop_tracing().unwrap().len();

        let len1 = thr.join().unwrap();

        assert!(len1 < len2);
    }

    /// Test that invalidating a trace works.
    /// `inv_fn` is a backend-specific function that invalidates the current thread's trace.
    pub(crate) fn test_trace_invalidated(kind: TracingKind, inv_fn: fn()) {
        let th = start_tracing(Some(kind));
        black_box(work(100));
        inv_fn();
        let trace = th.stop_tracing();

        assert!(trace.is_none());
    }

    /// Test that accessing an out of bounds index fails.
    /// Tests calling this should be marked `#[should_panic]`.
    pub(crate) fn test_oob_trace_index(kind: TracingKind) {
        // Construct a really short trace.
        let th = start_tracing(Some(kind));
        let trace = th.stop_tracing().unwrap();
        trace.loc(100000);
    }

    /// Test that accessing locations 0 through trace.len() -1 does not panic.
    pub(crate) fn test_in_bounds_trace_indices(kind: TracingKind) {
        // Construct a really short trace.
        let th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.stop_tracing().unwrap();

        for i in 0..trace.len() {
            trace.loc(i);
        }
    }

    /// Test iteration over a trace.
    pub(crate) fn test_trace_iterator(kind: TracingKind) {
        let th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.stop_tracing().unwrap();

        let mut num_elems = 0;
        for _ in trace.as_ref() {
            num_elems += 1;
        }

        assert_eq!(num_elems, trace.len());
    }
}
