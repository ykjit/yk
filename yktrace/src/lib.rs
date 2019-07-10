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

use core::yk_swt::SirLoc;
use std::iter::{IntoIterator, Iterator};
#[macro_use]
extern crate lazy_static;

mod swt;
pub mod tir;

use tir::TirTrace;

/// Generic representation of a trace of SIR block locations.
pub trait SirTrace {
    /// Returns the length of the trace, measured in SIR locations.
    fn len(&self) -> usize;
    /// Returns the SIR location at index `idx`.
    fn loc(&self, idx: usize) -> &SirLoc;
}

/// An iterator over a SIR trace.
pub struct SirTraceIntoIter<'a> {
    trace: &'a dyn SirTrace,
    next_idx: usize
}

impl<'a> IntoIterator for &'a dyn SirTrace {
    type Item = &'a SirLoc;
    type IntoIter = SirTraceIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SirTraceIntoIter {
            trace: self,
            next_idx: 0
        }
    }
}

impl<'a> Iterator for SirTraceIntoIter<'a> {
    type Item = &'a SirLoc;

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
pub struct ThreadTracer {
    /// The tracing implementation.
    t_impl: Box<dyn ThreadTracerImpl>
}

impl ThreadTracer {
    /// Stops tracing on the current thread, returning a TIR trace on success. Returns `None` if
    /// the trace was invalidated.
    pub fn stop_tracing(self) -> Option<TirTrace> {
        self.t_impl
            .stop_tracing()
            .map(|mir_trace| TirTrace::new(&*mir_trace).ok().unwrap())
    }
}

// An generic interface which tracing backends must fulfill.
trait ThreadTracerImpl {
    /// Stops tracing on the current thread, returning the SIR trace on success. Returns `None` is
    /// if the trace was invalidated.
    fn stop_tracing(&self) -> Option<Box<dyn SirTrace>>;
}

/// Start tracing on the current thread using the specified tracing kind.
/// If `None` is passed, then an appropriate tracing kind will be selected; by passing `Some(...)`,
/// a specific kind can be chosen. Any given thread can at most one active tracer; calling
/// `start_tracing()` on a thread where there is already an active tracer leads to undefined
/// behaviour.
pub fn start_tracing(kind: Option<TracingKind>) -> ThreadTracer {
    match kind {
        None | Some(TracingKind::SoftwareTracing) => swt::start_tracing(),
        _ => unimplemented!("tracing kind not implemented")
    }
}

/// The bodies of tests that we want to run on all tracing kinds live in here.
#[cfg(test)]
mod test_helpers {
    use super::{start_tracing, TracingKind};
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
        let trace = th.t_impl.stop_tracing().unwrap();
        assert!(trace.len() > 0);
    }

    /// Test that tracing twice sequentially in the same thread works.
    pub(crate) fn test_trace_twice(kind: TracingKind) {
        let th1 = start_tracing(Some(kind));
        black_box(work(100));
        let trace1 = th1.t_impl.stop_tracing().unwrap();

        let th2 = start_tracing(Some(kind));
        black_box(work(1000));
        let trace2 = th2.t_impl.stop_tracing().unwrap();

        assert!(trace1.len() < trace2.len());
    }

    /// Test that tracing in different threads works.
    pub(crate) fn test_trace_concurrent(kind: TracingKind) {
        let thr = thread::spawn(move || {
            let th1 = start_tracing(Some(kind));
            black_box(work(100));
            th1.t_impl.stop_tracing().unwrap().len()
        });

        let th2 = start_tracing(Some(kind));
        black_box(work(1000));
        let len2 = th2.t_impl.stop_tracing().unwrap().len();

        let len1 = thr.join().unwrap();

        assert!(len1 < len2);
    }

    /// Test that invalidating a trace works.
    /// `inv_fn` is a backend-specific function that invalidates the current thread's trace.
    pub(crate) fn test_trace_invalidated(kind: TracingKind, inv_fn: fn()) {
        let th = start_tracing(Some(kind));
        black_box(work(100));
        inv_fn();
        let trace = th.t_impl.stop_tracing();

        assert!(trace.is_none());
    }

    /// Test that accessing an out of bounds index fails.
    /// Tests calling this should be marked `#[should_panic]`.
    pub(crate) fn test_oob_trace_index(kind: TracingKind) {
        // Construct a really short trace.
        let th = start_tracing(Some(kind));
        let trace = th.t_impl.stop_tracing().unwrap();
        trace.loc(100000);
    }

    /// Test that accessing locations 0 through trace.len() -1 does not panic.
    pub(crate) fn test_in_bounds_trace_indices(kind: TracingKind) {
        // Construct a really short trace.
        let th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.t_impl.stop_tracing().unwrap();

        for i in 0..trace.len() {
            trace.loc(i);
        }
    }

    /// Test iteration over a trace.
    pub(crate) fn test_trace_iterator(kind: TracingKind) {
        let th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.t_impl.stop_tracing().unwrap();

        let mut num_elems = 0;
        for _ in trace.as_ref() {
            num_elems += 1;
        }

        assert_eq!(num_elems, trace.len());
    }
}
