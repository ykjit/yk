#![feature(yk_swt)]
#![feature(test)]

extern crate test;

use core::yk::SirLoc as CoreSirLoc;
use std::{fmt::Debug, iter::Iterator};
#[macro_use]
extern crate lazy_static;

pub mod debug;
mod errors;
mod hwt;
// FIXME: Software tracing is currently broken. Not just here, but in ykrustc too.
//mod swt;
pub mod tir;

use errors::InvalidTraceError;
use tir::SIR;

/// The same as core::SirLoc, just with a String representation of the symbol name and with the
/// traits we were disallowed from using in libcore.
#[derive(Debug, Hash, Eq, PartialEq)]
pub struct SirLoc {
    pub symbol_name: String,
    pub bb_idx: u32
}

impl From<&CoreSirLoc> for SirLoc {
    fn from(core_loc: &CoreSirLoc) -> SirLoc {
        SirLoc {
            symbol_name: String::from_utf8(core_loc.symbol_name().to_vec()).unwrap(),
            bb_idx: core_loc.bb_idx()
        }
    }
}

impl SirLoc {
    fn new(symbol_name: String, bb_idx: u32) -> Self {
        Self {
            symbol_name,
            bb_idx
        }
    }
}

/// Generic representation of a trace of SIR block locations.
pub trait SirTrace: Debug {
    /// Returns the length of the *raw* (untrimmed) trace, measured in SIR locations.
    fn raw_len(&self) -> usize;

    /// Returns the SIR location at index `idx` in the *raw* (untrimmed) trace.
    fn raw_loc(&self, idx: usize) -> &SirLoc;
}

impl<'a> IntoIterator for &'a dyn SirTrace {
    type Item = &'a SirLoc;
    type IntoIter = SirTraceIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SirTraceIterator::new(self)
    }
}

/// An iterator over a trimmed SIR trace.
pub struct SirTraceIterator<'a> {
    trace: &'a dyn SirTrace,
    next_idx: usize
}

impl<'a> SirTraceIterator<'a> {
    fn new(trace: &'a dyn SirTrace) -> Self {
        // We are going to present a "trimmed trace", so we do a backwards scan looking for the end
        // of the code that starts the tracer.
        let mut begin_idx = None;
        for blk_idx in (0..trace.raw_len()).rev() {
            let sym = &trace.raw_loc(blk_idx).symbol_name;
            if SIR.markers.trace_heads.contains(sym) {
                begin_idx = Some(blk_idx + 1);
                break;
            }
        }

        SirTraceIterator {
            trace,
            next_idx: begin_idx.expect("Couldn't find the end of the code that starts the tracer")
        }
    }
}

impl<'a> Iterator for SirTraceIterator<'a> {
    type Item = &'a SirLoc;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx < self.trace.raw_len() {
            let sym = &self.trace.raw_loc(self.next_idx).symbol_name;
            if SIR.markers.trace_tails.contains(sym) {
                // Stop when we find the start of the code that stops the tracer, thus trimming the
                // end of the trace. By setting the next index to one above the last one in the
                // trace, we ensure the iterator will return `None` forever more.
                self.next_idx = self.trace.raw_len();
                None
            } else {
                let ret = self.trace.raw_loc(self.next_idx);
                self.next_idx += 1;
                Some(ret)
            }
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
    /// Stops tracing on the current thread, returning a TIR trace on success.
    #[trace_tail]
    pub fn stop_tracing(mut self) -> Result<Box<dyn SirTrace>, InvalidTraceError> {
        self.t_impl.stop_tracing()
    }
}

// An generic interface which tracing backends must fulfill.
trait ThreadTracerImpl {
    /// Stops tracing on the current thread, returning the SIR trace on success.
    #[trace_tail]
    fn stop_tracing(&mut self) -> Result<Box<dyn SirTrace>, InvalidTraceError>;
}

/// Start tracing on the current thread using the specified tracing kind.
/// If `None` is passed, then an appropriate tracing kind will be selected; by passing `Some(...)`,
/// a specific kind can be chosen. Any given thread can at most one active tracer; calling
/// `start_tracing()` on a thread where there is already an active tracer leads to undefined
/// behaviour.
#[trace_head]
pub fn start_tracing(kind: Option<TracingKind>) -> ThreadTracer {
    match kind {
        Some(TracingKind::SoftwareTracing) => unimplemented!("software tracing is broken"),
        None | Some(TracingKind::HardwareTracing) => hwt::start_tracing()
    }
}

/// The bodies of tests that we want to run on all tracing kinds live in here.
#[cfg(test)]
mod test_helpers {
    use super::{start_tracing, SirLoc, TracingKind};
    use crate::tir::SIR;
    use std::thread;
    use test::black_box;
    use ykpack::bodyflags;

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
        let mut th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.t_impl.stop_tracing().unwrap();
        assert!(trace.raw_len() > 0);
    }

    /// Test that tracing twice sequentially in the same thread works.
    pub(crate) fn test_trace_twice(kind: TracingKind) {
        let mut th1 = start_tracing(Some(kind));
        black_box(work(100));
        let trace1 = th1.t_impl.stop_tracing().unwrap();

        let mut th2 = start_tracing(Some(kind));
        black_box(work(1000));
        let trace2 = th2.t_impl.stop_tracing().unwrap();

        assert!(trace1.raw_len() < trace2.raw_len());
    }

    /// Test that tracing in different threads works.
    pub(crate) fn test_trace_concurrent(kind: TracingKind) {
        let thr = thread::spawn(move || {
            let mut th1 = start_tracing(Some(kind));
            black_box(work(100));
            th1.t_impl.stop_tracing().unwrap().raw_len()
        });

        let mut th2 = start_tracing(Some(kind));
        black_box(work(1000));
        let len2 = th2.t_impl.stop_tracing().unwrap().raw_len();

        let len1 = thr.join().unwrap();

        assert!(len1 < len2);
    }

    /// Test that accessing an out of bounds index fails.
    /// Tests calling this should be marked `#[should_panic]`.
    pub(crate) fn test_oob_trace_index(kind: TracingKind) {
        // Construct a really short trace.
        let mut th = start_tracing(Some(kind));
        let trace = th.t_impl.stop_tracing().unwrap();
        trace.raw_loc(100000);
    }

    /// Test that accessing locations 0 through trace.raw_len() -1 does not panic.
    pub(crate) fn test_in_bounds_trace_indices(kind: TracingKind) {
        // Construct a really short trace.
        let mut th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.t_impl.stop_tracing().unwrap();

        for i in 0..trace.raw_len() {
            trace.raw_loc(i);
        }
    }

    /// Test iteration over a trace.
    pub(crate) fn test_trace_iterator(kind: TracingKind) {
        let mut th = start_tracing(Some(kind));
        black_box(work(100));
        let trace = th.t_impl.stop_tracing().unwrap();
        // The length of the iterator will be shorter due to trimming.
        assert!(trace.into_iter().count() < trace.raw_len());
    }

    #[test]
    fn trim_trace() {
        #[cfg(tracermode = "sw")]
        let tracer = start_tracing(Some(TracingKind::SoftwareTracing));
        #[cfg(tracermode = "hw")]
        let tracer = start_tracing(Some(TracingKind::HardwareTracing));
        work(black_box(100));
        let sir_trace = tracer.stop_tracing().unwrap();

        let contains_tracer_start_stop = |locs: Vec<&SirLoc>| {
            let mut found_start_code = false;
            let mut found_stop_code = false;
            for loc in locs {
                let body = SIR
                    .bodies
                    .get(&loc.symbol_name)
                    .expect("No SIR for the location");

                if body.flags & bodyflags::TRACE_HEAD != 0 {
                    found_start_code = true;
                }
                if body.flags & bodyflags::TRACE_TAIL != 0 {
                    found_stop_code = true;
                }
            }
            (found_start_code, found_stop_code)
        };

        // The raw SIR trace will contain the end of the code which starts tracing, and the start
        // of the code which stops tracing. The trimmed SIR trace will contain neither.
        let raw_locs = (0..(sir_trace.raw_len()))
            .map(|i| sir_trace.raw_loc(i))
            .collect();
        assert_eq!(contains_tracer_start_stop(raw_locs), (true, true));

        let trimmed_locs = sir_trace.into_iter().collect();
        assert_eq!(contains_tracer_start_stop(trimmed_locs), (false, false));
    }
}
