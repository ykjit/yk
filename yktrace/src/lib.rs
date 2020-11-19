#![feature(test)]
#![feature(thread_local)]
#![feature(core_intrinsics)]
#![feature(global_asm)]

extern crate test;

#[macro_use]
extern crate lazy_static;

mod errors;
pub mod sir;
pub mod tir;

#[cfg(tracermode = "hw")]
mod hwt;
#[cfg(tracermode = "sw")]
mod swt;

use errors::InvalidTraceError;
use sir::{SirLoc, SirTrace};
use ykpack::Local;

// In TIR traces, the argument to the interp_step is always local #1.
pub const INTERP_STEP_ARG: Local = Local(1);

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
pub enum TracingKind {
    /// Software tracing via ykrustc.
    SoftwareTracing,
    /// Hardware tracing via ykrustc + hwtracer.
    HardwareTracing
}

impl Default for TracingKind {
    /// Returns the default tracing kind.
    fn default() -> Self {
        #[cfg(tracermode = "hw")]
        return TracingKind::HardwareTracing;
        #[cfg(tracermode = "sw")]
        return TracingKind::SoftwareTracing;
    }
}

/// Represents a thread which is currently tracing.
pub struct ThreadTracer {
    /// The tracing implementation.
    t_impl: Box<dyn ThreadTracerImpl>
}

impl ThreadTracer {
    /// Stops tracing on the current thread, returning a TIR trace on success.
    pub fn stop_tracing(mut self) -> Result<Box<dyn SirTrace>, InvalidTraceError> {
        self.t_impl.stop_tracing()
    }
}

// An generic interface which tracing backends must fulfill.
trait ThreadTracerImpl {
    /// Stops tracing on the current thread, returning the SIR trace on success.
    fn stop_tracing(&mut self) -> Result<Box<dyn SirTrace>, InvalidTraceError>;
}

/// Start tracing on the current thread using the specified tracing kind.
/// Each thread can have at most one active tracer; calling `start_tracing()` on a thread where
/// there is already an active tracer leads to undefined behaviour.
pub fn start_tracing(kind: TracingKind) -> ThreadTracer {
    #[cfg(not(any(doctest, tracermode = "hw", tracermode = "sw")))]
    compile_error!("Please compile with `-C tracer=T`, where T is one of 'hw' or 'sw'");

    match kind {
        TracingKind::SoftwareTracing => {
            #[cfg(tracermode = "hw")]
            panic!("requested software tracing, but `-C tracer=hw`");
            #[cfg(tracermode = "sw")]
            swt::start_tracing()
        }
        TracingKind::HardwareTracing => {
            #[cfg(tracermode = "sw")]
            panic!("requested hardware tracing, but `-C tracer=sw`");
            #[cfg(tracermode = "hw")]
            hwt::start_tracing()
        }
    }
}

/// A debugging aid for traces.
/// Calls to this function are recognised by Yorick and a special debug TIR statement is inserted
/// into the trace. Interpreter writers should compile-time guard calls to this so as to only emit
/// the extra bytecodes when explicitely turned on.
#[inline(never)]
#[trace_debug]
pub fn trace_debug(_msg: &'static str) {}

/// The bodies of tests that we want to run on all tracing kinds live in here.
#[cfg(test)]
mod test_helpers {
    use super::{start_tracing, TracingKind};
    use std::thread;
    use test::black_box;

    // Some work to trace.
    #[interp_step]
    fn work(io: &mut WorkIO) {
        let mut res = 0;
        for i in 0..(io.0) {
            if i % 2 == 0 {
                res += 5;
            } else {
                res += 10 / i;
            }
        }
        println!("{}", res); // prevents the above from being optimised out.
    }

    struct WorkIO(usize);

    /// Test that basic tracing works.
    pub(crate) fn test_trace(kind: TracingKind) {
        let mut th = start_tracing(kind);
        black_box(work(&mut WorkIO(10)));
        let trace = th.t_impl.stop_tracing().unwrap();
        assert!(trace.raw_len() > 0);
    }

    /// Test that tracing twice sequentially in the same thread works.
    pub(crate) fn test_trace_twice(kind: TracingKind) {
        let mut th1 = start_tracing(kind);
        black_box(work(&mut WorkIO(10)));
        let trace1 = th1.t_impl.stop_tracing().unwrap();

        let mut th2 = start_tracing(kind);
        black_box(work(&mut WorkIO(20)));
        let trace2 = th2.t_impl.stop_tracing().unwrap();

        assert!(trace1.raw_len() < trace2.raw_len());
    }

    /// Test that tracing in different threads works.
    pub(crate) fn test_trace_concurrent(kind: TracingKind) {
        let thr = thread::spawn(move || {
            let mut th1 = start_tracing(kind);
            black_box(work(&mut WorkIO(10)));
            th1.t_impl.stop_tracing().unwrap().raw_len()
        });

        let mut th2 = start_tracing(kind);
        black_box(work(&mut WorkIO(20)));
        let len2 = th2.t_impl.stop_tracing().unwrap().raw_len();

        let len1 = thr.join().unwrap();

        assert!(len1 < len2);
    }

    /// Test that accessing an out of bounds index fails.
    /// Tests calling this should be marked `#[should_panic]`.
    pub(crate) fn test_oob_trace_index(kind: TracingKind) {
        // Construct a really short trace.
        let mut th = start_tracing(kind);
        // Empty trace -- no call to an interp_step.
        let trace = th.t_impl.stop_tracing().unwrap();
        trace.raw_loc(100000);
    }

    /// Test that accessing locations 0 through trace.raw_len() -1 does not panic.
    pub(crate) fn test_in_bounds_trace_indices(kind: TracingKind) {
        // Construct a really short trace.
        let mut th = start_tracing(kind);
        black_box(work(&mut WorkIO(10)));
        let trace = th.t_impl.stop_tracing().unwrap();

        for i in 0..trace.raw_len() {
            trace.raw_loc(i);
        }
    }
}
