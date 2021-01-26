#![cfg_attr(test, feature(test))]
#![cfg_attr(tracermode = "sw", feature(thread_local))]
#![cfg_attr(tracermode = "sw", feature(core_intrinsics))]
#![cfg_attr(tracermode = "sw", feature(global_asm))]

#[cfg(test)]
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

pub use errors::InvalidTraceError;
use sir::SirTrace;
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
    pub fn stop_tracing(mut self) -> Result<SirTrace, InvalidTraceError> {
        let trace = self.t_impl.stop_tracing();
        if let Ok(inner) = &trace {
            if inner.len() == 0 {
                return Err(InvalidTraceError::EmptyTrace);
            }
        }
        trace
    }
}

// An generic interface which tracing backends must fulfill.
trait ThreadTracerImpl {
    /// Stops tracing on the current thread, returning the SIR trace on success.
    fn stop_tracing(&mut self) -> Result<SirTrace, InvalidTraceError>;
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
