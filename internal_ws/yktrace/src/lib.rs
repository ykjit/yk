//! Utilities for collecting and decoding traces.

#![cfg_attr(test, feature(test))]
#![cfg_attr(feature = "trace_sw", feature(thread_local))]
#![cfg_attr(feature = "trace_sw", feature(core_intrinsics))]
#![cfg_attr(feature = "trace_sw", feature(global_asm))]

#[cfg(test)]
extern crate test;

#[macro_use]
extern crate lazy_static;

mod errors;
pub mod sir;
pub mod tir;

#[cfg(feature = "trace_hw")]
mod hwt;
#[cfg(feature = "trace_sw")]
mod swt;

pub use errors::InvalidTraceError;
use sir::SirTrace;

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
pub enum TracingKind {
    /// Software tracing via ykrustc.
    SoftwareTracing,
    /// Hardware tracing via ykrustc + hwtracer.
    HardwareTracing,
}

impl Default for TracingKind {
    /// Returns the default tracing kind.
    fn default() -> Self {
        #[cfg(feature = "trace_hw")]
        return TracingKind::HardwareTracing;
        #[cfg(feature = "trace_sw")]
        return TracingKind::SoftwareTracing;
    }
}

/// Represents a thread which is currently tracing.
pub struct ThreadTracer {
    /// The tracing implementation.
    t_impl: Box<dyn ThreadTracerImpl>,
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
    #[cfg(not(any(doctest, feature = "trace_hw", feature = "trace_sw")))]
    compile_error!("Please compile with `-C tracer=T`, where T is one of 'hw' or 'sw'");

    match kind {
        TracingKind::SoftwareTracing => {
            #[cfg(feature = "trace_hw")]
            panic!("requested software tracing, but `-C tracer=hw`");
            #[cfg(feature = "trace_sw")]
            swt::start_tracing()
        }
        TracingKind::HardwareTracing => {
            #[cfg(feature = "trace_sw")]
            panic!("requested hardware tracing, but `-C tracer=sw`");
            #[cfg(feature = "trace_hw")]
            hwt::start_tracing()
        }
    }
}
