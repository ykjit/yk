use yktrace::sir::{SirTrace, SIR};
use yktrace::tir::TirTrace;
use yktrace::InvalidTraceError;

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum TracingKind {
    /// Software tracing via ykrustc.
    SoftwareTracing = 0,
    /// Hardware tracing via ykrustc + hwtracer.
    HardwareTracing = 1,
}

#[repr(C)]
pub(crate) struct ThreadTracer(yktrace::ThreadTracer);

impl ThreadTracer {
    pub(crate) fn stop_tracing(self) -> Result<SirTrace, InvalidTraceError> {
        self.0.stop_tracing()
    }
}

pub(crate) fn start_tracing(tracing_kind: TracingKind) -> ThreadTracer {
    ThreadTracer(yktrace::start_tracing(match tracing_kind {
        TracingKind::SoftwareTracing => yktrace::TracingKind::SoftwareTracing,
        TracingKind::HardwareTracing => yktrace::TracingKind::HardwareTracing,
    }))
}

#[repr(C)]
pub(crate) struct CompiledTrace<I>(ykcompile::CompiledTrace<I>);

impl<I> CompiledTrace<I> {
    pub(crate) fn ptr(&self) -> *const u8 {
        self.0.ptr()
    }
}

pub(crate) fn compile_trace<T>(sir_trace: SirTrace) -> CompiledTrace<T> {
    let tt = TirTrace::new(&*SIR, &sir_trace).unwrap();
    CompiledTrace(ykcompile::compile_trace(tt))
}
