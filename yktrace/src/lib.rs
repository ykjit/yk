//! Utilities for collecting and decoding traces.

mod errors;
mod hwt;
use std::marker::PhantomData;

pub use errors::InvalidTraceError;

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
pub enum TracingKind {
    /// Software tracing.
    SoftwareTracing,
    /// Hardware tracing via a branch tracer (e.g. Intel PT).
    HardwareTracing,
}

impl Default for TracingKind {
    /// Returns the default tracing kind.
    fn default() -> Self {
        // FIXME this should query the hardware for a suitable hardware tracer and failing that
        // fall back on software tracing.
        TracingKind::HardwareTracing
    }
}

/// A globally unique block ID for an LLVM IR block.
pub struct IRBlock {
    /// The name of the function containing the block.
    _func_name: String,
    /// The index of the block within the function.
    _bb: usize,
}

/// An LLVM IR trace.
pub struct IRTrace {
    blocks: Vec<IRBlock>,
}

unsafe impl Send for IRTrace {}
unsafe impl Sync for IRTrace {}

impl IRTrace {
    pub(crate) fn new(blocks: Vec<IRBlock>) -> Self {
        Self { blocks }
    }

    pub(crate) fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// Binary executable trace code.
pub struct CompiledTrace<I> {
    _pd: PhantomData<I>,
}

unsafe impl<I> Send for CompiledTrace<I> {}
unsafe impl<I> Sync for CompiledTrace<I> {}

/// Represents a thread which is currently tracing.
pub struct ThreadTracer {
    /// The tracing implementation.
    t_impl: Box<dyn ThreadTracerImpl>,
}

impl ThreadTracer {
    /// Stops tracing on the current thread, returning a IR trace on success.
    pub fn stop_tracing(mut self) -> Result<IRTrace, InvalidTraceError> {
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
    /// Stops tracing on the current thread, returning the IR trace on success.
    fn stop_tracing(&mut self) -> Result<IRTrace, InvalidTraceError>;
}

/// Start tracing on the current thread using the specified tracing kind.
/// Each thread can have at most one active tracer; calling `start_tracing()` on a thread where
/// there is already an active tracer leads to undefined behaviour.
pub fn start_tracing(kind: TracingKind) -> ThreadTracer {
    match kind {
        TracingKind::SoftwareTracing => todo!(),
        TracingKind::HardwareTracing => hwt::start_tracing(),
    }
}
