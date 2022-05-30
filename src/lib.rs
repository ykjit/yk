#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::new_without_default)]

mod block;
pub use block::Block;
pub mod backends;
pub mod errors;
#[cfg(test)]
mod test_helpers;

pub use errors::HWTracerError;
use std::fmt::Debug;
#[cfg(test)]
use std::fs::File;
use std::iter::Iterator;

/// Represents a generic trace.
///
/// Each backend has its own concrete implementation.
pub trait Trace: Debug + Send {
    /// Iterate over the blocks of the trace.
    fn iter_blocks<'t: 'i, 'i>(
        &'t self,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + 'i>;

    /// Get the capacity of the trace in bytes.
    #[cfg(test)]
    fn capacity(&self) -> usize;

    /// Dump the trace to the specified filename.
    ///
    /// The exact format varies per-backend.
    #[cfg(test)]
    fn to_file(&self, file: &mut File);
}

/// The interface offered by all tracer types.
pub trait Tracer: Send + Sync {
    /// Return a `ThreadTracer` for tracing the current thread.
    fn thread_tracer(&self) -> Box<dyn ThreadTracer>;
}

pub trait ThreadTracer {
    /// Start recording a trace.
    ///
    /// Tracing continues until [stop_tracing](trait.ThreadTracer.html#method.stop_tracing) is called.
    fn start_tracing(&mut self) -> Result<(), HWTracerError>;
    /// Turns off the tracer.
    ///
    /// [start_tracing](trait.ThreadTracer.html#method.start_tracing) must have been called prior.
    fn stop_tracing(&mut self) -> Result<Box<dyn Trace>, HWTracerError>;
}
