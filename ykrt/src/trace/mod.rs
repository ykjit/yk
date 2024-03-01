//! Record and process traces.
//!
//! "Tracing" is split into the following stages:
//!
//! 1. *Record* the trace with a [Tracer], which abstracts over a specific *tracer backend*. The
//!    tracer backend may use one of several low-level tracing methods (e.g. a hardware tracer like
//!    PT or a software tracer). The tracer backend stores the recorded low-level trace in an
//!    internal format of its choosing.
//! 2. *Process* the recorded trace. The tracer backend returns an iterator which produces
//!    [TraceAction]s.
//! 3. *Compile* the processed trace. That happens in [compile](crate::compile) module.
//!
//! This module thus contains tracing backends which can record and process traces.

#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::missing_safety_doc)]

mod errors;
use std::{error::Error, ffi::CString, sync::Arc};

#[cfg(tracer_hwt)]
pub(crate) mod hwt;
#[cfg(tracer_swt)]
pub(crate) mod swt;

pub(crate) use errors::InvalidTraceError;

/// A `Tracer` is a front-end to a tracer backend (e.g. hardware or software tracing). The tracer
/// backend may have its own configuration options, which is why `Tracer` does not have a `new`
/// method.
pub(crate) trait Tracer: Send + Sync {
    /// Start recording a trace of the current thread.
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>>;
}

/// Return a [Tracer] instance or `Err` if none can be found. The [Tracer] returned will be
/// selected on a combination of what the platform can support and other (possibly run-time) user
/// configuration.
pub(crate) fn default_tracer() -> Result<Arc<dyn Tracer>, Box<dyn Error>> {
    #[cfg(tracer_hwt)]
    {
        return Ok(Arc::new(hwt::HWTracer::new()?));
    }
    #[cfg(tracer_swt)]
    {
        return Ok(Arc::new(swt::SWTracer::new()?));
    }
    #[allow(unreachable_code)]
    Err("No tracing backend for this platform/configuration.".into())
}

/// An instance of a [Tracer] which is currently recording a trace of the current thread.
pub(crate) trait TraceRecorder {
    /// Stop recording a trace of the current thread and return an iterator which successively
    /// produces [TraceAction]s.
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, InvalidTraceError>;
}

/// An iterator which [TraceRecord]s use to process a trace into [TraceAction]s.
pub(crate) trait AOTTraceIterator:
    Iterator<Item = Result<TraceAction, AOTTraceIteratorError>> + Send
{
}

pub(crate) enum AOTTraceIteratorError {}

/// A processed item from a trace.
#[derive(Debug, Eq, PartialEq)]
pub enum TraceAction {
    /// A sucessfully mapped block.
    MappedAOTBlock {
        /// The name of the function containing the block.
        ///
        /// PERF: Use a string pool to avoid duplicated function names in traces.
        func_name: CString,
        /// The index of the block within the function.
        bb: usize,
    },
    /// One or more machine blocks that could not be mapped.
    ///
    /// This usually means that the blocks were compiled outside of ykllvm.
    UnmappableBlock,
    /// A value promoted and recorded within the low-level trace (e.g. `PTWRITE`). In essence these
    /// are calls to `yk_promote` that have been inlined so that the tracer backend can handle them
    /// rather than being handled by yk's generic run-time support for `yk_promote`.
    ///
    /// While no tracer backend currently uses this variant, it's present to remind us that this a
    /// useful possibility.
    Promotion,
}

impl TraceAction {
    pub fn new_mapped_aot_block(func_name: CString, bb: usize) -> Self {
        // At one point, `bb = usize::MAX` was a special value, but it no longer is. We believe
        // that no part of the code sets/checks for this value, but just in case there is a
        // laggardly part of the code which does so, we've left this `assert` behind to catch it.
        debug_assert_ne!(bb, usize::MAX);
        Self::MappedAOTBlock { func_name, bb }
    }

    pub fn new_unmappable_block() -> Self {
        Self::UnmappableBlock
    }
}

#[cfg(tracer_swt)]
pub(crate) fn trace_basicblock(function_index: usize, block_index: usize) {
    swt::trace_basicblock(function_index, block_index)
}
