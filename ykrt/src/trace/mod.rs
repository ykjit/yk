//! Utilities for collecting and decoding traces.

#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::missing_safety_doc)]

mod errors;
use libc::c_void;
use std::{
    collections::HashMap,
    error::Error,
    ffi::{CStr, CString},
    sync::Arc,
};

#[cfg(tracer_hwt)]
pub(crate) mod hwt;

pub(crate) use errors::InvalidTraceError;

/// A tracer is an object which can start / stop collecting traces. It may have its own
/// configuration, but that is dependent on the concrete tracer itself.
pub(crate) trait Tracer: Send + Sync {
    /// Start collecting a trace of the current thread.
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, Box<dyn Error>>;
}

/// Return a [Tracer] instance or `Err` if none can be found. The [Tracer] returned will be
/// selected on a combination of what the platform can support and other (possibly run-time) user
/// configuration.
pub(crate) fn default_tracer() -> Result<Arc<dyn Tracer>, Box<dyn Error>> {
    #[cfg(tracer_hwt)]
    {
        return Ok(Arc::new(hwt::HWTracer::new()?));
    }

    #[allow(unreachable_code)]
    Err("No tracing backend this platform/configuration.".into())
}

/// Represents a thread which is currently tracing.
pub(crate) trait ThreadTracer {
    /// Stop collecting a trace of the current thread.
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn RawTrace>, InvalidTraceError>;
}

/// A raw trace resulting from a tracer.
///
/// Depending on the backend: the raw trace may need considerable processing to convert into basic
/// block addresses; or it may contain those basic block addresses in an easily digestible fashion.
pub(crate) trait RawTrace: Send {
    fn map(self: Box<Self>) -> Result<MappedTrace, InvalidTraceError>;
}

/// A mapped trace of AOT LLVM IR blocks.
pub struct MappedTrace {
    /// The blocks of the trace.
    pub(crate) blocks: Vec<TracedAOTBlock>,
    /// Function addresses discovered dynamically via the trace. symbol-name -> address.
    pub(crate) faddrs: HashMap<CString, *const c_void>,
}

impl MappedTrace {
    pub fn new(blocks: Vec<TracedAOTBlock>, faddrs: HashMap<CString, *const c_void>) -> Self {
        debug_assert!(blocks.len() < usize::MAX);
        Self { blocks, faddrs }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// An AOT LLVM IR block that has been traced at JIT time.
#[derive(Debug, Eq, PartialEq)]
pub enum TracedAOTBlock {
    /// A sucessfully mapped block.
    Mapped {
        /// The name of the function containing the block.
        ///
        /// PERF: Use a string pool to avoid duplicated function names in traces.
        func_name: CString,
        /// The index of the block within the function.
        ///
        /// The special value `usize::MAX` indicates unmappable code.
        bb: usize,
    },
    /// One or more machine blocks that could not be mapped.
    ///
    /// This usually means that the blocks were compiled outside of ykllvm.
    Unmappable {
        /// The change to the stack depth as a result of executing the unmappable region.
        stack_adjust: isize,
    },
}

impl TracedAOTBlock {
    pub fn new_mapped(func_name: CString, bb: usize) -> Self {
        Self::Mapped { func_name, bb }
    }

    pub fn new_unmappable(stack_adjust: isize) -> Self {
        Self::Unmappable { stack_adjust }
    }

    /// If `self` is a mapped block, return the function name, otherwise panic.
    pub fn func_name(&self) -> &CStr {
        if let Self::Mapped { func_name, .. } = self {
            func_name.as_c_str()
        } else {
            panic!();
        }
    }

    /// If `self` is a mapped block, return the basic block index, otherwise panic.
    pub fn bb(&self) -> usize {
        if let Self::Mapped { bb, .. } = self {
            *bb
        } else {
            panic!();
        }
    }

    /// Determines whether `self` represents unmappable code.
    pub fn is_unmappable(&self) -> bool {
        matches!(self, Self::Unmappable { .. })
    }

    /// If `self` is an unmappable region, return the stack adjustment value, otherwise panic.
    pub fn stack_adjust(&self) -> isize {
        if let Self::Unmappable { stack_adjust } = self {
            *stack_adjust
        } else {
            panic!();
        }
    }

    pub fn stack_adjust_mut(&mut self) -> &mut isize {
        if let Self::Unmappable { stack_adjust } = self {
            stack_adjust
        } else {
            panic!();
        }
    }
}
