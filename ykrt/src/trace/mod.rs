//! Utilities for collecting and decoding traces.

#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::missing_safety_doc)]

mod errors;
use libc::c_void;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::{
    collections::HashMap,
    error::Error,
    ffi::{CStr, CString},
    sync::Arc,
};
pub mod hwt;

pub use errors::InvalidTraceError;

/// A globally unique block ID for an LLVM IR block.
#[derive(Debug, Eq, PartialEq)]
pub enum IRBlock {
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

impl IRBlock {
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

/// An LLVM IR trace.
pub struct IRTrace {
    /// The blocks of the trace.
    pub(crate) blocks: Vec<IRBlock>,
    /// Function addresses discovered dynamically via the trace. symbol-name -> address.
    pub(crate) faddrs: HashMap<CString, *const c_void>,
}

impl IRTrace {
    pub fn new(blocks: Vec<IRBlock>, faddrs: HashMap<CString, *const c_void>) -> Self {
        debug_assert!(blocks.len() < usize::MAX);
        Self { blocks, faddrs }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// A tracer is an object which can start / stop collecting traces. It may have its own
/// configuration, but that is dependent on the concrete tracer itself.
pub trait Tracer: Send + Sync {
    /// Start collecting a trace of the current thread.
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, Box<dyn Error>>;
}

/// Represents a thread which is currently tracing.
pub trait ThreadTracer {
    /// Stop collecting a trace of the current thread.
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError>;
}

pub fn default_tracer_for_platform() -> Result<Arc<dyn Tracer>, Box<dyn Error>> {
    Ok(Arc::new(hwt::HWTracer::new()?))
}

pub trait UnmappedTrace: Send {
    fn map(self: Box<Self>, tracer: Arc<dyn Tracer>) -> Result<IRTrace, InvalidTraceError>;
}
