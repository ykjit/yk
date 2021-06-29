//! Utilities for collecting and decoding traces.

mod errors;
use libc::c_void;
use std::{
    ffi::{CStr, CString},
    ptr,
};
mod hwt;

pub use errors::InvalidTraceError;
pub use hwt::mapper::BlockMap;

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
#[derive(Debug, Eq, PartialEq)]
pub struct IRBlock {
    /// The name of the function containing the block.
    func_name: CString,
    /// The index of the block within the function.
    bb: usize,
}

impl IRBlock {
    pub fn func_name(&self) -> &CStr {
        &self.func_name.as_c_str()
    }

    pub fn bb(&self) -> usize {
        self.bb
    }
}

/// An LLVM IR trace.
pub struct IRTrace {
    // The blocks of the trace. A None element represents an arbitrarily sized portion of unknown
    // code which we were unable to determine LLVM IR for, e.g. external library code.
    //
    // PERF: Use a special value instead of a None to save memory.
    blocks: Vec<Option<IRBlock>>,
}

unsafe impl Send for IRTrace {}
unsafe impl Sync for IRTrace {}

impl IRTrace {
    pub(crate) fn new(blocks: Vec<Option<IRBlock>>) -> Self {
        Self { blocks }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    // Get the block at the specified position.
    // Returns None if there is no block at this index. Returns Some(None) if the block at that
    // index couldn't be mapped.
    pub fn get(&self, idx: usize) -> Option<&Option<IRBlock>> {
        self.blocks.get(idx)
    }

    pub fn compile(&self) -> *const c_void {
        let len = self.len();
        let mut func_names = Vec::with_capacity(len);
        let mut bbs = Vec::with_capacity(len);
        for blk in &self.blocks {
            if let Some(blk) = blk {
                func_names.push(blk.func_name().as_ptr());
                bbs.push(blk.bb());
            } else {
                // The block was unmappable. Indicate this with a null function name.
                func_names.push(ptr::null());
                bbs.push(0);
            }
        }

        let ret = unsafe {
            ykllvmwrap::__ykllvmwrap_irtrace_compile(func_names.as_ptr(), bbs.as_ptr(), len)
        };
        assert_ne!(ret, ptr::null());
        ret
    }
}

/// Binary executable trace code.
pub struct CompiledTrace {}

unsafe impl Send for CompiledTrace {}
unsafe impl Sync for CompiledTrace {}

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
