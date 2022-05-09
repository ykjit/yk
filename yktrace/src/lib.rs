//! Utilities for collecting and decoding traces.

#![feature(once_cell)]
#![feature(naked_functions)]
#![allow(clippy::new_without_default)]

mod errors;
use libc::c_void;
use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error,
    ffi::{CStr, CString},
    ptr,
};
mod hwt;
use std::arch::asm;
use ykutil::obj::llvmbc_section;

pub use errors::InvalidTraceError;
pub use hwt::mapper::BlockMap;

thread_local! {
    // When `Some`, contains the `ThreadTracer` for the current thread. When `None`, the current
    // thread is not being traced.
    //
    // We hide the `ThreadTracer` in a thread local (rather than returning it to the consumer of
    // yk). This ensures that the `ThreadTracer` itself cannot appear in traces.
    pub static THREAD_TRACER: RefCell<Option<ThreadTracer>> = const { RefCell::new(None) };
}

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    ///
    /// PERF: Use a string pool to avoid duplicated function names in traces.
    func_name: CString,
    /// The index of the block within the function.
    ///
    /// The special value `usize::MAX` indicates unmappable code.
    bb: usize,
}

impl IRBlock {
    pub fn new(func_name: CString, bb: usize) -> Self {
        Self { func_name, bb }
    }

    pub fn func_name(&self) -> &CStr {
        self.func_name.as_c_str()
    }

    pub fn bb(&self) -> usize {
        self.bb
    }

    /// Returns an IRBlock whose `bb` field indicates unmappable code.
    pub fn unmappable() -> Self {
        Self {
            func_name: CString::new("").unwrap(),
            bb: usize::MAX,
        }
    }

    /// Determines whether `self` represents unmappable code.
    pub fn is_unmappable(&self) -> bool {
        self.bb == usize::MAX
    }
}

/// An LLVM IR trace.
pub struct IRTrace {
    /// The blocks of the trace.
    blocks: Vec<IRBlock>,
    /// Function addresses discovered dynamically via the trace. symbol-name -> address.
    faddrs: HashMap<CString, *const c_void>,
}

unsafe impl Send for IRTrace {}
unsafe impl Sync for IRTrace {}

impl IRTrace {
    pub fn new(blocks: Vec<IRBlock>, faddrs: HashMap<CString, *const c_void>) -> Self {
        debug_assert!(blocks.len() < usize::MAX);
        Self { blocks, faddrs }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    // Get the block at the specified position.
    // Returns None if there is no block at this index. Returns Some(None) if the block at that
    // index couldn't be mapped.
    pub fn get(&self, idx: usize) -> Option<Option<&IRBlock>> {
        // usize::MAX is a reserved index.
        debug_assert_ne!(idx, usize::MAX);
        self.blocks
            .get(idx)
            .map(|b| if b.is_unmappable() { None } else { Some(b) })
    }

    fn encode_trace(&self) -> (Vec<*const i8>, Vec<usize>, usize) {
        let trace_len = self.len();
        let mut func_names = Vec::with_capacity(trace_len);
        let mut bbs = Vec::with_capacity(trace_len);
        for blk in &self.blocks {
            if blk.is_unmappable() {
                // The block was unmappable. Indicate this with a null function name.
                func_names.push(ptr::null());
                bbs.push(0);
            } else {
                func_names.push(blk.func_name().as_ptr());
                bbs.push(blk.bb());
            }
        }
        (func_names, bbs, trace_len)
    }

    pub fn compile(&self) -> Result<*const c_void, Box<dyn Error>> {
        let (func_names, bbs, trace_len) = self.encode_trace();

        let mut faddr_keys = Vec::new();
        let mut faddr_vals = Vec::new();
        for k in self.faddrs.iter() {
            faddr_keys.push(k.0.as_ptr());
            faddr_vals.push(*k.1);
        }

        let (llvmbc_data, llvmbc_len) = llvmbc_section();

        let ret = unsafe {
            ykllvmwrap::__ykllvmwrap_irtrace_compile(
                func_names.as_ptr(),
                bbs.as_ptr(),
                trace_len,
                faddr_keys.as_ptr(),
                faddr_vals.as_ptr(),
                faddr_keys.len(),
                llvmbc_data,
                llvmbc_len,
            )
        };
        assert_ne!(ret, ptr::null());
        Ok(ret)
    }

    #[cfg(feature = "yk_testing")]
    pub unsafe fn compile_for_tc_tests(&self, llvmbc_data: *const u8, llvmbc_len: usize) {
        let (func_names, bbs, trace_len) = self.encode_trace();

        // These would only need to be populated if we were to load the resulting compiled code
        // into the address space, which for trace compiler tests, we don't.
        let faddr_keys = Vec::new();
        let faddr_vals = Vec::new();

        let ret = ykllvmwrap::__ykllvmwrap_irtrace_compile_for_tc_tests(
            func_names.as_ptr(),
            bbs.as_ptr(),
            trace_len,
            faddr_keys.as_ptr(),
            faddr_vals.as_ptr(),
            faddr_keys.len(),
            llvmbc_data,
            llvmbc_len,
        );
        assert_ne!(ret, ptr::null());
    }
}

/// A trace compiled into machine code. Note that these are passed around as raw pointers and
/// potentially referenced by multiple threads so, once created, instances of this struct can only
/// be updated if a lock is held or a field is atomic.
#[derive(Debug)]
pub struct CompiledTrace {
    /// A function which when called, executes the compiled trace.
    ///
    /// The argument to the function is a pointer to a struct containing the live variables at the
    /// control point. The exact definition of this struct is not known to Rust: the struct is
    /// generated at interpreter compile-time by ykllvm.
    entry: *const c_void,
    /// Pointer to the stackmap, required to parse the stackmap during a guard failure.
    smptr: *const c_void,
    /// The stackmaps size.
    smsize: usize,
}

use std::mem;
use std::slice;
impl CompiledTrace {
    /// Create a `CompiledTrace` from a pointer to an array containing: the pointer to the compiled
    /// trace, the pointer to the stackmap, and the size of the stackmap.
    pub fn new(code_ptr: *const c_void) -> Self {
        let slice = unsafe { slice::from_raw_parts(code_ptr as *const usize, 3) };
        let funcptr = slice[0] as *const c_void;
        let smptr = slice[1] as *const c_void;
        let smsize = slice[2] as usize;
        Self {
            entry: funcptr,
            smptr,
            smsize,
        }
    }

    #[cfg(feature = "yk_testing")]
    #[doc(hidden)]
    /// Create a `CompiledTrace` with null contents. This is unsafe and only intended for testing
    /// purposes where a `CompiledTrace` instance is required, but cannot sensibly be constructed
    /// without overwhelming the test. The resulting instance must not be inspected or executed.
    pub unsafe fn new_null() -> Self {
        Self {
            entry: std::ptr::null(),
            smptr: std::ptr::null() as *const _,
            smsize: 0,
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[naked]
    #[no_mangle]
    /// Taking the deoptimisation path during a guard failure causes the epilogue of the compiled
    /// trace to be skipped. This means that used callee-saved registers (CSRs) are not restored.
    /// Until we've figured out how to restore only the used registers we take the sledge hammer
    /// approach and save and restore all CSRs here.
    /// OPT: Find a way to only restore needed registers (ideally right within the deopt code).
    pub extern "C" fn exec(&self, ctrlp_vars: *mut c_void, returnval: *mut c_void) -> u8 {
        unsafe {
            asm!(
                "push rbx",
                "push rsp",
                "push rbp",
                "push r12",
                "push r13",
                "push r14",
                "push r15",
                "call real_exec",
                "pop r15",
                "pop r14",
                "pop r13",
                "pop r12",
                "pop rbp",
                "pop rsp",
                "pop rbx",
                "ret",
                options(noreturn)
            )
        }
    }

    #[no_mangle]
    extern "C" fn real_exec(&self, ctrlp_vars: *mut c_void, returnval: *mut c_void) -> u8 {
        #[cfg(feature = "yk_testing")]
        assert_ne!(self.entry as *const (), std::ptr::null());
        unsafe {
            let f = mem::transmute::<
                _,
                unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> u8,
            >(self.entry);
            f(ctrlp_vars, self.smptr, self.smsize, returnval)
        }
    }
}

unsafe impl Send for CompiledTrace {}
unsafe impl Sync for CompiledTrace {}

/// Represents a thread which is currently tracing.
pub struct ThreadTracer {
    /// The tracing implementation.
    t_impl: Box<dyn ThreadTracerImpl>,
}

impl ThreadTracer {
    /// Stops tracing on the current thread, returning a IR trace on success.
    pub fn stop_tracing(mut self) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
        self.t_impl.stop_tracing()
    }
}

// An generic interface which tracing backends must fulfill.
trait ThreadTracerImpl {
    /// Stops tracing on the current thread, returning the IR trace on success.
    fn stop_tracing(&mut self) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError>;
}

/// Start tracing on the current thread using the specified tracing kind.
/// Each thread can have at most one active tracer; calling `start_tracing()` on a thread where
/// there is already an active tracer leads to undefined behaviour.
pub fn start_tracing(kind: TracingKind) {
    let tt = match kind {
        TracingKind::SoftwareTracing => todo!(),
        TracingKind::HardwareTracing => hwt::start_tracing(),
    };
    THREAD_TRACER.with(|tl| *tl.borrow_mut() = Some(tt));
}

/// Stop tracing on the current thread. Calling this when the current thread is not already tracing
/// leads to undefined behaviour.
pub fn stop_tracing() -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
    let mut res = Err(InvalidTraceError::EmptyTrace);
    THREAD_TRACER.with(|tt| {
        let tt_owned = tt.borrow_mut().take();
        res = tt_owned.unwrap().stop_tracing();
    });
    res
}

pub trait UnmappedTrace: Send {
    fn map(self: Box<Self>) -> Result<IRTrace, InvalidTraceError>;
}
