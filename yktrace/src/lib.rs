//! Utilities for collecting and decoding traces.

#![feature(lazy_cell)]
#![feature(naked_functions)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::missing_safety_doc)]

mod errors;
use hwtracer::decode::TraceDecoderKind;
use libc::c_void;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    error::Error,
    ffi::{c_char, c_int, CStr, CString},
    ptr,
};
pub mod hwt;
use std::arch::asm;
use tempfile::NamedTempFile;
use ykutil::obj::llvmbc_section;

pub use errors::InvalidTraceError;

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
                // The block was unmappable. Indicate this with a null function name and the block
                // index encodes the stack adjustment value.
                func_names.push(ptr::null());
                // Subtle cast from `isize` to `usize`. `as` is used deliberately here to preserve
                // the exact bit pattern. The consumer on the other side of the FFI knows to
                // reverse this.
                bbs.push(blk.stack_adjust() as usize);
            } else {
                func_names.push(blk.func_name().as_ptr());
                bbs.push(blk.bb());
            }
        }
        (func_names, bbs, trace_len)
    }

    // If necessary, create a temporary file for us to write the trace's debugging "source code"
    // into. Elsewhere, the JIT module will have `DebugLoc`s inserted into it which will point to
    // lines in this temporary file.
    //
    // If the `YKD_TRACE_DEBUGINFO` environment variable is set to "1", then this function returns
    // a `NamedTempFile`, a non-negative file descriptor, and a path to the file.
    //
    // If the `YKD_TRACE_DEBUGINFO` environment variable is *not* set to "1", then no file is
    // created and this function returns `(None, -1, ptr::null())`.
    #[cfg(unix)]
    fn create_debuginfo_temp_file() -> (Option<NamedTempFile>, c_int, *const c_char) {
        let mut di_tmp = None;
        let mut di_fd = -1;
        let mut di_tmpname_c = ptr::null() as *const c_char;
        if let Ok(di_val) = env::var("YKD_TRACE_DEBUGINFO") {
            if di_val == "1" {
                let tmp = NamedTempFile::new().unwrap();
                di_tmpname_c = tmp.path().to_str().unwrap().as_ptr() as *const c_char;
                di_fd = tmp.as_raw_fd();
                di_tmp = Some(tmp);
            }
        }
        (di_tmp, di_fd, di_tmpname_c)
    }

    pub fn compile(&self) -> Result<(*const c_void, Option<NamedTempFile>), Box<dyn Error>> {
        let (func_names, bbs, trace_len) = self.encode_trace();

        let mut faddr_keys = Vec::new();
        let mut faddr_vals = Vec::new();
        for k in self.faddrs.iter() {
            faddr_keys.push(k.0.as_ptr());
            faddr_vals.push(*k.1);
        }

        let (llvmbc_data, llvmbc_len) = llvmbc_section();
        let (di_tmp, di_fd, di_tmpname_c) = Self::create_debuginfo_temp_file();

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
                di_fd,
                di_tmpname_c,
            )
        };
        if ret.is_null() {
            Err("Could not compile trace.".into())
        } else {
            Ok((ret, di_tmp))
        }
    }

    #[cfg(feature = "yk_testing")]
    pub unsafe fn compile_for_tc_tests(&self, llvmbc_data: *const u8, llvmbc_len: u64) {
        let (func_names, bbs, trace_len) = self.encode_trace();
        let (_di_tmp, di_fd, di_tmpname_c) = Self::create_debuginfo_temp_file();

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
            di_fd,
            di_tmpname_c,
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
    /// Pointer to heap allocated live AOT values.
    aotvals: *const c_void,
    /// If requested, a temporary file containing the "source code" for the trace, to be shown in
    /// debuggers when stepping over the JITted code.
    ///
    /// (rustc incorrectly identifies this field as dead code. Although it isn't being "used", the
    /// act of storing it is preventing the deletion of the file via its `Drop`)
    #[allow(dead_code)]
    di_tmpfile: Option<NamedTempFile>,
}

use std::mem;
use std::slice;
impl CompiledTrace {
    /// Create a `CompiledTrace` from a pointer to an array containing: the pointer to the compiled
    /// trace, the pointer to the stackmap and the size of the stackmap, and the pointer to the
    /// live AOT values.
    pub fn new(data: *const c_void, di_tmpfile: Option<NamedTempFile>) -> Self {
        let slice = unsafe { slice::from_raw_parts(data as *const usize, 4) };
        let funcptr = slice[0] as *const c_void;
        let smptr = slice[1] as *const c_void;
        let smsize = slice[2];
        let aotvals = slice[3] as *mut c_void;
        // We heap allocated this array in ykllvmwrap to pass the data here. Now that we've
        // extracted it we no longer need to keep the array around.
        unsafe { libc::free(data as *mut c_void) };
        Self {
            entry: funcptr,
            smptr,
            smsize,
            aotvals,
            di_tmpfile,
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
            aotvals: std::ptr::null() as *const _,
            di_tmpfile: None,
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
    pub extern "C" fn exec(
        &self,
        ctrlp_vars: *mut c_void,
        frameaddr: *mut c_void,
    ) -> *const c_void {
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
    extern "C" fn real_exec(
        &self,
        ctrlp_vars: *mut c_void,
        frameaddr: *mut c_void,
    ) -> *const c_void {
        #[cfg(feature = "yk_testing")]
        assert_ne!(self.entry as *const (), std::ptr::null());
        unsafe {
            let f = mem::transmute::<
                _,
                unsafe extern "C" fn(
                    *mut c_void,
                    *const c_void,
                    usize,
                    *mut c_void,
                    *const c_void,
                ) -> *const c_void,
            >(self.entry);
            f(ctrlp_vars, self.smptr, self.smsize, frameaddr, self.aotvals)
        }
    }
}

impl Drop for CompiledTrace {
    fn drop(&mut self) {
        // The memory holding the AOT live values needs to live as long as the trace. Now that we
        // no longer need the trace, this can be freed too.
        // FIXME: Free the memory for the stackmap which was allocated in ykllvmwrap/memman.cc.
        unsafe { libc::free(self.aotvals as *mut c_void) };
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
    fn map(self: Box<Self>, decoder: TraceDecoderKind) -> Result<IRTrace, InvalidTraceError>;
}
