//! An LLVM JIT backend. Currently a minimal wrapper around the fact that [MappedAOTBlockTrace]s are hardcoded
//! to be compiled with LLVM.

use crate::{
    compile::{CompilationError, CompiledTrace, Compiler, Guard, GuardId},
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::{AOTTraceIterator, AOTTraceIteratorError, TraceAction},
    ykstats::TimingState,
};
use object::{Object, ObjectSection};
use parking_lot::Mutex;
#[cfg(any(debug_assertions, test))]
use std::error::Error;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(not(test))]
use std::slice;
use std::{
    collections::HashMap,
    env,
    ffi::{c_char, c_int, c_void},
    fmt, ptr,
    sync::{Arc, LazyLock, Weak},
};
use tempfile::NamedTempFile;
use ykaddr::obj::SELF_BIN_MMAP;
use yksmp::LiveVar;
#[cfg(not(test))]
use yksmp::StackMapParser;

mod deopt;
pub(crate) mod frame;
mod llvmbridge;

pub static LLVM_BITCODE: LazyLock<&[u8]> = LazyLock::new(|| {
    let object = object::File::parse(&**SELF_BIN_MMAP).unwrap();
    let sec = object.section_by_name(".llvmbc").unwrap();
    sec.data().unwrap()
});

#[cfg(not(test))]
struct SendSyncConstPtr<T>(*const T);
#[cfg(not(test))]
unsafe impl<T> Send for SendSyncConstPtr<T> {}
#[cfg(not(test))]
unsafe impl<T> Sync for SendSyncConstPtr<T> {}

/// A trace compiled into machine code. Note that these are passed around as raw pointers and
/// potentially referenced by multiple threads so, once created, instances of this struct can only
/// be updated if a lock is held or a field is atomic.
#[cfg(not(test))]
pub(crate) struct LLVMCompiledTrace {
    // Reference to the meta-tracer required for side tracing.
    mt: Arc<MT>,
    /// A function which when called, executes the compiled trace.
    ///
    /// The argument to the function is a pointer to a struct containing the live variables at the
    /// control point. The exact definition of this struct is not known to Rust: the struct is
    /// generated at interpreter compile-time by ykllvm.
    entry: SendSyncConstPtr<c_void>,
    /// Parsed stackmap of this trace. We only need to read this once, and can then use it to
    /// lookup stackmap information for each guard failure as needed.
    smap: HashMap<u64, Vec<LiveVar>>,
    /// Pointer to heap allocated live AOT values.
    aotvals: SendSyncConstPtr<c_void>,
    /// List of guards containing hotness counts and compiled side traces.
    guards: Vec<Guard>,
    /// If requested, a temporary file containing the "source code" for the trace, to be shown in
    /// debuggers when stepping over the JITted code.
    ///
    /// (rustc incorrectly identifies this field as dead code. Although it isn't being "used", the
    /// act of storing it is preventing the deletion of the file via its `Drop`)
    #[allow(dead_code)]
    di_tmpfile: Option<NamedTempFile>,
    /// Reference to the HotLocation, required for side tracing.
    hl: Weak<Mutex<HotLocation>>,
}

#[cfg(not(test))]
impl LLVMCompiledTrace {
    /// Create a `CompiledTrace` from a pointer to an array containing: the pointer to the compiled
    /// trace, the pointer to the stackmap and the size of the stackmap, and the pointer to the
    /// live AOT values. The arguments `mt` and `hl` are required for side-tracing.
    pub(crate) fn new(
        mt: Arc<MT>,
        data: *const c_void,
        di_tmpfile: Option<NamedTempFile>,
        hl: Weak<Mutex<HotLocation>>,
    ) -> Self {
        let slice = unsafe { slice::from_raw_parts(data as *const usize, 5) };
        let funcptr = slice[0] as *const c_void;
        let smptr = slice[1] as *const c_void;
        let smsize = slice[2];
        let aotvals = slice[3] as *mut c_void;
        let guardcount = slice[4];

        // Parse the stackmap of this trace and cache it.
        let smslice = unsafe { slice::from_raw_parts(smptr as *mut u8, smsize) };
        let smap = StackMapParser::parse(smslice).unwrap();

        // We heap allocated this array in yktracec to pass the data here. Now that we've
        // extracted it we no longer need to keep the array around.
        unsafe { libc::free(data as *mut c_void) };
        let mut guards = Vec::new();
        for _ in 0..=guardcount {
            guards.push(Guard {
                failed: 0.into(),
                ct: None.into(),
            });
        }
        Self {
            mt,
            entry: SendSyncConstPtr(funcptr),
            smap,
            aotvals: SendSyncConstPtr(aotvals),
            di_tmpfile,
            guards,
            hl,
        }
    }

    fn smap(&self) -> &HashMap<u64, Vec<LiveVar>> {
        &self.smap
    }
}

#[cfg(not(test))]
impl CompiledTrace for LLVMCompiledTrace {
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
        self
    }

    fn mt(&self) -> &Arc<MT> {
        &self.mt
    }

    /// Return a reference to the guard `id`.
    fn guard(&self, id: GuardId) -> &Guard {
        &self.guards[id.0]
    }

    /// Is the guard `id` the last guard in this `CompiledTrace`?
    fn is_last_guard(&self, id: GuardId) -> bool {
        id.0 + 1 == self.guards.len()
    }

    fn aotvals(&self) -> *const c_void {
        self.aotvals.0
    }

    fn entry(&self) -> *const c_void {
        self.entry.0
    }

    fn hl(&self) -> &Weak<Mutex<HotLocation>> {
        &self.hl
    }

    #[cfg(any(debug_assertions, test))]
    fn disassemble(&self) -> Result<String, Box<dyn Error>> {
        todo!()
    }
}

#[cfg(not(test))]
impl Drop for LLVMCompiledTrace {
    fn drop(&mut self) {
        // The memory holding the AOT live values needs to live as long as the trace. Now that we
        // no longer need the trace, this can be freed too.
        unsafe { libc::free(self.aotvals.0 as *mut c_void) };
        // FIXME: This should drop the JITted code.
    }
}

impl fmt::Debug for LLVMCompiledTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LLVMCompiledTrace {{ ... }}")
    }
}

#[cfg(test)]
pub(crate) struct LLVMCompiledTrace;

#[cfg(test)]
impl LLVMCompiledTrace {
    pub(crate) fn new(
        _mt: Arc<MT>,
        _data: *const c_void,
        _di_tmpfile: Option<NamedTempFile>,
        _hl: Weak<Mutex<HotLocation>>,
    ) -> Self {
        todo!();
    }

    /// Create a `CompiledTrace` suitable for testing purposes. The resulting instance is not
    /// useful other than as a placeholder: calling any of its methods will cause a panic.
    pub(crate) fn new_testing() -> Self {
        Self
    }

    fn smap(&self) -> &HashMap<u64, Vec<LiveVar>> {
        todo!();
    }
}

#[cfg(test)]
impl CompiledTrace for LLVMCompiledTrace {
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
        todo!();
    }

    fn mt(&self) -> &Arc<MT> {
        todo!();
    }

    fn guard(&self, _id: GuardId) -> &Guard {
        todo!();
    }

    fn is_last_guard(&self, _id: GuardId) -> bool {
        todo!();
    }

    fn aotvals(&self) -> *const c_void {
        todo!();
    }

    fn entry(&self) -> *const c_void {
        todo!();
    }

    fn hl(&self) -> &Weak<Mutex<HotLocation>> {
        todo!();
    }

    fn disassemble(&self) -> Result<String, Box<dyn Error>> {
        todo!();
    }
}

pub(crate) struct JITCLLVM;

impl Compiler for JITCLLVM {
    fn compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: (Box<dyn AOTTraceIterator>, Box<[usize]>),
        sti: Option<SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let mut irtrace = Vec::new();
        mt.stats.timing_state(TimingState::TraceMapping);
        for ta in aottrace_iter.0 {
            match ta {
                Ok(x) => irtrace.push(x),
                Err(AOTTraceIteratorError::LongJmpEncountered) => {
                    return Err(CompilationError("Encountered longjmp".to_owned()));
                }
                Err(AOTTraceIteratorError::TraceTooLong) => {
                    return Err(CompilationError("Trace too long".to_owned()))
                }
            }
        }
        mt.stats.timing_state(TimingState::Compiling);
        let (func_names, bbs, trace_len) = self.encode_trace(&irtrace);

        let llvmbc = llvmbc_section();
        let (di_tmp, di_fd, di_tmpname_c) = Self::create_debuginfo_temp_file();

        let (callstack, aotvalsptr, aotvalslen) = match sti {
            Some(sti) => (sti.callstack, sti.aotvalsptr, sti.aotvalslen),
            None => (std::ptr::null(), std::ptr::null(), 0),
        };

        let ret = unsafe {
            yktracec::__yktracec_irtrace_compile(
                func_names.as_ptr(),
                bbs.as_ptr(),
                trace_len,
                llvmbc.as_ptr(),
                u64::try_from(llvmbc.len()).unwrap(),
                di_fd,
                di_tmpname_c,
                callstack,
                aotvalsptr,
                aotvalslen,
                aottrace_iter.1.as_ptr(),
                aottrace_iter.1.len(),
            )
        };
        if ret.is_null() {
            // The LLVM backend is now legacy code and is pending deletion, so it's not worth us
            // spending time auditing all of the failure modes and categorising them into
            // recoverable/temporary. So for now we say any error is temporary.
            Err(CompilationError("llvm backend error".into()))
        } else {
            Ok(Arc::new(LLVMCompiledTrace::new(
                mt,
                ret,
                di_tmp,
                Arc::downgrade(&hl),
            )))
        }
    }
}

impl JITCLLVM {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(JITCLLVM)
    }

    fn encode_trace(&self, irtrace: &Vec<TraceAction>) -> (Vec<*const i8>, Vec<usize>, usize) {
        let trace_len = irtrace.len();
        let mut func_names = Vec::with_capacity(trace_len);
        let mut bbs = Vec::with_capacity(trace_len);
        for blk in irtrace {
            match blk {
                TraceAction::MappedAOTBlock { func_name, bb } => {
                    func_names.push(func_name.as_ptr());
                    bbs.push(*bb);
                }
                TraceAction::UnmappableBlock => {
                    // The block was unmappable. Indicate this with a null function name.
                    func_names.push(ptr::null());
                    // Block indices for unmappable blocks are irrelevant so we may pass anything here.
                    bbs.push(0);
                }
                TraceAction::Promotion => todo!(),
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
}

/// Returns a pointer to (and the size of) the raw LLVM bitcode in the current address space.
pub(crate) fn llvmbc_section() -> &'static [u8] {
    &LLVM_BITCODE
}
