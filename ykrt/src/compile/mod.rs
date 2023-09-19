use crate::{
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::MappedTrace,
};
use libc::c_void;
use parking_lot::Mutex;
#[cfg(not(test))]
use std::slice;
use std::{
    collections::HashMap,
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Weak,
    },
};
use tempfile::NamedTempFile;
use yksmp::LiveVar;
#[cfg(not(test))]
use yksmp::StackMapParser;

#[cfg(jitc_llvm)]
pub(crate) mod jitc_llvm;

/// The trait that every JIT compiler backend must implement.
pub(crate) trait Compiler: Send + Sync {
    /// Compile an [MappedTrace] into machine code.
    fn compile(
        &self,
        mt: Arc<MT>,
        irtrace: MappedTrace,
        sti: Option<SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<CompiledTrace, Box<dyn Error>>;

    #[cfg(feature = "yk_testing")]
    unsafe fn compile_for_tc_tests(
        &self,
        irtrace: MappedTrace,
        llvmbc_data: *const u8,
        llvmbc_len: u64,
    );
}

pub(crate) fn default_compiler() -> Result<Arc<dyn Compiler>, Box<dyn Error>> {
    #[cfg(jitc_llvm)]
    {
        return Ok(jitc_llvm::JITCLLVM::new()?);
    }

    #[allow(unreachable_code)]
    Err("No JIT compiler supported on this platform/configuration.".into())
}

#[cfg(feature = "yk_testing")]
pub unsafe fn compile_for_tc_tests(irtrace: MappedTrace, llvmbc_data: *const u8, llvmbc_len: u64) {
    default_compiler()
        .unwrap()
        .compile_for_tc_tests(irtrace, llvmbc_data, llvmbc_len);
}

struct SendSyncConstPtr<T>(*const T);
unsafe impl<T> Send for SendSyncConstPtr<T> {}
unsafe impl<T> Sync for SendSyncConstPtr<T> {}

/// Responsible for tracking how often a guard in a `CompiledTrace` fails. A hotness counter is
/// incremented each time the matching guard failure in a `CompiledTrace` is triggered. Also stores
/// the side-trace once its compiled.
pub(crate) struct Guard {
    failed: AtomicU32,
    ct: Mutex<Option<Arc<CompiledTrace>>>,
}

impl Guard {
    /// Increments the guard failure counter.
    pub fn inc(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the current guard failure counter.
    pub fn failcount(&self) -> u32 {
        self.failed.load(Ordering::Relaxed)
    }

    /// Stores a compiled side-trace inside this guard.
    pub fn setct(&self, ct: Arc<CompiledTrace>) {
        let _ = self.ct.lock().insert(ct);
    }

    /// Retrieves the stored side-trace or None, if no side-trace has been compiled yet.
    pub fn getct(&self) -> Option<Arc<CompiledTrace>> {
        self.ct.lock().as_ref().map(|x| Arc::clone(x))
    }
}

/// A trace compiled into machine code. Note that these are passed around as raw pointers and
/// potentially referenced by multiple threads so, once created, instances of this struct can only
/// be updated if a lock is held or a field is atomic.
pub(crate) struct CompiledTrace {
    // Reference to the meta-tracer required for side tracing.
    #[cfg(not(test))]
    mt: Arc<MT>,
    /// A function which when called, executes the compiled trace.
    ///
    /// The argument to the function is a pointer to a struct containing the live variables at the
    /// control point. The exact definition of this struct is not known to Rust: the struct is
    /// generated at interpreter compile-time by ykllvm.
    #[cfg(not(test))]
    entry: SendSyncConstPtr<c_void>,
    /// Parsed stackmap of this trace. We only need to read this once, and can then use it to
    /// lookup stackmap information for each guard failure as needed.
    #[cfg(not(test))]
    smap: HashMap<u64, Vec<LiveVar>>,
    /// Pointer to heap allocated live AOT values.
    aotvals: SendSyncConstPtr<c_void>,
    /// List of guards containing hotness counts and compiled side traces.
    pub(crate) guards: Vec<Guard>,
    /// If requested, a temporary file containing the "source code" for the trace, to be shown in
    /// debuggers when stepping over the JITted code.
    ///
    /// (rustc incorrectly identifies this field as dead code. Although it isn't being "used", the
    /// act of storing it is preventing the deletion of the file via its `Drop`)
    #[allow(dead_code)]
    di_tmpfile: Option<NamedTempFile>,
    /// Reference to the HotLocation, required for side tracing.
    pub(crate) hl: Weak<Mutex<HotLocation>>,
}

#[cfg(not(test))]
impl CompiledTrace {
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
        let guardcount = slice[4] as usize;

        // Parse the stackmap of this trace and cache it. The original data allocated by memman.cc
        // is now no longer needed and can be freed.
        let smslice = unsafe { slice::from_raw_parts(smptr as *mut u8, smsize) };
        let smap = StackMapParser::parse(smslice).unwrap();
        unsafe { libc::munmap(smptr as *mut c_void, smsize) };

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

    pub(crate) fn mt(&self) -> &Arc<MT> {
        &self.mt
    }

    pub(crate) fn smap(&self) -> &HashMap<u64, Vec<LiveVar>> {
        &self.smap
    }

    pub(crate) fn aotvals(&self) -> *const c_void {
        self.aotvals.0
    }

    pub(crate) fn entry(&self) -> *const c_void {
        self.entry.0
    }
}

#[cfg(test)]
impl CompiledTrace {
    pub(crate) fn new(
        _mt: Arc<MT>,
        _data: *const c_void,
        _di_tmpfile: Option<NamedTempFile>,
        _hl: Weak<Mutex<HotLocation>>,
    ) -> Self {
        todo!();
    }

    /// Create a `CompiledTrace` with null contents. This is unsafe and only intended for testing
    /// purposes where a `CompiledTrace` instance is required, but cannot sensibly be constructed
    /// without overwhelming the test. The resulting instance must not be inspected or executed.
    pub(crate) unsafe fn new_null() -> Self {
        Self {
            aotvals: SendSyncConstPtr(std::ptr::null()),
            di_tmpfile: None,
            guards: Vec::new(),
            hl: Weak::new(),
        }
    }

    pub(crate) fn mt(&self) -> &Arc<MT> {
        todo!();
    }

    pub(crate) fn smap(&self) -> &HashMap<u64, Vec<LiveVar>> {
        todo!();
    }

    pub(crate) fn aotvals(&self) -> *const c_void {
        todo!();
    }

    pub(crate) fn entry(&self) -> *const c_void {
        todo!();
    }
}

impl Drop for CompiledTrace {
    fn drop(&mut self) {
        // The memory holding the AOT live values needs to live as long as the trace. Now that we
        // no longer need the trace, this can be freed too.
        unsafe { libc::free(self.aotvals.0 as *mut c_void) };
    }
}

impl fmt::Debug for CompiledTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompiledTrace {{ ... }}")
    }
}
