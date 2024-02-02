use crate::{
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::TracedAOTBlock,
};
use libc::c_void;
use parking_lot::Mutex;
#[cfg(not(test))]
use std::slice;
use std::{
    collections::HashMap,
    env, fmt,
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

#[cfg(jitc_yk)]
pub mod jitc_yk;

/// A failure to compile a trace.
#[derive(Debug, thiserror::Error)]
pub enum CompilationError {
    #[error("Unrecoverable error: {0}")]
    Unrecoverable(String),
    #[error("Temporary error: {0}")]
    Temporary(String),
}

/// The trait that every JIT compiler backend must implement.
pub(crate) trait Compiler: Send + Sync {
    /// Compile a mapped trace into machine code.
    fn compile(
        &self,
        mt: Arc<MT>,
        irtrace: Vec<TracedAOTBlock>,
        sti: Option<SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<CompiledTrace, CompilationError>;
}

pub(crate) fn default_compiler() -> Result<Arc<dyn Compiler>, CompilationError> {
    #[cfg(jitc_yk)]
    // Transitionary env var to turn on the new code generator.
    //
    // This will be removed once the transition away from LLVM as a trace compiler is complete.
    if let Ok(v) = env::var("YKD_NEW_CODEGEN") {
        if v == "1" {
            return Ok(jitc_yk::JITCYk::new()?);
        }
    }
    #[cfg(jitc_llvm)]
    {
        return Ok(jitc_llvm::JITCLLVM::new());
    }

    #[allow(unreachable_code)]
    {
        Err(CompilationError::Unrecoverable(
            "No JIT compiler supported on this platform/configuration".into(),
        ))
    }
}

#[cfg(not(test))]
struct SendSyncConstPtr<T>(*const T);
#[cfg(not(test))]
unsafe impl<T> Send for SendSyncConstPtr<T> {}
#[cfg(not(test))]
unsafe impl<T> Sync for SendSyncConstPtr<T> {}

/// Responsible for tracking how often a guard in a `CompiledTrace` fails. A hotness counter is
/// incremented each time the matching guard failure in a `CompiledTrace` is triggered. Also stores
/// the side-trace once its compiled.
pub(crate) struct Guard {
    /// How often has this guard failed?
    failed: AtomicU32,
    ct: Mutex<Option<Arc<CompiledTrace>>>,
}

impl Guard {
    /// Increments the guard failure counter. Returns `true` if the guard has failed often enough
    /// to be worth side-tracing.
    pub fn inc_failed(&self, mt: &Arc<MT>) -> bool {
        self.failed.fetch_add(1, Ordering::Relaxed) + 1 >= mt.sidetrace_threshold()
    }

    /// Stores a compiled side-trace inside this guard.
    pub fn setct(&self, ct: Arc<CompiledTrace>) {
        let _ = self.ct.lock().insert(ct);
    }

    /// Retrieves the stored side-trace or None, if no side-trace has been compiled yet.
    pub fn getct(&self) -> Option<Arc<CompiledTrace>> {
        self.ct.lock().as_ref().map(Arc::clone)
    }
}

/// A trace compiled into machine code. Note that these are passed around as raw pointers and
/// potentially referenced by multiple threads so, once created, instances of this struct can only
/// be updated if a lock is held or a field is atomic.
#[cfg(not(test))]
pub(crate) struct CompiledTrace {
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

    pub(crate) fn mt(&self) -> &Arc<MT> {
        &self.mt
    }

    pub(crate) fn smap(&self) -> &HashMap<u64, Vec<LiveVar>> {
        &self.smap
    }

    /// Return a reference to the guard `id`.
    pub(crate) fn guard(&self, id: GuardId) -> &Guard {
        &self.guards[id.0]
    }

    /// Is the guard `id` the last guard in this `CompiledTrace`?
    pub(crate) fn is_last_guard(&self, id: GuardId) -> bool {
        id.0 + 1 == self.guards.len()
    }

    pub(crate) fn aotvals(&self) -> *const c_void {
        self.aotvals.0
    }

    pub(crate) fn entry(&self) -> *const c_void {
        self.entry.0
    }

    pub(crate) fn hl(&self) -> &Weak<Mutex<HotLocation>> {
        &self.hl
    }
}

#[cfg(not(test))]
impl Drop for CompiledTrace {
    fn drop(&mut self) {
        // The memory holding the AOT live values needs to live as long as the trace. Now that we
        // no longer need the trace, this can be freed too.
        unsafe { libc::free(self.aotvals.0 as *mut c_void) };
        // FIXME: This should drop the JITted code.
    }
}

impl fmt::Debug for CompiledTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompiledTrace {{ ... }}")
    }
}

#[cfg(test)]
pub(crate) struct CompiledTrace;

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

    /// Create a `CompiledTrace` suitable for testing purposes. The resulting instance is not
    /// useful other than as a placeholder: calling any of its methods will cause a panic.
    pub(crate) fn new_testing() -> Self {
        Self
    }

    pub(crate) fn mt(&self) -> &Arc<MT> {
        todo!();
    }

    pub(crate) fn smap(&self) -> &HashMap<u64, Vec<LiveVar>> {
        todo!();
    }

    pub(crate) fn guard(&self, _id: GuardId) -> &Guard {
        todo!();
    }

    pub(crate) fn is_last_guard(&self, _id: GuardId) -> bool {
        todo!();
    }

    pub(crate) fn aotvals(&self) -> *const c_void {
        todo!();
    }

    pub(crate) fn entry(&self) -> *const c_void {
        todo!();
    }

    pub(crate) fn hl(&self) -> &Weak<Mutex<HotLocation>> {
        todo!();
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct GuardId(pub(crate) usize);

impl GuardId {
    #[cfg(test)]
    /// Only when testing, create a `GuardId` with an illegal value: trying to use this `GuardId`
    /// will either cause an error or lead to undefined behaviour.
    pub(crate) fn illegal() -> Self {
        GuardId(usize::max_value())
    }
}
