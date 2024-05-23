use crate::{
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::AOTTraceIterator,
};
use libc::c_void;
use parking_lot::Mutex;
use std::{
    env,
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Weak,
    },
};

#[cfg(jitc_llvm)]
pub(crate) mod jitc_llvm;

#[cfg(jitc_yk)]
pub mod jitc_yk;

/// A failure to compile a trace.
#[derive(Debug)]
pub(crate) enum CompilationError {
    /// Compilation failed for reasons that might be of interest to a programmer augmenting an
    /// interpreter with yk but not to the end user running a program on the interpreter.
    General(String),
    /// Something went wrong when compiling that is probably the result of a bug in yk.
    InternalError(String),
    /// A limit was exceeded (e.g. a pointer add that went beyond a struct). We try and check for
    /// as many of these as possible at compile time, but some can only be detected at JIT time.
    /// Most, perhaps all, of these suggest bugs in the interpreter.
    LimitExceeded(String),
    /// Compilation failed because an external resource was exhausted: the end user running the
    /// interpreter probably wants to be informed of this.
    ResourceExhausted(Box<dyn Error>),
}

impl fmt::Display for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationError::General(s) => write!(f, "General error: {s}"),
            CompilationError::InternalError(s) => write!(f, "Internal error: {s}"),
            CompilationError::LimitExceeded(s) => write!(f, "Limit exceeded: {s}"),
            CompilationError::ResourceExhausted(e) => write!(f, "Resource exhausted: {e:}"),
        }
    }
}

/// The trait that every JIT compiler backend must implement.
pub(crate) trait Compiler: Send + Sync {
    /// Compile a mapped trace into machine code.
    fn compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: (Box<dyn AOTTraceIterator>, Box<[usize]>),
        sti: Option<SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError>;
}

pub(crate) fn default_compiler() -> Result<Arc<dyn Compiler>, Box<dyn Error>> {
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
        Err("No JIT compiler supported on this platform/configuration".into())
    }
}

/// Responsible for tracking how often a guard in a `CompiledTrace` fails. A hotness counter is
/// incremented each time the matching guard failure in a `CompiledTrace` is triggered. Also stores
/// the side-trace once its compiled.
pub(crate) struct Guard {
    /// How often has this guard failed?
    failed: AtomicU32,
    ct: Mutex<Option<Arc<dyn CompiledTrace>>>,
}

impl Guard {
    /// Increments the guard failure counter. Returns `true` if the guard has failed often enough
    /// to be worth side-tracing.
    pub fn inc_failed(&self, mt: &Arc<MT>) -> bool {
        self.failed.fetch_add(1, Ordering::Relaxed) + 1 >= mt.sidetrace_threshold()
    }

    /// Stores a compiled side-trace inside this guard.
    pub fn setct(&self, ct: Arc<dyn CompiledTrace>) {
        let _ = self.ct.lock().insert(ct);
    }

    /// Retrieves the stored side-trace or None, if no side-trace has been compiled yet.
    pub fn getct(&self) -> Option<Arc<dyn CompiledTrace>> {
        self.ct.lock().as_ref().map(Arc::clone)
    }
}

pub(crate) trait CompiledTrace: fmt::Debug + Send + Sync {
    /// Upcast this [CompiledTrace] to `Any`. This method is a hack that's only needed since trait
    /// upcasting in Rust is incomplete.
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static>;

    fn mt(&self) -> &Arc<MT>;

    /// Return a reference to the guard `id`.
    fn guard(&self, id: GuardId) -> &Guard;

    /// Is the guard `id` the last guard in this `CompiledTrace`?
    fn is_last_guard(&self, id: GuardId) -> bool;

    fn aotvals(&self) -> *const c_void;

    fn entry(&self) -> *const c_void;

    /// Return a weak reference to the [HotLocation] that started the top-level trace. Note that a
    /// given `CompiledTrace` may be a side (i.e. a "sub") trace of that top-level trace: the same
    /// [HotLocation] is passed down to each of the child `CompiledTrace`s.
    fn hl(&self) -> &Weak<Mutex<HotLocation>>;

    /// Disassemble the JITted code into a string, for testing and deubgging.
    #[cfg(any(debug_assertions, test))]
    fn disassemble(&self) -> Result<String, Box<dyn Error>>;
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct GuardId(pub(crate) usize);

impl GuardId {
    #[cfg(test)]
    /// Only when testing, create a `GuardId` with an illegal value: trying to use this `GuardId`
    /// will either cause an error or lead to undefined behaviour.
    pub(crate) fn illegal() -> Self {
        GuardId(usize::MAX)
    }
}
