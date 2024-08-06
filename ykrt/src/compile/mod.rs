use crate::{location::HotLocation, mt::MT, trace::AOTTraceIterator};
use libc::c_void;
use parking_lot::Mutex;
use std::{
    error::Error,
    fmt,
    sync::{atomic::AtomicU32, Arc, Weak},
};

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
        aottrace_iter: Box<dyn AOTTraceIterator>,
        sti: Option<Arc<dyn SideTraceInfo>>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[usize]>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError>;
}

pub(crate) fn default_compiler() -> Result<Arc<dyn Compiler>, Box<dyn Error>> {
    #[cfg(jitc_yk)]
    return Ok(jitc_yk::JITCYk::new()?);

    #[allow(unreachable_code)]
    {
        Err("No JIT compiler supported on this platform/configuration".into())
    }
}

/// Responsible for tracking how often a guard in a `CompiledTrace` fails. A hotness counter is
/// incremented each time the matching guard failure in a `CompiledTrace` is triggered. Also stores
/// the side-trace once its compiled.
#[derive(Debug)]
pub(crate) struct Guard {
    /// How often has this guard failed?
    #[allow(dead_code)]
    failed: AtomicU32,
    ct: Mutex<Option<Arc<dyn CompiledTrace>>>,
}

impl Guard {
    /// This guard has failed (i.e. evaluated to true/false when false/true was expected). Returns
    /// `true` if this guard has failed often enough to be worth side-tracing.
    pub fn inc_failed(&self, _mt: &Arc<MT>) -> bool {
        // FIXME: for now we forcibly disable side-tracing, as it's broken.
        //self.failed.fetch_add(1, Ordering::Relaxed) + 1 >= mt.sidetrace_threshold()
        false
    }

    /// Stores a compiled side-trace inside this guard.
    pub fn set_ctr(&self, ct: Arc<dyn CompiledTrace>) {
        let _ = self.ct.lock().insert(ct);
    }

    /// Return the compiled side-trace or None if no side-trace has been compiled.
    pub fn ctr(&self) -> Option<Arc<dyn CompiledTrace>> {
        self.ct.lock().as_ref().map(Arc::clone)
    }
}

pub(crate) trait CompiledTrace: fmt::Debug + Send + Sync {
    /// Upcast this [CompiledTrace] to `Any`. This method is a hack that's only needed since trait
    /// upcasting in Rust is incomplete.
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static>;

    fn sidetraceinfo(&self, gidx: GuardIdx) -> Arc<dyn SideTraceInfo>;

    /// Return a reference to the guard `id`.
    fn guard(&self, gidx: GuardIdx) -> &Guard;

    fn entry(&self) -> *const c_void;

    /// Return a weak reference to the [HotLocation] that started the top-level trace. Note that a
    /// given `CompiledTrace` may be a side (i.e. a "sub") trace of that top-level trace: the same
    /// [HotLocation] is passed down to each of the child `CompiledTrace`s.
    fn hl(&self) -> &Weak<Mutex<HotLocation>>;

    /// Disassemble the JITted code into a string, for testing and deubgging.
    fn disassemble(&self) -> Result<String, Box<dyn Error>>;
}

/// Stores information required for compiling a side-trace. Passed down from a (parent) trace
/// during deoptimisation.
pub(crate) trait SideTraceInfo {
    /// Upcast this [SideTraceInfo] to `Any`. This method is a hack that's only needed since trait
    /// upcasting in Rust is incomplete.
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static>;
}

/// Identify a [Guard] within a trace.
///
/// This is guaranteed to be an index into an array that is freely convertible to/from [usize].
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuardIdx(usize);

impl From<usize> for GuardIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<GuardIdx> for usize {
    fn from(v: GuardIdx) -> Self {
        v.0
    }
}

#[cfg(test)]
mod compiled_trace_testing {
    use super::*;

    /// A [CompiledTrace] implementation suitable only for testing: when any of its methods are
    /// called it will `panic`.
    #[derive(Debug)]
    pub(crate) struct CompiledTraceTesting;

    impl CompiledTraceTesting {
        pub(crate) fn new() -> Self {
            Self
        }
    }

    impl CompiledTrace for CompiledTraceTesting {
        fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
            panic!();
        }

        fn sidetraceinfo(&self, _gidx: GuardIdx) -> Arc<dyn SideTraceInfo> {
            panic!();
        }

        fn guard(&self, _gidx: GuardIdx) -> &Guard {
            panic!();
        }

        fn entry(&self) -> *const c_void {
            panic!();
        }

        fn hl(&self) -> &Weak<Mutex<HotLocation>> {
            panic!();
        }

        fn disassemble(&self) -> Result<String, Box<dyn Error>> {
            panic!();
        }
    }

    /// A [CompiledTrace] implementation suitable only for testing. The `hl` method will return a
    /// [HotLocation] but all other methods will `panic` if called.
    #[derive(Debug)]
    pub(crate) struct CompiledTraceTestingWithHl(Weak<Mutex<HotLocation>>);

    impl CompiledTraceTestingWithHl {
        pub(crate) fn new(hl: Weak<Mutex<HotLocation>>) -> Self {
            Self(hl)
        }
    }

    impl CompiledTrace for CompiledTraceTestingWithHl {
        fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
            panic!();
        }

        fn sidetraceinfo(&self, _gidx: GuardIdx) -> Arc<dyn SideTraceInfo> {
            panic!();
        }

        fn guard(&self, _gidx: GuardIdx) -> &Guard {
            panic!();
        }

        fn entry(&self) -> *const c_void {
            panic!();
        }

        fn hl(&self) -> &Weak<Mutex<HotLocation>> {
            &self.0
        }

        fn disassemble(&self) -> Result<String, Box<dyn Error>> {
            panic!();
        }
    }
}

#[cfg(test)]
pub(crate) use compiled_trace_testing::*;
