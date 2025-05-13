use crate::{
    compile::jitc_yk::aot_ir::DeoptSafepoint,
    location::HotLocation,
    mt::{TraceId, MT},
    trace::AOTTraceIterator,
};
use libc::c_void;
use parking_lot::Mutex;
use std::{
    error::Error,
    fmt,
    sync::{Arc, Weak},
};

pub(crate) mod guard;
pub(crate) use guard::{Guard, GuardIdx};
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
    /// Compile a mapped root trace into machine code.
    fn root_compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        ctrid: TraceId,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        connector_tid: Option<Arc<dyn CompiledTrace>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError>;

    /// Compile a mapped root trace into machine code.
    fn sidetrace_compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        ctrid: TraceId,
        sti: Arc<dyn SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        connector_tid: Option<Arc<dyn CompiledTrace>>,
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

pub(crate) trait CompiledTrace: fmt::Debug + Send + Sync {
    /// Return this trace's [TraceId].
    fn ctrid(&self) -> TraceId;

    fn safepoint(&self) -> &Option<DeoptSafepoint>;

    /// Upcast this [CompiledTrace] to `Any`. This method is a hack that's only needed since trait
    /// upcasting in Rust is incomplete.
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static>;

    fn sidetraceinfo(
        &self,
        root_ctr: Arc<dyn CompiledTrace>,
        gidx: GuardIdx,
        connect_ctr: Option<Arc<dyn CompiledTrace>>,
    ) -> Arc<dyn SideTraceInfo>;

    /// Return a reference to the guard `id`.
    fn guard(&self, gidx: GuardIdx) -> &Guard;

    fn patch_guard(&self, gidx: GuardIdx, target: *const std::ffi::c_void);

    /// The pointer to this trace's executable code.
    fn entry(&self) -> *const c_void;

    /// The stack adjustment necessary when calling this trace.
    fn entry_sp_off(&self) -> usize;

    /// Return a weak reference to the [HotLocation] that started the top-level trace. Note that a
    /// given `CompiledTrace` may be a side (i.e. a "sub") trace of that top-level trace: the same
    /// [HotLocation] is passed down to each of the child `CompiledTrace`s.
    fn hl(&self) -> &Weak<Mutex<HotLocation>>;

    /// Disassemble the JITted code into a string, for testing and deubgging.
    fn disassemble(&self, with_addrs: bool) -> Result<String, Box<dyn Error>>;
}

/// Stores information required for compiling a side-trace. Passed down from a (parent) trace
/// during deoptimisation.
pub(crate) trait SideTraceInfo: fmt::Debug {
    /// Upcast this [SideTraceInfo] to `Any`. This method is a hack that's only needed since trait
    /// upcasting in Rust is incomplete.
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static>;

    /// Return the [CompiledTrace] this side-trace should jump to at its end. Note: this may be the
    /// "root" trace of a trace tree, or a completely different trace altogether.
    fn target_ctr(&self) -> Arc<dyn CompiledTrace>;
}

#[cfg(test)]
pub(crate) use compiled_trace_testing::*;

#[cfg(test)]
mod compiled_trace_testing {
    use super::*;

    /// A [CompiledTrace] implementation suitable only for testing: when any of its methods are
    /// called it will `panic`.
    #[derive(Debug)]
    pub(crate) struct CompiledTraceTestingMinimal;

    impl CompiledTraceTestingMinimal {
        pub(crate) fn new() -> Self {
            Self
        }
    }

    impl CompiledTrace for CompiledTraceTestingMinimal {
        fn ctrid(&self) -> TraceId {
            TraceId::testing()
        }

        fn safepoint(&self) -> &Option<DeoptSafepoint> {
            todo!()
        }

        fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
            panic!();
        }

        fn sidetraceinfo(
            &self,
            _root_ctr: Arc<dyn CompiledTrace>,
            _gidx: GuardIdx,
            _connect_ctr: Option<Arc<dyn CompiledTrace>>,
        ) -> Arc<dyn SideTraceInfo> {
            panic!();
        }

        fn guard(&self, _gidx: GuardIdx) -> &Guard {
            panic!();
        }

        fn patch_guard(&self, _gidx: GuardIdx, _target: *const std::ffi::c_void) {
            panic!();
        }

        fn entry(&self) -> *const c_void {
            panic!();
        }

        fn entry_sp_off(&self) -> usize {
            panic!();
        }

        fn hl(&self) -> &Weak<Mutex<HotLocation>> {
            panic!();
        }

        fn disassemble(&self, _with_addrs: bool) -> Result<String, Box<dyn Error>> {
            panic!();
        }
    }

    /// A [CompiledTrace] implementation suitable only for testing basic transitions. The `hl` method will return a
    /// [HotLocation] but all other methods will `panic` if called.
    #[derive(Debug)]
    pub(crate) struct CompiledTraceTestingBasicTransitions {
        guard: Guard,
        hl: Weak<Mutex<HotLocation>>,
    }

    impl CompiledTraceTestingBasicTransitions {
        pub(crate) fn new(hl: Weak<Mutex<HotLocation>>) -> Self {
            Self {
                guard: Guard::new(),
                hl,
            }
        }
    }

    impl CompiledTrace for CompiledTraceTestingBasicTransitions {
        fn ctrid(&self) -> TraceId {
            panic!();
        }

        fn safepoint(&self) -> &Option<DeoptSafepoint> {
            todo!()
        }

        fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
            panic!();
        }

        fn sidetraceinfo(
            &self,
            _root_ctr: Arc<dyn CompiledTrace>,
            _gidx: GuardIdx,
            _connect_ctr: Option<Arc<dyn CompiledTrace>>,
        ) -> Arc<dyn SideTraceInfo> {
            panic!();
        }

        fn guard(&self, gidx: GuardIdx) -> &Guard {
            assert_eq!(usize::from(gidx), 0);
            &self.guard
        }

        fn patch_guard(&self, _gidx: GuardIdx, _target: *const std::ffi::c_void) {
            panic!();
        }

        fn entry(&self) -> *const c_void {
            panic!();
        }

        fn entry_sp_off(&self) -> usize {
            panic!();
        }

        fn hl(&self) -> &Weak<Mutex<HotLocation>> {
            &self.hl
        }

        fn disassemble(&self, _with_addrs: bool) -> Result<String, Box<dyn Error>> {
            panic!();
        }
    }
}
