//! Guards: track the state of a guard in a trace.

use crate::{
    compile::CompiledTrace,
    mt::{AtomicTraceCompilationErrorThreshold, HotThreshold, MT},
};
use parking_lot::Mutex;
use std::sync::{atomic::Ordering, Arc};

/// Responsible for tracking how often a guard in a `CompiledTrace` fails. A hotness counter is
/// incremented each time the matching guard failure in a `CompiledTrace` is triggered. Also stores
/// the side-trace once its compiled.
#[derive(Debug)]
pub(crate) struct Guard {
    kind: Mutex<GuardState>,
    /// How many errors have been encountered when tracing or compiling traces resulting from this
    /// guard?
    errors: AtomicTraceCompilationErrorThreshold,
}

#[derive(Debug, PartialEq)]
enum GuardState {
    Counting(HotThreshold),
    /// We are either creating a trace or waiting for that trace to compile.
    SideTracing,
    Compiled,
    /// This guard encountered errors sufficiently often when tracing or compiling that we don't
    /// want to try again.
    DontTrace,
}

impl Guard {
    pub(crate) fn new() -> Self {
        Self {
            kind: Mutex::new(GuardState::Counting(0)),
            errors: AtomicTraceCompilationErrorThreshold::new(0),
        }
    }

    /// This guard has failed (i.e. evaluated to true/false when false/true was expected). Returns
    /// `true` if this guard has failed often enough to be worth side-tracing.
    pub fn inc_failed(&self, mt: &Arc<MT>) -> bool {
        let mut lk = self.kind.lock();
        match &*lk {
            GuardState::Counting(x) => {
                if x + 1 >= mt.sidetrace_threshold() {
                    *lk = GuardState::SideTracing;
                    true
                } else {
                    *lk = GuardState::Counting(x + 1);
                    false
                }
            }
            GuardState::SideTracing | GuardState::Compiled | GuardState::DontTrace => false,
        }
    }

    /// Inform this guard that a trace started from it failed (either in tracing or compiling).
    pub fn trace_or_compile_failed(&self, mt: &Arc<MT>) {
        let mut lk = self.kind.lock();
        if let GuardState::SideTracing = &*lk {
            let failures = self.errors.fetch_add(1, Ordering::Relaxed);
            if failures >= mt.trace_failure_threshold() {
                assert_eq!(*lk, GuardState::SideTracing);
                *lk = GuardState::DontTrace;
            } else {
                *lk = GuardState::Counting(0);
            }
        } else {
            panic!();
        }
    }

    /// Stores a compiled side-trace inside this guard while patching a jump to the side-trace
    /// directly into the parent trace.
    /// * `ctr`: The compiled side-trace.
    /// * `parent`: The immediate parent of the side-trace.
    /// * `gidx`: The guard id of the side-trace.
    pub fn set_ctr(
        &self,
        ctr: Arc<dyn CompiledTrace>,
        parent: &Arc<dyn CompiledTrace>,
        gidx: GuardIdx,
    ) {
        let mut lk = self.kind.lock();
        let addr = ctr.entry();
        match &*lk {
            GuardState::SideTracing => *lk = GuardState::Compiled,
            _ => panic!(),
        }
        // It's important to patch the parent only after we've updated the `GuardState` to avoid a
        // race condition. If we were to patch the parent trace first, there is a small window
        // where another thread takes the patched jump and deopts before we had a chance to call
        // `set_ctr` which sets information required by deopt.
        parent.patch_guard(gidx, addr);
    }
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
