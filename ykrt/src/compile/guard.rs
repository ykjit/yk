//! Guards: track the state of a guard in a trace.

use crate::{
    compile::CompiledTrace,
    mt::{HotThreshold, MT},
};
use parking_lot::Mutex;
use std::sync::Arc;

/// Responsible for tracking how often a guard in a `CompiledTrace` fails. A hotness counter is
/// incremented each time the matching guard failure in a `CompiledTrace` is triggered. Also stores
/// the side-trace once its compiled.
#[derive(Debug)]
pub(crate) struct Guard(Mutex<GuardState>);

#[derive(Debug)]
enum GuardState {
    Counting(HotThreshold),
    SideTracing,
    Compiled,
}

impl Guard {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(GuardState::Counting(0)))
    }

    /// This guard has failed (i.e. evaluated to true/false when false/true was expected). Returns
    /// `true` if this guard has failed often enough to be worth side-tracing.
    pub fn inc_failed(&self, mt: &Arc<MT>) -> bool {
        let mut lk = self.0.lock();
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
            GuardState::SideTracing => false,
            GuardState::Compiled => false,
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
        let mut lk = self.0.lock();
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
