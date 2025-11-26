use crate::{
    compile::{
        CompiledTrace,
        guard::{Guard, GuardId},
        j2::{
            codebuf::ExeCodeBuf,
            hir::{GuardRestoreIdx, Switch},
            regalloc::{RegT, VarLocs},
        },
        jitc_yk::aot_ir::{self, DeoptSafepoint, InstId},
    },
    location::HotLocation,
    mt::{MT, TraceId},
};
use index_vec::IndexVec;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    any::Any,
    error::Error,
    ffi::c_void,
    sync::{Arc, Weak},
};

#[derive(Debug)]
pub(super) struct J2CompiledTrace<Reg: RegT> {
    pub mt: Arc<MT>,
    pub trid: TraceId,
    pub hl: Weak<Mutex<HotLocation>>,
    codebuf: ExeCodeBuf,
    pub guard_restores: IndexVec<GuardRestoreIdx, J2CompiledGuard<Reg>>,
    pub kind: J2CompiledTraceKind<Reg>,
}

impl<Reg: RegT> J2CompiledTrace<Reg> {
    pub(super) fn new(
        mt: Arc<MT>,
        trid: TraceId,
        hl: Weak<Mutex<HotLocation>>,
        codebuf: ExeCodeBuf,
        guards: IndexVec<GuardRestoreIdx, J2CompiledGuard<Reg>>,
        kind: J2CompiledTraceKind<Reg>,
    ) -> Self {
        Self {
            mt,
            trid,
            hl,
            codebuf,
            guard_restores: guards,
            kind,
        }
    }

    pub(super) fn bid(&self, gridx: GuardRestoreIdx) -> aot_ir::BBlockId {
        self.guard_restores[gridx].bid()
    }

    pub(super) fn switch(&self, gridx: GuardRestoreIdx) -> Option<&Switch> {
        self.guard_restores[gridx].switch.as_ref()
    }

    pub(super) fn deopt_frames(&self, gridx: GuardRestoreIdx) -> &[DeoptFrame<Reg>] {
        self.guard_restores[gridx].deopt_frames()
    }

    pub(super) fn entry_vlocs(&self) -> &[VarLocs<Reg>] {
        match &self.kind {
            J2CompiledTraceKind::Coupler { entry_vlocs, .. }
            | J2CompiledTraceKind::Loop { entry_vlocs, .. } => entry_vlocs,
            J2CompiledTraceKind::Side { stack_off: _ } => todo!(),
            #[cfg(test)]
            J2CompiledTraceKind::Test => todo!(),
        }
    }

    pub(super) fn exe(&self) -> *mut c_void {
        self.codebuf.as_ptr() as *mut c_void
    }

    pub(super) fn guard_stack_off(&self, gridx: GuardRestoreIdx) -> u32 {
        match self.kind {
            J2CompiledTraceKind::Coupler { stack_off, .. }
            | J2CompiledTraceKind::Loop { stack_off, .. }
            | J2CompiledTraceKind::Side { stack_off } => {
                stack_off + self.guard_restores[gridx].extra_stack_len
            }
            #[cfg(test)]
            J2CompiledTraceKind::Test => todo!(),
        }
    }

    /// Return the size of the stack of the entry block. This is used by side-traces to set the
    /// stack pointer to the right value just before jumping to a loop / coupler trace.
    pub(super) fn entry_stack_off(&self) -> u32 {
        match self.kind {
            J2CompiledTraceKind::Coupler { stack_off, .. }
            | J2CompiledTraceKind::Loop { stack_off, .. } => stack_off,
            J2CompiledTraceKind::Side { .. } => todo!(),
            #[cfg(test)]
            J2CompiledTraceKind::Test => todo!(),
        }
    }
}

impl<Reg: RegT + 'static> CompiledTrace for J2CompiledTrace<Reg> {
    fn ctrid(&self) -> TraceId {
        self.trid
    }

    fn safepoint(&self) -> &Option<aot_ir::DeoptSafepoint> {
        todo!()
    }

    fn as_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn guard(&self, gid: GuardId) -> &Guard {
        let gridx = GuardRestoreIdx::from(usize::from(gid));
        self.guard_restores[gridx].guard()
    }

    // FIXME: This should really be handled in the backend, but the structure of CompiledTrace
    // / J2CompiledTrace makes this awkward.
    #[cfg(target_arch = "x86_64")]
    fn patch_guard(&self, gid: GuardId, tgt: *const std::ffi::c_void) {
        let gridx = GuardRestoreIdx::from(usize::from(gid));
        let patch_off = usize::try_from(self.guard_restores[gridx].patch_off()).unwrap();
        self.codebuf.patch(patch_off, 5, |patch_addr| {
            assert_eq!(unsafe { patch_addr.read() }, 0xE9);
            let patch_addr = unsafe { patch_addr.byte_add(1) };
            let next_ip = patch_addr.addr() + 4;
            let diff = i32::try_from(tgt.addr().checked_signed_diff(next_ip).unwrap()).unwrap();
            unsafe {
                (patch_addr as *mut u32).write(diff.cast_unsigned());
            }
        });
    }

    fn entry(&self) -> *const c_void {
        self.codebuf.as_ptr() as *const c_void
    }

    fn entry_sp_off(&self) -> usize {
        todo!()
    }

    fn hl(&self) -> &std::sync::Weak<parking_lot::Mutex<HotLocation>> {
        &self.hl
    }

    fn disassemble(&self, _with_addrs: bool) -> Result<String, Box<dyn Error>> {
        todo!()
    }

    fn code(&self) -> &[u8] {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }
}

/// What kind of [J2CompiledTrace] is this trace?
#[derive(Debug)]
pub(super) enum J2CompiledTraceKind<Reg: RegT> {
    /// A coupler trace.
    Coupler {
        /// The entry safepoint. See [entry_vlocs].
        entry_safepoint: &'static DeoptSafepoint,
        /// Every entry in `entry_safepoint.lives` will have an entry in `entry_vlocs`, in order.
        /// In other words, `entry_safepoint.lives.iter().zip(entry_vlocs.iter())` is guaranteed to
        /// work as expected.
        ///
        /// However, some variables will have empty `VarLocs`. In other words, while this coupler
        /// trace guarantees to accept variables being set in accordance with
        /// `entry_safepoint.lives`, it is also happy with a non-strict subset of those. That means
        /// that other traces jumping to this coupler trace only need to deal with the subset
        /// recorded in `entry_vlocs` (i.e. they can ignore the superset in
        /// `entry_safepoint.lives`).
        entry_vlocs: Vec<VarLocs<Reg>>,
        stack_off: u32,
        /// The offset into the compiled trace that sidetraces should jump to.
        sidetrace_off: usize,
    },
    /// A loop trace.
    Loop {
        /// The entry safepoint. See [entry_vlocs].
        entry_safepoint: &'static DeoptSafepoint,
        /// Every entry in `entry_safepoint.lives` will have an entry in `entry_vlocs`, in order.
        /// In other words, `entry_safepoint.lives.iter().zip(entry_vlocs.iter())` is guaranteed to
        /// work as expected.
        ///
        /// However, some variables will have empty `VarLocs`. In other words, while this loop
        /// trace guarantees to accept variables being set in accordance with
        /// `entry_safepoint.lives`, it is also happy with a non-strict subset of those. That means
        /// that other traces jumping to this loop trace only need to deal with the subset recorded
        /// in `entry_vlocs` (i.e. they can ignore the superset in `entry_safepoint.lives`).
        entry_vlocs: Vec<VarLocs<Reg>>,
        stack_off: u32,
        /// The offset into the compiled trace that sidetraces should jump to.
        sidetrace_off: usize,
    },
    Side {
        stack_off: u32,
    },
    #[cfg(test)]
    #[allow(dead_code)]
    Test,
}

#[derive(Debug)]
pub(super) struct J2CompiledGuard<Reg: RegT> {
    // Generic yk stuff that probably should be stored in a struct at the `compile/mod.rs` level.
    guard: Guard,

    // X64 / j2 specific stuff.
    /// The block ID of the guard, needed for `prev_bid` in `aot_to_hir`.
    bid: aot_ir::BBlockId,
    deopt_frames: SmallVec<[DeoptFrame<Reg>; 1]>,
    patch_off: u32,
    /// How much additional space will this guard have consumed relative to the main part of the
    /// trace it was part of?
    pub extra_stack_len: u32,
    /// If this guard:
    ///
    ///   1. is the first guard in a trace,
    ///   2. relates to an LLVM-level `switch`,
    ///
    /// then this records the information necessary for subsequent sidetraces to deal with the
    /// switch properly.
    pub switch: Option<Switch>,
}

impl<Reg: RegT> J2CompiledGuard<Reg> {
    pub(super) fn new(
        bid: aot_ir::BBlockId,
        deopt_frames: SmallVec<[DeoptFrame<Reg>; 1]>,
        patch_off: u32,
        extra_stack_len: u32,
        switch: Option<Switch>,
    ) -> Self {
        Self {
            bid,
            deopt_frames,
            guard: Guard::new(),
            patch_off,
            extra_stack_len,
            switch,
        }
    }

    pub(super) fn bid(&self) -> aot_ir::BBlockId {
        self.bid
    }

    pub(super) fn deopt_frames(&self) -> &SmallVec<[DeoptFrame<Reg>; 1]> {
        &self.deopt_frames
    }

    pub(super) fn guard(&self) -> &Guard {
        &self.guard
    }

    pub(super) fn patch_off(&self) -> u32 {
        self.patch_off
    }
}

/// The information about a frame necessary for deopt and side-tracing.
#[derive(Debug)]
pub(super) struct DeoptFrame<Reg: RegT> {
    pub safepoint: &'static DeoptSafepoint,
    /// If this is an inlined frame (i.e. for all but the bottom frame), this is the [InstId] of
    /// the call instruction. This is used to link the return value when the inlined frame is
    /// popped.
    pub call_iid: Option<aot_ir::InstId>,
    pub func: aot_ir::FuncIdx,
    /// The information necessary for deopt and side-tracing: in a sense we precalculate this from
    /// `self.vars` to (a) avoid us having to carry around a [hir::Module] (b) optimise how much we
    /// need to read/write. It's currently unclear whether doing this is a good trade
    /// memory/performance trade-off or not.
    pub vars: Vec<(InstId, u32, VarLocs<Reg>, VarLocs<Reg>)>,
}
