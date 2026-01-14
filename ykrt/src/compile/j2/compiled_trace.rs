use crate::{
    compile::{
        CompiledTrace,
        guard::{Guard, GuardId},
        j2::{
            codebuf::ExeCodeBuf,
            hir::Switch,
            hir_to_asm::AsmGuardIdx,
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
    pub guards: IndexVec<AsmGuardIdx, J2CompiledGuard<Reg>>,
    pub trace_start: J2TraceStart<Reg>,
}

impl<Reg: RegT> J2CompiledTrace<Reg> {
    pub(super) fn new(
        mt: Arc<MT>,
        trid: TraceId,
        hl: Weak<Mutex<HotLocation>>,
        codebuf: ExeCodeBuf,
        guards: IndexVec<AsmGuardIdx, J2CompiledGuard<Reg>>,
        trace_start: J2TraceStart<Reg>,
    ) -> Self {
        Self {
            mt,
            trid,
            hl,
            codebuf,
            guards,
            trace_start,
        }
    }

    pub(super) fn bid(&self, gidx: AsmGuardIdx) -> aot_ir::BBlockId {
        self.guards[gidx].bid()
    }

    pub(super) fn switch(&self, gidx: AsmGuardIdx) -> Option<&Switch> {
        self.guards[gidx].switch.as_ref()
    }

    pub(super) fn guard(&self, gidx: AsmGuardIdx) -> &J2CompiledGuard<Reg> {
        &self.guards[gidx]
    }

    pub(super) fn entry_vlocs(&self) -> &[VarLocs<Reg>] {
        match &self.trace_start {
            J2TraceStart::ControlPoint { entry_vlocs, .. } => entry_vlocs,
            J2TraceStart::Guard { stack_off: _ } => todo!(),
        }
    }

    pub(super) fn exe(&self) -> *mut c_void {
        self.codebuf.as_ptr() as *mut c_void
    }

    pub(super) fn guard_stack_off(&self, gidx: AsmGuardIdx) -> u32 {
        match self.trace_start {
            J2TraceStart::ControlPoint { stack_off, .. }
            | J2TraceStart::Guard { stack_off, .. } => {
                stack_off + self.guards[gidx].extra_stack_len
            }
        }
    }

    /// Return the size of the stack of the entry block. This is used by side-traces to set the
    /// stack pointer to the right value just before jumping to a loop / coupler trace.
    pub(super) fn entry_stack_off(&self) -> u32 {
        match self.trace_start {
            J2TraceStart::ControlPoint { stack_off, .. }
            | J2TraceStart::Guard { stack_off, .. } => stack_off,
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
        let gidx = AsmGuardIdx::from(usize::from(gid));
        self.guards[gidx].guard()
    }

    // FIXME: This should really be handled in the backend, but the structure of CompiledTrace
    // / J2CompiledTrace makes this awkward.
    #[cfg(target_arch = "x86_64")]
    fn patch_guard(&self, gid: GuardId, tgt: *const std::ffi::c_void) {
        let gidx = AsmGuardIdx::from(usize::from(gid));
        let patch_off = usize::try_from(self.guards[gidx].patch_off()).unwrap();
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

/// Where did this J2 compiled trace start?
#[derive(Debug)]
pub(super) enum J2TraceStart<Reg: RegT> {
    ControlPoint {
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
    Guard {
        stack_off: u32,
    },
}

#[derive(Debug)]
pub(super) struct J2CompiledGuard<Reg: RegT> {
    // Generic yk stuff that probably should be stored in a struct at the `compile/mod.rs` level.
    guard: Guard,

    // X64 / j2 specific stuff.
    /// The block ID of the guard, needed for `prev_bid` in `aot_to_hir`.
    bid: aot_ir::BBlockId,
    /// The [DeoptFrame]s necessary to reconstruct the stackframes for this guard. See also
    /// [Self::deopt_vars].
    pub deopt_frames: SmallVec<[DeoptFrame; 2]>,
    /// The variables used on entry to the guard. These are stored as an ordered, flat, sequence,
    /// corresponding to the sequence of [Self::deopt_frames]. For example, if `deopt_frames` has 2
    /// frames, the first of which needs 3 variables and the second 1 variable entry_vars would
    /// look as follows:
    /// ```text
    /// [a, b, c, d]
    ///  ^^^^^^^
    ///     |     ^
    ///     |   deopt_frames[1] variable
    ///     |
    ///  deopt_frames[0] variables
    /// ```
    pub deopt_vars: Vec<DeoptVar<Reg>>,
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
        deopt_frames: SmallVec<[DeoptFrame; 2]>,
        deopt_vars: Vec<DeoptVar<Reg>>,
        patch_off: u32,
        extra_stack_len: u32,
        switch: Option<Switch>,
    ) -> Self {
        Self {
            bid,
            deopt_frames,
            deopt_vars,
            guard: Guard::new(),
            patch_off,
            extra_stack_len,
            switch,
        }
    }

    pub(super) fn bid(&self) -> aot_ir::BBlockId {
        self.bid
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
pub(super) struct DeoptFrame {
    pub pc: InstId,
    pub pc_safepoint: &'static DeoptSafepoint,
}

#[derive(Debug)]
pub(super) struct DeoptVar<Reg: RegT> {
    pub bitw: u32,
    pub fromvlocs: VarLocs<Reg>,
    pub tovlocs: VarLocs<Reg>,
}
