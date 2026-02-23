use crate::{
    compile::{
        CompiledTrace,
        guard::{Guard, GuardId},
        j2::{
            codebuf::ExeCodeBuf,
            hir::{Mod, Switch, TraceEnd, TraceStart},
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
    assert_matches,
    ffi::c_void,
    sync::{Arc, Weak},
};

#[derive(Debug)]
pub(super) struct J2CompiledTrace<Reg: RegT> {
    pub mt: Arc<MT>,
    pub trid: TraceId,
    pub hl: Weak<Mutex<HotLocation>>,
    codebuf: ExeCodeBuf,
    pub guards: IndexVec<CompiledGuardIdx, J2CompiledGuard<Reg>>,
    pub trace_start: J2TraceStart<Reg>,
    /// The name used for this trace as linker symbol.
    symbol_name: String,
}

impl<Reg: RegT> J2CompiledTrace<Reg> {
    pub(super) fn new(
        mt: Arc<MT>,
        m: &Mod<Reg>,
        hl: Weak<Mutex<HotLocation>>,
        codebuf: ExeCodeBuf,
        guards: IndexVec<CompiledGuardIdx, J2CompiledGuard<Reg>>,
        trace_start: J2TraceStart<Reg>,
    ) -> Self {
        // Extract source trace ID for guard traces.
        let src_ctr = match &m.trace_start {
            TraceStart::Guard { src_ctr, .. } => Some(src_ctr.trid),
            _ => None,
        };

        // Extract target trace ID for coupler traces.
        let tgt_ctr = match &m.trace_end {
            TraceEnd::Coupler { tgt_ctr, .. } => Some(tgt_ctr.trid),
            _ => None,
        };

        let symbol_name =
            Self::get_trace_name(m.trid, &trace_start, &m.trace_end, src_ctr, tgt_ctr);

        Self {
            mt,
            trid: m.trid,
            hl,
            codebuf,
            guards,
            trace_start,
            symbol_name,
        }
    }

    /// Generate a linker symbol name for the compiled trace.
    ///
    /// The symbol name follows the format `__yk_trace_{trid}_{start_type}_{end_type}`, where:
    /// - `start_type` is either `control` (for control point traces) or `guard` (for guard traces,
    ///   optionally including the source trace ID).
    /// - `end_type` indicates how the trace terminates (`loop`, `coupler`, `return`, etc.).
    fn get_trace_name<R: RegT>(
        trid: TraceId,
        trace_start: &J2TraceStart<R>,
        trace_end: &TraceEnd<R>,
        src_ctr: Option<TraceId>,
        tgt_ctr: Option<TraceId>,
    ) -> String {
        let trace_name = match (trace_start, src_ctr) {
            (J2TraceStart::ControlPoint { .. }, _) => {
                format!("__yk_trace_{trid}_control")
            }
            (J2TraceStart::Guard { .. }, Some(src)) => {
                format!("__yk_trace_{trid}_guard_{src}")
            }
            (J2TraceStart::Guard { .. }, None) => format!("__yk_trace_{trid}_guard"),
        };

        match trace_end {
            TraceEnd::Loop { .. } => format!("{trace_name}_loop"),
            TraceEnd::Coupler { .. } => {
                let tgt = tgt_ctr.unwrap();
                format!("{trace_name}_coupler_{tgt}")
            }
            TraceEnd::Return { .. } => format!("{trace_name}_return"),
            #[cfg(test)]
            TraceEnd::Test { .. } => format!("{trace_name}_test"),
            #[cfg(test)]
            TraceEnd::TestPeel { .. } => format!("{trace_name}_testpeel"),
        }
    }

    pub(super) fn bid(&self, gidx: CompiledGuardIdx) -> aot_ir::BBlockId {
        self.guards[gidx].bid()
    }

    pub(super) fn switch(&self, gidx: CompiledGuardIdx) -> Option<&Switch> {
        self.guards[gidx].switch.as_ref()
    }

    pub(super) fn guard(&self, gidx: CompiledGuardIdx) -> &J2CompiledGuard<Reg> {
        &self.guards[gidx]
    }

    pub(super) fn args_vlocs(&self) -> &[VarLocs<Reg>] {
        match &self.trace_start {
            J2TraceStart::ControlPoint { args_vlocs, .. } => args_vlocs,
            J2TraceStart::Guard { stack_off: _ } => todo!(),
        }
    }

    pub(super) fn guard_stack_off(&self, gidx: CompiledGuardIdx) -> u32 {
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

    pub(super) fn sidetrace_entry(&self, sidetrace_off: usize) -> *const u8 {
        self.codebuf.sidetrace_entry(sidetrace_off)
    }
}

impl<Reg: RegT + 'static> CompiledTrace for J2CompiledTrace<Reg> {
    fn ctrid(&self) -> TraceId {
        self.trid
    }

    fn as_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }

    fn guard(&self, gid: GuardId) -> &Guard {
        let gidx = CompiledGuardIdx::from(usize::from(gid));
        self.guards[gidx].guard()
    }

    // FIXME: This should really be handled in the backend, but the structure of CompiledTrace
    // / J2CompiledTrace makes this awkward.
    #[cfg(target_arch = "x86_64")]
    fn patch_guard(&self, gid: GuardId, tgt: *const std::ffi::c_void) {
        let gidx = CompiledGuardIdx::from(usize::from(gid));
        for patch_off in &self.guards[gidx].patch_offs {
            let patch_off = usize::try_from(*patch_off).unwrap();
            self.codebuf.patch(patch_off, 10, |patch_addr| {
                // We can only patch `mov r64, imm64`.
                assert_matches!(unsafe { patch_addr.read() }, 0x48 | 0x49);
                let patch_addr = unsafe { patch_addr.byte_add(2) };
                unsafe {
                    (patch_addr as *mut u64).write(u64::try_from(tgt.addr()).unwrap());
                }
            });
        }
    }

    fn entry(&self) -> *const c_void {
        self.codebuf.entry_ptr() as *const c_void
    }

    fn hl(&self) -> &std::sync::Weak<parking_lot::Mutex<HotLocation>> {
        &self.hl
    }

    fn code(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.codebuf.entry_ptr(), self.codebuf.len()) }
    }

    fn name(&self) -> String {
        self.symbol_name.clone()
    }
}

/// Where did this J2 compiled trace start?
#[derive(Debug)]
pub(super) enum J2TraceStart<Reg: RegT> {
    ControlPoint {
        entry_safepoint: &'static DeoptSafepoint,
        /// Every entry in `entry_safepoint.lives` will have an entry in `args_vlocs`, in order.
        /// In other words, `entry_safepoint.lives.iter().zip(args_vlocs.iter())` is guaranteed to
        /// work as expected.
        ///
        /// However, some variables will have empty `VarLocs`. In other words, while this coupler
        /// trace guarantees to accept variables being set in accordance with
        /// `entry_safepoint.lives`, it is also happy with a non-strict subset of those. That means
        /// that other traces jumping to this coupler trace only need to deal with the subset
        /// recorded in `args_vlocs` (i.e. they can ignore the superset in
        /// `entry_safepoint.lives`).
        args_vlocs: Vec<VarLocs<Reg>>,
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

    // j2 specific stuff.
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
    /// All of the instruction offsets in the associated machine code which will need to be patched
    /// when a sidetrace is compiled.
    patch_offs: SmallVec<[u32; 2]>,
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
        patch_offs: SmallVec<[u32; 2]>,
        extra_stack_len: u32,
        switch: Option<Switch>,
    ) -> Self {
        Self {
            bid,
            deopt_frames,
            deopt_vars,
            guard: Guard::new(),
            patch_offs,
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
}

/// The information about a frame necessary for deopt and side-tracing.
#[derive(Debug)]
pub(super) struct DeoptFrame {
    pub pc: InstId,
    pub pc_safepoint: &'static DeoptSafepoint,
}

impl PartialEq for DeoptFrame {
    fn eq(&self, other: &Self) -> bool {
        self.pc == other.pc && std::ptr::eq(self.pc_safepoint, other.pc_safepoint)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DeoptVar<Reg: RegT> {
    pub bitw: u32,
    pub fromvlocs: VarLocs<Reg>,
    pub tovlocs: VarLocs<Reg>,
}

index_vec::define_index_type! {
    pub(super) struct CompiledGuardIdx = u16;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::j2::{
        hir::{Block, TraceEnd},
        x64::Reg,
    };
    use index_vec::IndexVec;

    fn empty_block() -> Block {
        Block {
            insts: IndexVec::new(),
            guard_extras: IndexVec::new(),
        }
    }

    // Dummy DeoptSafepoint for testing
    static TEST_DEOPT_SAFEPOINT: DeoptSafepoint = DeoptSafepoint {
        id: 0,
        lives: Vec::new(),
    };

    #[test]
    fn test_get_trace_name_control_loop() {
        let start = J2TraceStart::ControlPoint::<Reg> {
            entry_safepoint: &TEST_DEOPT_SAFEPOINT,
            args_vlocs: vec![],
            stack_off: 0,
            sidetrace_off: 0,
        };
        let end = TraceEnd::Loop::<Reg> {
            entry: empty_block(),
            peel: None,
        };
        assert_eq!(
            J2CompiledTrace::<Reg>::get_trace_name(TraceId::from_u64(42), &start, &end, None, None),
            "__yk_trace_42_control_loop"
        );
    }

    #[test]
    fn test_format_trace_name_control_return() {
        let trid = TraceId::from_u64(42);
        let start = J2TraceStart::ControlPoint::<Reg> {
            entry_safepoint: &TEST_DEOPT_SAFEPOINT,
            args_vlocs: vec![],
            stack_off: 0,
            sidetrace_off: 0,
        };
        let end = TraceEnd::Return::<Reg> {
            entry: empty_block(),
            exit_safepoint: &TEST_DEOPT_SAFEPOINT,
        };
        assert_eq!(
            J2CompiledTrace::<Reg>::get_trace_name(trid, &start, &end, None, None),
            "__yk_trace_42_control_return"
        );
    }

    #[test]
    fn test_get_trace_name_guard_loop() {
        let trid = TraceId::from_u64(42);
        let start = J2TraceStart::Guard::<Reg> { stack_off: 0 };
        let end = TraceEnd::Loop::<Reg> {
            entry: empty_block(),
            peel: None,
        };
        assert_eq!(
            J2CompiledTrace::<Reg>::get_trace_name(
                trid,
                &start,
                &end,
                Some(TraceId::from_u64(5)),
                None
            ),
            "__yk_trace_42_guard_5_loop"
        );
    }

    #[test]
    fn test_get_trace_name_guard_return() {
        let trid = TraceId::from_u64(42_u64);
        let start = J2TraceStart::Guard::<Reg> { stack_off: 0 };
        let end = TraceEnd::Return::<Reg> {
            entry: empty_block(),
            exit_safepoint: &TEST_DEOPT_SAFEPOINT,
        };
        assert_eq!(
            J2CompiledTrace::<Reg>::get_trace_name(
                trid,
                &start,
                &end,
                Some(TraceId::from_u64(5_u64)),
                None
            ),
            "__yk_trace_42_guard_5_return"
        );
    }
}
