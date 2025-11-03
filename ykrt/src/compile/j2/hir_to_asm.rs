//! High-level HIR to asm.
//!
//! This module is where j2 does the high-level parts of HIR to asm conversion ([HirToAsm]). It has
//! no understanding of backend details, which are hidden behind the [HirToAsmBackend] trait.
//! Backends (e.g. x64) then implement that trait to do all the platform specific things they need
//! and want to.
//!
//! We expect backends to want to perform optimisations of their own, which means that they may be
//! able to sometimes combine *m* HIR instructions into *n* machine instruction where *m > n*. When
//! this is done on the main body of a trace, it opens up further opportunities to push
//! instructions into a guard body. Because of this, the mechanism for assembling a trace is that
//! we first process the trace's main body and then process guard bodies.
//!
//! To keep things simple, this module makes the following assumptions about the machine code which
//! is generated:
//!
//!   1. Code generation is performed backwards.
//!   2. Loop traces are of the form:
//!      ```text
//!      <adjust stack pointer>
//!      start_label:
//!      <body of loop>
//!      jmp start_label
//!      <guard body n>
//!      ...
//!      <guard body 1>
//!      ```

use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        CompilationError, CompiledTrace,
        j2::{
            compiled_trace::{DeoptFrame, J2CompiledGuard, J2CompiledTrace, J2CompiledTraceKind},
            hir::*,
            regalloc::{RegAlloc, RegFill, RegT, SnapshotIdx, VarLocs},
        },
        jitc_yk::aot_ir::{self},
    },
    location::HotLocation,
    log::{IRPhase, log_ir, should_log_ir},
    mt::{MT, TraceId},
};
use index_vec::IndexVec;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use std::{ffi::c_void, sync::Arc};

pub(super) struct HirToAsm<'a, AB: HirToAsmBackend> {
    m: &'a Mod<AB::Reg>,
    hl: Arc<Mutex<HotLocation>>,
    be: AB,
}

impl<'a, AB: HirToAsmBackend> HirToAsm<'a, AB> {
    pub(super) fn new(m: &'a Mod<AB::Reg>, hl: Arc<Mutex<HotLocation>>, be: AB) -> Self {
        Self { m, hl, be }
    }

    pub(super) fn build(mut self, mt: Arc<MT>) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let logging = should_log_ir(IRPhase::Asm);
        let (buf, guards, log, modkind) = match &self.m.kind {
            ModKind::Loop {
                entry_safepoint,
                entry,
                inner: body,
            } => {
                let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
                // FIXME: Relying on stackmap 0 being the control point is a horrible hack.
                let base_stack_off = u32::try_from({
                    let (smap, prologue) = aot_smaps.get(0);
                    if prologue.hasfp {
                        // The frame size includes RBP, but we only want to include the
                        // local variables.
                        smap.size - 8
                    } else {
                        smap.size
                    }
                })
                .unwrap();

                assert!(body.is_none());
                let (rec, _) = aot_smaps.get(usize::try_from(entry_safepoint.id).unwrap());
                let entry_vlocs = rec
                    .live_vals
                    .iter()
                    .map(|smap| AB::smp_to_vloc(smap))
                    .collect::<Vec<_>>();

                // Create the backwards jump at the end of a loop trace.
                let loop_label = self.be.loop_backwards_jump()?;

                // Assemble the body
                let (guards, body_ra) = self.p_block(
                    entry,
                    base_stack_off,
                    &entry_vlocs,
                    &entry_vlocs,
                    true,
                    logging,
                )?;
                let entry_stack_off = body_ra.stack_off();
                self.be
                    .body_completed(Some(loop_label.clone()), entry_stack_off - base_stack_off);
                self.asm_guards(entry, guards, body_ra)?;
                let (buf, guards, log, label_offs) = self.be.build_exe(logging, &[loop_label])?;
                let [sidetrace_off] = &*label_offs else {
                    panic!()
                };
                let modkind = J2CompiledTraceKind::Loop {
                    entry_vlocs,
                    entry_safepoint,
                    stack_off: entry_stack_off,
                    sidetrace_off: *sidetrace_off,
                };
                (buf, guards, log, modkind)
            }
            ModKind::Side {
                entry_vlocs,
                entry,
                src_ctr,
                src_gridx,
                tgt_ctr,
            } => {
                let stack_off = src_ctr.guard_stack_off(*src_gridx);
                self.be.sidetrace_end(tgt_ctr)?;
                // Assemble the body
                let exit_vlocs = tgt_ctr.entry_vlocs();
                let (guards, body_ra) =
                    self.p_block(entry, stack_off, entry_vlocs, exit_vlocs, true, logging)?;
                let body_stack_off = body_ra.stack_off();
                self.be.body_completed(None, body_stack_off - stack_off);
                self.asm_guards(entry, guards, body_ra)?;
                let modkind = J2CompiledTraceKind::Side { stack_off };
                let (buf, guards, log, _) = self.be.build_exe(logging, &[])?;
                (buf, guards, log, modkind)
            }
            ModKind::Coupler { .. } => todo!(),
            #[cfg(test)]
            ModKind::Test { .. } => panic!(),
        };

        if logging {
            let ds = if let Some(x) = &self.hl.lock().debug_str {
                format!(": {}", x.as_str())
            } else {
                "".to_owned()
            };
            log_ir(&format!(
                "--- Begin jit-asm{ds} ---\n{}\n--- End jit-asm ---\n",
                log.unwrap()
            ));
        }

        Ok(Arc::new(J2CompiledTrace::<AB::Reg>::new(
            mt,
            self.m.trid,
            Arc::downgrade(&self.hl),
            buf,
            guards,
            modkind,
        )))
    }

    #[cfg(test)]
    pub(super) fn build_test(mut self) -> Result<AB::BuildTest, CompilationError> {
        match &self.m.kind {
            ModKind::Test { entry_vlocs, block } => {
                // Assemble the body
                let (guards, body_ra) =
                    self.p_block(block, 0, entry_vlocs, entry_vlocs, true, true)?;
                self.be.body_completed(None, 0);

                // Guards
                for (gd, _grestore) in guards.into_iter().rev().zip(self.m.guard_restores()) {
                    let mut ra = body_ra.from_snapshot(gd.sidx);
                    let patch_label =
                        self.be
                            .guard_end(&mut ra, block, TraceId::testing(), gd.gridx)?;
                    self.be.guard_completed(
                        gd.label,
                        patch_label,
                        ra.stack_off() - body_ra.stack_off(),
                        gd.bid,
                        SmallVec::new(),
                        None,
                    );
                }

                Ok(self.be.build_test(&[]))
            }
            _ => unreachable!(),
        }
    }

    /// Assemble guards.
    fn asm_guards(
        &mut self,
        entry: &Block,
        grestores: Vec<AsmGuardRestore<AB>>,
        body_ra: RegAlloc<AB>,
    ) -> Result<(), CompilationError> {
        assert_eq!(grestores.len(), self.m.guard_restores().len());
        let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
        for (asmgrestore, grestore) in grestores.into_iter().rev().zip(self.m.guard_restores()) {
            let mut ra = body_ra.from_snapshot(asmgrestore.sidx);
            let exit_var_regs = grestore
                .exit_frames
                .iter()
                .flat_map(|dfr| {
                    dfr.exit_vars
                        .iter()
                        .filter_map(|iidx| ra.iter_reg_for(*iidx).nth(0).map(|x| (*iidx, x)))
                })
                .collect::<Vec<_>>();

            let patch_label = self
                .be
                .guard_end(&mut ra, entry, self.m.trid, asmgrestore.gridx)?;
            for (_iidx, reg) in exit_var_regs {
                ra.ensure_spilled_for_deopt(&mut self.be, reg)?;
            }

            let deopt_frames = grestore
                .exit_frames
                .iter()
                .map(
                    |Frame {
                         safepoint,
                         call_iid,
                         func,
                         exit_vars,
                     }| {
                        let smap = aot_smaps.get(usize::try_from(safepoint.id).unwrap()).0;
                        assert_eq!(exit_vars.len(), safepoint.lives.len());
                        assert_eq!(exit_vars.len(), smap.live_vals.len());
                        DeoptFrame {
                            safepoint,
                            call_iid: call_iid.to_owned(),
                            func: *func,
                            vars: exit_vars
                                .iter()
                                .zip(safepoint.lives.iter().zip(smap.live_vals.iter()))
                                .map(|(iidx, (aot_op, smap_loc))| {
                                    let fromvlocs = ra.varlocs_for_deopt(*iidx);
                                    let mut tovlocs = AB::smp_to_vloc(smap_loc);
                                    if fromvlocs == tovlocs {
                                        // Optimise away situations where we would just move a
                                        // value from VLoc X to VLoc X.
                                        tovlocs = VarLocs::new(smallvec![]);
                                    }
                                    (
                                        aot_op.to_inst_id(),
                                        entry.inst_bitw(self.m, *iidx),
                                        fromvlocs,
                                        tovlocs,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        }
                    },
                )
                .collect::<SmallVec<[_; 1]>>();
            self.be.guard_completed(
                asmgrestore.label,
                patch_label,
                ra.stack_off() - body_ra.stack_off(),
                asmgrestore.bid,
                deopt_frames,
                asmgrestore.switch.map(|x| *x),
            );
        }

        Ok(())
    }

    /// Returns the guards in reverse order.
    fn p_block(
        &mut self,
        b: &'a Block,
        stack_off: u32,
        entry_vlocs: &[VarLocs<AB::Reg>],
        exit_vlocs: &[VarLocs<AB::Reg>],
        guards_allowed: bool,
        logging: bool,
    ) -> Result<(Vec<AsmGuardRestore<AB>>, RegAlloc<'a, AB>), CompilationError> {
        let mut ra = RegAlloc::<AB>::new(self.m, b, stack_off);
        ra.set_entry_stacks_at_end(entry_vlocs);

        // As we process the trace, we push [AsmGuardRestore]s in here with a 1:1 mapping to
        // [self.m.guard_restores]. However, because we're iterating backwards over the trace,
        // that means that the indexes will be backwards too! This causes us to do some indexing
        // acrobatics in the treatment of [Guard]s below.
        let mut grestores = Vec::new();
        let mut insts_iter = b.insts_iter(..).rev().peekable();
        for _ in entry_vlocs.len()..b.insts_len() {
            let Some((iidx, hinst)) = insts_iter.next() else {
                // By definition there must be no `Arg` instructions in this trace, which is only
                // plausible in testing mode.
                #[cfg(not(test))]
                panic!();
                #[cfg(test)]
                break;
            };

            match hinst {
                Inst::Abs(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_abs(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Add(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_add(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::And(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_and(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Arg(_) => {
                    // These are handled specially after this loop.
                    unreachable!();
                }
                Inst::AShr(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_ashr(&mut ra, b, iidx, x)?;
                    }
                }
                #[cfg(test)]
                Inst::BlackBox(BlackBox { val }) => {
                    ra.blackbox(iidx, *val);
                }
                Inst::Call(x) => self.be.i_call(&mut ra, b, iidx, x)?,
                Inst::Const(_) => {
                    ra.alloc_const(&mut self.be, iidx)?;
                }
                Inst::CtPop(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_ctpop(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::DynPtrAdd(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_dynptradd(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Exit(Exit(exit_vars)) => {
                    let is_loop = match &self.m.kind {
                        ModKind::Loop { .. } => true,
                        ModKind::Side { .. } | ModKind::Coupler { .. } => false,
                        #[cfg(test)]
                        ModKind::Test {
                            entry_vlocs: _,
                            block: _,
                        } => false,
                    };
                    ra.set_exit_vlocs(is_loop, iidx, exit_vars, exit_vlocs);
                }
                Inst::FAdd(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fadd(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FCmp(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fcmp(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FDiv(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fdiv(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FMul(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fmul(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FSub(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fsub(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FPExt(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fpext(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FPToSI(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fptosi(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Guard(
                    x @ Guard {
                        entry_vars: _,
                        gridx,
                        bid,
                        switch,
                        ..
                    },
                ) => {
                    assert!(guards_allowed);
                    assert!(grestores.len() < self.m.guard_restores().len());
                    // We now have to be careful about guard indexes due to backwards iteration.
                    // We may already have processed `gridx` (remember: multiple [Guard]s can map
                    // to a single [GuardRestore]), so we need to check for that, but `gridx` will
                    // be a "forwards" index, and `grestores` will have a "backwards" index.
                    let cnd_idx = self.m.guard_restores().len() - usize::from(*gridx) - 1;
                    assert!(cnd_idx <= grestores.len());
                    if cnd_idx == grestores.len() {
                        let label = self.be.i_guard(&mut ra, b, iidx, x)?;
                        let sidx = ra.snapshot(iidx, *gridx);
                        grestores.push(AsmGuardRestore {
                            gridx: *gridx,
                            label,
                            sidx,
                            bid: *bid,
                            switch: switch.clone(),
                        });
                    } else {
                        // We're mapping multiple [Guard]s to a single [GuardRestore]. The
                        // challenge then becomes that the register allocator's snapshot might --
                        // and almost certainly will! -- be different at each guard. In such cases
                        // we will need to create an intermediate piece of code which harmonises
                        // the machine state so that they can then all use the same side-trace. [As
                        // an optimisation we could allow them to have different machine states for
                        // deopt, and only harmonise the machine state when a side-trace is
                        // present, but that's probably quite a lot of effort for not much gain.]
                        todo!()
                    }
                }
                Inst::ICmp(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_icmp(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::IntToPtr(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_inttoptr(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Load(x @ Load { is_volatile, .. }) => {
                    if *is_volatile || ra.is_used(iidx) {
                        self.be.i_load(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::LShr(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_lshr(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Mul(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_mul(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Or(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_or(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::PtrAdd(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_ptradd(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::PtrToInt(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_ptrtoint(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Select(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_select(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::SExt(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_sext(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Shl(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_shl(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::SIToFP(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_sitofp(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::SRem(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_srem(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Store(x) => self.be.i_store(&mut ra, b, iidx, x)?,
                Inst::Sub(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_sub(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::ThreadLocal(ThreadLocal(addr)) => {
                    if ra.is_used(iidx) {
                        let tloff = AB::thread_local_off(*addr);
                        self.be.i_threadlocal(&mut ra, b, iidx, tloff)?;
                    }
                }
                Inst::Trunc(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_trunc(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::UDiv(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_udiv(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::Xor(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_xor(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::ZExt(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_zext(&mut ra, b, iidx, x)?;
                    }
                }
            }
            if logging {
                let ty = hinst.ty(self.m);
                if ty == &Ty::Void {
                    self.be.log(hinst.to_string(self.m, b));
                } else {
                    self.be.log(format!(
                        "%{}: {} = {}",
                        usize::from(iidx),
                        ty.to_string(self.m),
                        hinst.to_string(self.m, b)
                    ));
                }
            }
        }

        // We deal with [Arg] instructions specially: they're best thought of as
        // pseudo-instructions in the sense that they define variables but don't directly
        // generate code themselves.
        ra.set_entry_vlocs_at_start(&mut self.be, entry_vlocs);
        if logging {
            for (iidx, hinst) in insts_iter {
                let ty = hinst.ty(self.m);
                let pp_vlocs = entry_vlocs[usize::from(iidx)]
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                if ty == &Ty::Void {
                    todo!();
                    // self.be.log(hinst.to_string(self.m, b));
                } else {
                    self.be.log(format!(
                        "%{}: {} = {} [{pp_vlocs}]",
                        usize::from(iidx),
                        ty.to_string(self.m),
                        hinst.to_string(self.m, b),
                    ));
                }
            }
        }

        Ok((grestores, ra))
    }
}

/// The trait that backends must implement to assemble a trace into machine code.
///
/// Backends must work with the following assumptions:
///
///   1. A fully assembled trace must appear to this module as a linear sequence of (abstract)
///      machine instructions, where each machine instruction can have one or more `Label`s
///      attached to it.
///   2. A fully compiled trace is a sequence of 1 or more blocks. The first block to be processed
///      is guaranteed to be the main body of the trace; subsequent blocks are guaranteed to be
///      guard bodies. Each block is assembled in reverse order.
///   3. After each block has been fully processed, [HirToAsmBackend::block_completed] will be
///      called.
pub(super) trait HirToAsmBackend {
    type Reg: RegT + 'static;

    /// A relocation label. These are generated by some functions (e.g.
    /// [Self::loop_backwards_jump]` and consumed by others (e.g. [Self::body_completed]). What
    /// exactly they mean is entirely up to the backend but one would expect that they are attached
    /// to backend instructions when generated, and when attached they cause jumps / calls (etc.)
    /// to be updated with a concrete address.
    type Label: Clone;

    #[cfg(test)]
    type BuildTest;

    /// Convert a yksmp stackmap to one or more `VarLoc`s.
    fn smp_to_vloc(smp_locs: &SmallVec<[yksmp::Location; 1]>) -> VarLocs<Self::Reg>;
    fn thread_local_off(addr: *const c_void) -> u32;

    /// Assemble everything into machine code. If `log` is `true`, return `Some(log)`. For each
    /// label [Self::Label] in `labels`, return a `usize` offset into the executable buffer for
    /// that label.
    fn build_exe(
        self,
        log: bool,
        labels: &[Self::Label],
    ) -> Result<
        (
            *mut c_void,
            IndexVec<GuardRestoreIdx, J2CompiledGuard<Self::Reg>>,
            Option<String>,
            Vec<usize>,
        ),
        CompilationError,
    >;

    /// For testing purposes only, this method will be called by [HirToAsm::build_test] instead of
    /// [Self::build_exe].
    #[cfg(test)]
    fn build_test(self, labels: &[Self::Label]) -> Self::BuildTest;

    /// Return an iterator of the registers that `iidx` can end up in. It is acceptable to
    /// return a subset of the possible registers (indeed, returning the empty list is
    /// semantically correct, though may cause other problems) *but* it is unacceptable to
    /// return a register that `iidx` is not allowed to end up in.
    fn iter_possible_regs(&self, b: &Block, iidx: InstIdx) -> impl Iterator<Item = Self::Reg>;

    fn log(&mut self, s: String);

    /// If the constant `c` will need a temporary register in order to put it into `reg`, return an
    /// iterator with the possible temporary registers. One of those will be selected and passed to
    /// [Self::move_const]. Note: this may end up spilling other registers, so if you can avoid
    /// allocating a temporary register, you should probably avoid doing so.
    fn const_needs_tmp_reg(
        &self,
        reg: Self::Reg,
        c: &ConstKind,
    ) -> Option<impl Iterator<Item = Self::Reg>>;
    /// Move the constant `c` into `bitw` bits of `reg`, filling upper bits as per `tgt_fill`. If
    /// [Self::const_needs_tmp_reg] returned `Some`, then a temporary register will be passed as
    /// `Some(Self::Reg))`
    fn move_const(
        &mut self,
        reg: Self::Reg,
        tmp_reg: Option<Self::Reg>,
        tgt_bitw: u32,
        tgt_fill: RegFill,
        c: &ConstKind,
    ) -> Result<(), CompilationError>;

    /// Move the [super::regalloc::VarLoc::StackOff] address `stack_off` into `reg`.
    fn move_stackoff(&mut self, reg: Self::Reg, stack_off: u32) -> Result<(), CompilationError>;

    /// Adjust `dst_bitw` bits of `reg` from `src_fill` to `dst_fill`.
    fn arrange_fill(&mut self, reg: Self::Reg, src_fill: RegFill, dst_bitw: u32, dst_fill: RegFill);

    /// Copy `from_reg` to `to_reg`.
    fn copy_reg(&mut self, from_reg: Self::Reg, to_reg: Self::Reg) -> Result<(), CompilationError>;

    /// If the stack is currently at offset `stack_off`, return an aligned offset suitable for a
    /// value of `bitw` bits: the value will be stored exactly at the returned offset. Note: this
    /// function can be called both before and after a value has been spilled, and should return
    /// the same value in all situations!
    ///
    /// In general, the backend will need to align for a value of `bitw` and account for storing
    /// `bitw` itself. For example, on x64, `align_spill(1, 64)` will return `((1 +
    /// 8).next_multiple_of(8)) == 16`, so there will be 7 bytes of padding before the first byte
    ///   of the spilt value.
    fn align_spill(&self, stack_off: u32, bitw: u32) -> u32;

    /// Spill `bitw` bits in `reg`, whose fill is `in_fill` to `stack_off` (which may or may not be
    /// aligned for `bitw`). Note: this function may leave CPU flags in an undefined state.
    fn spill(
        &mut self,
        reg: Self::Reg,
        in_fill: RegFill,
        stack_off: u32,
        bitw: u32,
    ) -> Result<(), CompilationError>;

    /// Unspill `bitw` bits from `stack_off` to `reg` with a fill of `out_fill`.
    fn unspill(
        &mut self,
        stack_off: u32,
        reg: Self::Reg,
        out_fill: RegFill,
        bitw: u32,
    ) -> Result<(), CompilationError>;

    /// Produce code for the backwards jump that finishes a loop trace.
    fn loop_backwards_jump(&mut self) -> Result<Self::Label, CompilationError>;

    /// Produce code for the jump to `tgt_ctr` at the end of a side-trace.
    fn sidetrace_end(
        &mut self,
        ctr: &Arc<J2CompiledTrace<Self::Reg>>,
    ) -> Result<(), CompilationError>;

    /// Produce code for the end of a guard: return a label for the instruction to be patched when
    /// a side-trace is produced.
    fn guard_end(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        trid: TraceId,
        gridx: GuardRestoreIdx,
    ) -> Result<Self::Label, CompilationError>;

    /// The current guard has been completed:
    ///   * `start_label` should be set to the beginning of the guard body.
    ///   * `patch_label` should be set to the instruction to be patched when a side-trace is
    ///     created.
    ///   * the guard body will consume `stack_off` additional bytes of stack space.
    fn guard_completed(
        &mut self,
        start_label: Self::Label,
        patch_label: Self::Label,
        stack_off: u32,
        bid: aot_ir::BBlockId,
        deopt_frames: SmallVec<[DeoptFrame<Self::Reg>; 1]>,
        switch: Option<Switch>,
    );

    /// The current body of a trace has been completed and has consumed `stack_off` additional
    /// bytes of stack space. If `label` is `Some`, it will be attached to the first instruction
    /// after the stack is adjusted.
    fn body_completed(&mut self, label: Option<Self::Label>, stack_off: u32);

    fn i_abs(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Abs,
    ) -> Result<(), CompilationError>;

    fn i_add(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Add,
    ) -> Result<(), CompilationError>;

    fn i_and(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &And,
    ) -> Result<(), CompilationError>;

    fn i_ashr(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &AShr,
    ) -> Result<(), CompilationError>;

    fn i_call(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Call,
    ) -> Result<(), CompilationError>;

    fn i_ctpop(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &CtPop,
    ) -> Result<(), CompilationError>;

    fn i_dynptradd(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &DynPtrAdd,
    ) -> Result<(), CompilationError>;

    fn i_fadd(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FAdd,
    ) -> Result<(), CompilationError>;

    fn i_fcmp(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FCmp,
    ) -> Result<(), CompilationError>;

    fn i_fdiv(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FDiv,
    ) -> Result<(), CompilationError>;

    fn i_fmul(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FMul,
    ) -> Result<(), CompilationError>;

    fn i_fsub(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FSub,
    ) -> Result<(), CompilationError>;

    fn i_fpext(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FPExt,
    ) -> Result<(), CompilationError>;

    fn i_fptosi(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FPToSI,
    ) -> Result<(), CompilationError>;

    /// The instruction should use [RegCnstr::KeepAlive] for the values in `Guard::entry_vars`.
    ///
    /// The label returned should be the jump instruction to the guard body.
    fn i_guard(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Guard,
    ) -> Result<Self::Label, CompilationError>;

    fn i_icmp(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &ICmp,
    ) -> Result<(), CompilationError>;

    fn i_inttoptr(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &IntToPtr,
    ) -> Result<(), CompilationError>;

    fn i_load(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Load,
    ) -> Result<(), CompilationError>;

    fn i_lshr(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &LShr,
    ) -> Result<(), CompilationError>;

    fn i_mul(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Mul,
    ) -> Result<(), CompilationError>;

    fn i_or(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Or,
    ) -> Result<(), CompilationError>;

    fn i_ptradd(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &PtrAdd,
    ) -> Result<(), CompilationError>;

    fn i_ptrtoint(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &PtrToInt,
    ) -> Result<(), CompilationError>;

    fn i_select(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Select,
    ) -> Result<(), CompilationError>;

    fn i_sext(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &SExt,
    ) -> Result<(), CompilationError>;

    fn i_shl(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Shl,
    ) -> Result<(), CompilationError>;

    fn i_sitofp(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &SIToFP,
    ) -> Result<(), CompilationError>;

    fn i_srem(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &SRem,
    ) -> Result<(), CompilationError>;

    fn i_store(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Store,
    ) -> Result<(), CompilationError>;

    fn i_sub(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Sub,
    ) -> Result<(), CompilationError>;

    fn i_threadlocal(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        tl_off: u32,
    ) -> Result<(), CompilationError>;

    fn i_trunc(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Trunc,
    ) -> Result<(), CompilationError>;

    fn i_udiv(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &UDiv,
    ) -> Result<(), CompilationError>;

    fn i_xor(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Xor,
    ) -> Result<(), CompilationError>;

    fn i_zext(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &ZExt,
    ) -> Result<(), CompilationError>;
}

index_vec::define_index_type! {
    pub(super) struct FrameIdx = u16;
}

#[derive(Debug)]
struct AsmGuardRestore<AB: HirToAsmBackend + ?Sized> {
    gridx: GuardRestoreIdx,
    label: AB::Label,
    sidx: SnapshotIdx,
    bid: aot_ir::BBlockId,
    switch: Option<Box<Switch>>,
}
