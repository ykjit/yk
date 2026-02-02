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
//!
//! ### Output format of different trace kinds
//!
//! Traces are represented as ([TraceStart], [TraceEnd]) pairs.
//!
//! ## (ControlPoint, Coupler) traces
//!
//! These traces take the form:
//!
//! ```text
//! <stack adjustment>
//! iter0_label:
//! <instrs>
//! jmp tgt_ctr.iter0_label
//! <guard body n>
//! ...
//! <guard body 1>
//! call __yk_j2_deopt
//! <data>
//! ```
//!
//! ## (ControlPoint, Loop) traces
//!
//! These traces take the form:
//!
//! ```text
//! <stack adjustment>
//! iter0_label:
//! <instrs for first iteration of the loop>
//! iter1_label:
//! <instrs for the first peeled iteration of the loop>
//! jmp iter1_label
//! <guard body n>
//! ...
//! <guard body 1>
//! call __yk_j2_deopt
//! <data>
//! ```
//!
//! If peeling has not occurred, there will be no difference between `iter0_label` and
//! `iter1_label`.
//!
//!
//! ## (ControlPoint, Return) traces
//!
//! These traces take the form:
//!
//! ```text
//! <stack adjustment>
//! iter0_label:
//! <instrs>
//! return
//! <guard body n>
//! ...
//! <guard body 1>
//! call __yk_j2_deopt
//! <data>
//! ```
//!
//! Where `return` is a machine level `return`.
//!
//!
//! ## (Guard, Coupler) traces
//!
//! These traces take the form:
//!
//! ```text
//! <instrs>
//! <stack adjustment>
//! jmp tgt_ctr.iter0_label
//! <guard body n>
//! ...
//! <guard body 1>
//! call __yk_j2_deopt
//! <data>
//! ```

use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        CompilationError, CompiledTrace, DeoptSafepoint,
        j2::{
            codebuf::ExeCodeBuf,
            compiled_trace::{
                CompiledGuardIdx, DeoptFrame, DeoptVar, J2CompiledGuard, J2CompiledTrace,
                J2TraceStart,
            },
            effects::Effects,
            hir::*,
            regalloc::{RegAlloc, RegFill, RegT, VarLoc, VarLocs},
        },
        jitc_yk::aot_ir::{self},
    },
    location::HotLocation,
    log::{IRPhase, log_ir, should_log_ir},
    mt::{MT, TraceId},
    varlocs,
};
use index_vec::IndexVec;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{ffi::c_void, sync::Arc};
use test_stubs::test_stubs;
use vob::Vob;

pub(super) struct HirToAsm<'a, AB: HirToAsmBackend> {
    m: &'a Mod<AB::Reg>,
    hl: Arc<Mutex<HotLocation>>,
    be: AB,
    /// The intermediate [AsmGuard] for each [Guard] block in a trace's "main" (i.e. non-[Guard])
    /// blocks. These use [GuardBlockIdx] to emphasise that there is a 1:1 mapping between
    /// [GuardExtra::gbidx] and this [IndexVec].
    /// These will initially be set to `None`; as the main blocks are processed, they will be set
    /// to `Some`. Note: [Self::asm_guards] will empty this [IndexVec] completely.
    asmguards: IndexVec<CompiledGuardIdx, AsmGuard<'a, AB>>,
}

impl<'a, AB: HirToAsmBackend> HirToAsm<'a, AB> {
    pub(super) fn new(m: &'a Mod<AB::Reg>, hl: Arc<Mutex<HotLocation>>, be: AB) -> Self {
        Self {
            m,
            hl,
            be,
            asmguards: IndexVec::new(),
        }
    }

    pub(super) fn build(mut self, mt: Arc<MT>) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let logging = should_log_ir(IRPhase::Asm);
        let (buf, guards, log, trace_start) = match &self.m.trace_start {
            TraceStart::ControlPoint { entry_safepoint } => {
                let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
                // FIXME: Relying on stackmap 0 being the control point is a horrible hack.
                let base_stack_off = u32::try_from({
                    let (smap, prologue) = aot_smaps.get(0);
                    if prologue.hasfp {
                        // FIXME: This needs porting! https://github.com/ykjit/yk/issues/1936
                        #[cfg(not(target_arch = "x86_64"))]
                        panic!();
                        // The frame size includes RBP, but we only want to include the
                        // local variables.
                        smap.size - 8
                    } else {
                        smap.size
                    }
                })
                .unwrap();

                let (rec, _) = aot_smaps.get(usize::try_from(entry_safepoint.id).unwrap());
                let mut entry_vlocs = rec
                    .live_vals
                    .iter()
                    .map(|smap| AB::smp_to_vloc(smap, RegFill::Undefined))
                    .collect::<Vec<_>>();

                for vlocs in entry_vlocs.iter_mut() {
                    vlocs.retain(|x| {
                        if let VarLoc::Reg(reg, _fill) = x
                            && reg.is_caller_saved()
                        {
                            // Because of the way we call traces (see bc59d8bff411931440459fa3377a137e8537a32f
                            // for details), caller saved registers are potentially corrupted at the very start
                            // of ControlPoint traces
                            false
                        } else {
                            true
                        }
                    });
                }

                let (post_stack_label, entry_stack_off) = match &self.m.trace_end {
                    TraceEnd::Coupler { entry, tgt_ctr } => {
                        let mut ra = RegAlloc::<AB>::new(self.m, entry, base_stack_off);
                        ra.set_entry_stacks_at_end(&entry_vlocs);
                        self.be.star_coupler_end(tgt_ctr)?;
                        ra.set_term_vlocs(
                            &mut self.be,
                            entry,
                            false,
                            &entry_vlocs,
                            tgt_ctr.entry_vlocs(),
                        )?;
                        let entry_stack_off =
                            self.p_block(entry, Some(entry), ra, &entry_vlocs, logging)?;
                        let post_stack_label = self.be.controlpoint_coupler_or_return_start(
                            entry_stack_off - base_stack_off,
                        )?;
                        (post_stack_label, entry_stack_off)
                    }
                    TraceEnd::Loop { entry, peel } => {
                        assert!(peel.is_none());
                        let mut ra = RegAlloc::<AB>::new(self.m, entry, base_stack_off);
                        ra.set_entry_stacks_at_end(&entry_vlocs);
                        let iter0_label = self.be.controlpoint_loop_end()?;
                        ra.set_term_vlocs(&mut self.be, entry, true, &entry_vlocs, &entry_vlocs)?;
                        let entry_stack_off =
                            self.p_block(entry, Some(entry), ra, &entry_vlocs, logging)?;
                        self.be.controlpoint_loop_start(
                            iter0_label.clone(),
                            entry_stack_off - base_stack_off,
                        );
                        (iter0_label, entry_stack_off)
                    }
                    TraceEnd::Return {
                        entry,
                        exit_safepoint,
                    } => {
                        let mut ra = RegAlloc::<AB>::new(self.m, entry, base_stack_off);
                        ra.set_entry_stacks_at_end(&entry_vlocs);
                        assert!(entry.term_vars().is_empty());
                        self.be.star_return_end(exit_safepoint)?;
                        ra.set_term_vlocs(&mut self.be, entry, false, &entry_vlocs, &[])?;
                        let entry_stack_off =
                            self.p_block(entry, Some(entry), ra, &entry_vlocs, logging)?;
                        let post_stack_label = self.be.controlpoint_coupler_or_return_start(
                            entry_stack_off - base_stack_off,
                        )?;
                        (post_stack_label, entry_stack_off)
                    }
                    #[cfg(test)]
                    TraceEnd::Test { .. } => todo!(),
                };

                self.asm_guards(logging)?;
                let (buf, guards, log, label_offs) =
                    self.be.build_exe(logging, &[post_stack_label])?;
                let [sidetrace_off] = &*label_offs else {
                    panic!()
                };
                let trace_start = J2TraceStart::ControlPoint {
                    entry_vlocs,
                    entry_safepoint,
                    stack_off: entry_stack_off,
                    sidetrace_off: *sidetrace_off,
                };
                (buf, guards, log, trace_start)
            }
            TraceStart::Guard {
                src_ctr,
                src_gidx: src_gridx,
                entry_vlocs,
            } => {
                let src_stack_off = src_ctr.guard_stack_off(*src_gridx);
                let entry_stack_off = match &self.m.trace_end {
                    TraceEnd::Coupler { entry, tgt_ctr } => {
                        let mut ra = RegAlloc::<AB>::new(self.m, entry, src_stack_off);
                        ra.set_entry_stacks_at_end(entry_vlocs);
                        self.be.star_coupler_end(tgt_ctr)?;
                        ra.set_term_vlocs(
                            &mut self.be,
                            entry,
                            false,
                            entry_vlocs,
                            tgt_ctr.entry_vlocs(),
                        )?;
                        self.p_block(entry, Some(entry), ra, entry_vlocs, logging)?
                    }
                    TraceEnd::Loop { .. } => unreachable!(),
                    TraceEnd::Return {
                        entry,
                        exit_safepoint,
                    } => {
                        let mut ra = RegAlloc::<AB>::new(self.m, entry, src_stack_off);
                        ra.set_entry_stacks_at_end(entry_vlocs);
                        self.be.star_return_end(exit_safepoint)?;
                        ra.set_term_vlocs(&mut self.be, entry, false, entry_vlocs, &[])?;
                        self.p_block(entry, Some(entry), ra, entry_vlocs, logging)?
                    }
                    #[cfg(test)]
                    TraceEnd::Test { .. } => todo!(),
                };
                self.be.guard_coupler_start(entry_stack_off - src_stack_off);
                self.asm_guards(logging)?;
                let modkind = J2TraceStart::Guard {
                    stack_off: entry_stack_off,
                };
                let (buf, guards, log, labels) = self.be.build_exe(logging, &[])?;
                assert!(labels.is_empty());
                (buf, guards, log, modkind)
            }
            #[cfg(test)]
            TraceStart::Test => unreachable!(),
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
            trace_start,
        )))
    }

    #[cfg(test)]
    pub(super) fn build_test(mut self) -> Result<AB::BuildTest, CompilationError> {
        let TraceEnd::Test { entry_vlocs, block } = &self.m.trace_end else {
            panic!()
        };
        // Assemble the body
        let mut ra = RegAlloc::<AB>::new(self.m, block, 0);
        ra.set_entry_stacks_at_end(entry_vlocs);
        // Currently we don't force tests to end with [Term] instructions.
        if let Inst::Term(_) = block.insts.last().unwrap() {
            ra.set_term_vlocs(&mut self.be, block, true, entry_vlocs, entry_vlocs)?;
        }
        let stack_off = self.p_block(block, Some(block), ra, entry_vlocs, true)?;
        self.be.guard_coupler_start(stack_off);

        // Guards

        self.asm_guards(true)?;

        Ok(self.be.build_test(&[]))
    }

    /// Assemble guards.
    fn asm_guards(&mut self, logging: bool) -> Result<(), CompilationError> {
        // For each guard we've encountered while assembling the main [Block]s, we now get the
        // backend to produce code for the associated guard body. We've already identified which
        // instructions should end up in the guard body, so this isn't too difficult: we create a
        // temporary [Block] with the appropriate instructions, work out the [VarLoc]s, and call
        // [Self::p_block].

        // Since we're creating temporary [Block]s, we can reuse the allocation for its
        // instructions and term_vars, so we hoist these out of the loop.
        let mut ginsts: IndexVec<InstIdx, Inst> = IndexVec::new();
        let mut gterms: Vec<InstIdx> = Vec::new();

        // There is a little bit of awkwardness in testing mode, where there are no stackmaps:
        // instead [Mod::smaps] contains stackmaps specified by the test author.
        #[cfg(not(test))]
        let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();

        let aguards = std::mem::take(&mut self.asmguards);
        for (gidx, aguard) in aguards.into_iter_enumerated() {
            let patch_label = self
                .be
                .guard_end(self.m.trid, CompiledGuardIdx::from(usize::from(gidx)))?;
            let gextra = aguard.block.gextra(aguard.geidx);

            // The temporary `Block` that is the guard body. This will end up as:
            //
            // ```
            // %0: arg
            // ...
            // %n: arg ; where `n` == aguard.exit_vars.len()
            // %n + 1: <copy_in[0]>
            // ...
            // %n + m: <copy_in[m]> ; where m == aguard.copy_in.len()
            // ```
            ginsts.clear();
            let mut gblock = Block {
                insts: std::mem::take(&mut ginsts),
                guard_extras: IndexVec::new(),
            };

            // Given an [InstIdx] `x` from the main `Block`, return its position in the guard's
            // `Block`. Because we now from above how many `arg`s and `copy_in` instructions there
            // are, this mapping is simple and because `exit_vars` and `copy_in` are ordered, can
            // be done with a binary search.
            let map = |x: InstIdx| {
                if let Ok(i) = aguard.exit_vars.binary_search(&x) {
                    InstIdx::from(i)
                } else if let Ok(i) = aguard.copy_in.binary_search(&x) {
                    InstIdx::from(aguard.exit_vars.len() + i)
                } else {
                    panic!()
                }
            };

            // Push the `arg` instructions.
            for iidx in aguard.exit_vars.iter() {
                let tyidx = aguard.block.inst(*iidx).tyidx(self.m);
                gblock.insts.push(Arg { tyidx }.into());
            }
            // Push the `copy_in` instructions, rewriting their operands as we go.
            for iidx in aguard.copy_in.iter() {
                let mut inst = aguard.block.inst(*iidx).clone();
                inst.rewrite_iidxs(&mut gblock, map);
                gblock.insts.push(inst);
            }

            // Finally, push the `term` instruction.
            gterms.clear();
            gterms.extend(gextra.deopt_vars.iter().map(|x| map(*x)));
            gblock.insts.push(Inst::Term(Term(gterms)));

            // At this point we have a complete, correct, `Block` representing the guard body. Now
            // we have to set this up in a way that both side-tracing and deopt are happy with.

            let mut stack_off = aguard.stack_off;
            let mut ra = RegAlloc::<AB>::new(self.m, &gblock, stack_off);
            ra.set_entry_stacks_at_end(&aguard.exit_vlocs);
            let mut deopt_frames = SmallVec::with_capacity(gextra.deopt_frames.len());
            let mut deopt_vars = Vec::with_capacity(gextra.deopt_vars.len());
            assert_eq!(gextra.deopt_vars.len(), gblock.term_vars().len());
            let mut deopt_term_iter = gextra.deopt_vars.iter().zip(gblock.term_vars().iter());
            for frame in gextra.deopt_frames.iter() {
                #[cfg(not(test))]
                let smap_lives_iter = aot_smaps
                    .get(usize::try_from(frame.pc_safepoint.id).unwrap())
                    .0
                    .live_vals
                    .iter();
                #[cfg(test)]
                let smap_lives_iter = self.m.smaps[usize::from(frame.smapidx)].iter();

                for smap_loc in smap_lives_iter {
                    let (deopt_iidx, term_iidx) = deopt_term_iter.next().unwrap();
                    // FIXME: This forces every variable represented in `gblock.term_vars` to be
                    // spilt before the end. This makes deopt simple, but is unnecessary for
                    // side-traces, where we could just pass things straight through in a register.
                    let fromvlocs = if let Inst::Const(Const { kind, .. }) = gblock.inst(*term_iidx)
                    {
                        varlocs![VarLoc::Const(kind.clone())]
                    } else {
                        let fromvlocs = aguard
                            .exit_vars
                            .binary_search(deopt_iidx)
                            .map(|x| &aguard.exit_vlocs[x])
                            .ok();
                        if let Some(fromvlocs) = fromvlocs
                            && let Some(VarLoc::Stack(x)) =
                                fromvlocs.iter().find(|x| matches!(x, VarLoc::Stack(_)))
                        {
                            varlocs![VarLoc::Stack(*x)]
                        } else if let Some(fromvlocs) = fromvlocs
                            && let Some(VarLoc::StackOff(x)) =
                                fromvlocs.iter().find(|x| matches!(x, VarLoc::StackOff(_)))
                        {
                            varlocs![VarLoc::StackOff(*x)]
                        } else {
                            match ra.get_stack_off(*term_iidx) {
                                Some(x) => varlocs![VarLoc::Stack(x)],
                                None => {
                                    let bitw = gblock.inst_bitw(self.m, *term_iidx);
                                    stack_off = self.be.align_spill(stack_off, bitw);
                                    ra.set_stack_off(*term_iidx, stack_off);
                                    varlocs![VarLoc::Stack(stack_off)]
                                }
                            }
                        }
                    };
                    #[cfg(not(test))]
                    let mut tovlocs = AB::smp_to_vloc(smap_loc, RegFill::Zeroed);
                    #[cfg(test)]
                    let mut tovlocs = smap_loc.clone();
                    if fromvlocs == tovlocs {
                        // Optimise away situations where we would just move a
                        // value from VLoc X to VLoc X.
                        tovlocs = VarLocs::new();
                    }
                    deopt_vars.push(DeoptVar {
                        bitw: gblock.inst_bitw(self.m, *term_iidx),
                        fromvlocs,
                        tovlocs,
                    });
                }
                deopt_frames.push(DeoptFrame {
                    pc: frame.pc.clone(),
                    pc_safepoint: frame.pc_safepoint,
                });
            }
            assert!(deopt_term_iter.next().is_none());
            assert!(!gblock.insts.is_empty());
            ra.keep_alive_at_term(InstIdx::from(gblock.insts.len() - 1), gblock.term_vars());
            stack_off = self.p_block(&gblock, None, ra, &aguard.exit_vlocs, logging)?;
            self.be.guard_completed(
                aguard.label.clone(),
                patch_label,
                stack_off - aguard.stack_off,
                gextra.bid,
                deopt_frames,
                deopt_vars,
                gextra.switch.clone(),
            );

            // Make sure we reuse the `ginsts` and `gterms` allocations.
            ginsts = std::mem::take(&mut gblock.insts);
            {
                let Inst::Term(Term(mut x)) = ginsts.pop().unwrap() else {
                    panic!()
                };
                gterms = std::mem::take(&mut x);
            }
        }

        Ok(())
    }

    /// Log information about the instruction `iidx`. `extra` is an additional string to add to the
    /// log after the "normal" instruction output has been logged.
    ///
    /// Note: it is the caller's duty to check that logging is enabled before calling this
    /// function.
    fn log_inst(&mut self, b: &Block, iidx: InstIdx, extra: &str) {
        let inst = b.inst(iidx);
        let ty = self.m.ty(inst.tyidx(self.m));
        if ty == &Ty::Void {
            self.be.log(inst.to_string(self.m, b));
        } else {
            self.be.log(format!(
                "%{}: {} = {}{extra}",
                usize::from(iidx),
                ty.to_string(self.m),
                inst.to_string(self.m, b)
            ));
        }
    }

    /// Generate a code for a [Block] `b` with the exception of its [Term] instruction which _must_
    /// have been handled prior to calling this function.
    ///
    /// For blocks that contain [Guard] instructions, `b_self` must be `Some` and must point to the
    /// same block as `b` (albeit with a stricter lifetime requirement). Blocks which do not
    /// contain guards can pass `None` to `b_self`.
    ///
    /// Returns the offset of the stack after code generation for this block has occurred.
    fn p_block(
        &mut self,
        b: &Block,
        b_self: Option<&'a Block>,
        mut ra: RegAlloc<AB>,
        entry_vlocs: &[VarLocs<AB::Reg>],
        logging: bool,
    ) -> Result<u32, CompilationError> {
        // These three variables are used to construct guard bodies: they're pulled out here
        // so that we only need to perform a single allocation per entry [Block]. See `Inst::Guard`
        // below to see how these are used.
        let mut gexit_vars = Vob::from_elem(false, b.insts_len());
        let mut gcopy = Vob::from_elem(false, b.insts_len());
        let mut gqueue = Vec::new();

        let mut insts_iter = b.insts_iter(..).rev().peekable();
        loop {
            if let Some((_iidx, Inst::Arg(_))) = insts_iter.peek() {
                break;
            }
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
                Inst::DebugStr(..) => {}
                Inst::DynPtrAdd(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_dynptradd(&mut ra, b, iidx, x)?;
                    }
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
                Inst::Floor(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_floor(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FMul(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fmul(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::FNeg(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_fneg(&mut ra, b, iidx, x)?;
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
                Inst::Guard(x @ Guard { geidx, .. }) => {
                    let gextra = b.gextra(*geidx);

                    // We now have to work out what will end up in the guard body. We have a simple
                    // fixed-point algorithm which examines instructions referenced in a guard's
                    // exit variables to see if they can be copied into the guard's body. This is
                    // done recursively until we hit instructions which can't (i.e. side effects)
                    // or shouldn't (e.g. in a register anyway) be copied.
                    //
                    // We make use of three variables (hoiked out of the loop for performance
                    // reasons) that we will use / calculate below:
                    // * `gexit_vars`: The new ordered, deduplicated, sequence of exit variables we
                    //   will associate the guard with.
                    // * `gcopy`: The ordered, deduplicated, sequence of instructions we've
                    //   determined should be copied into the guard body.
                    // * `gqueue`: The unordered queue of instructions we want to examine to see
                    //   how they affect the guard and guard body. Note: this may contain
                    //   duplicates and/or elements that have been processed before.
                    // The first two of these are `Vob`s, because they are naturally ordered and
                    // allow us to deduplicate for free.
                    gexit_vars.set_all(false);
                    gcopy.set_all(false);
                    assert!(gqueue.is_empty());
                    gqueue.extend(&gextra.deopt_vars);

                    while let Some(giidx) = gqueue.pop() {
                        let inst = b.inst(giidx);

                        if let Inst::Load(_) = inst {
                            // We can copy `Load`s in if there are no write effects between the
                            // `Load` and the current guard.
                            if ra.is_used(giidx)
                                || b.insts_iter(giidx + 1..iidx).any(|(_, inst)| {
                                    inst.write_effects()
                                        .interferes(Effects::all().minus_guard())
                                })
                            {
                                gexit_vars.set(usize::from(giidx), true);
                                continue;
                            }
                        } else if inst.read_write_effects().interferes(Effects::all())
                            || (ra.is_used(giidx)
                                && !matches!(inst, Inst::Const(_))
                                && ra.iter_reg_for(giidx).nth(0).is_some())
                        {
                            // We don't copy instructions that are used by non-guard instructions
                            // unless: they're a `Const`; aren't in a register; don't have
                            // side-effects.
                            gexit_vars.set(usize::from(giidx), true);
                            continue;
                        }

                        // We can copy this instruction!
                        gcopy.set(usize::from(giidx), true);
                        // Add all this instruction's operand references to the queue.
                        for op_iidx in inst.iter_iidxs(b) {
                            gqueue.push(op_iidx);
                        }
                    }

                    // At this point, `gexit_vars` is a new sequence of exit variables for this
                    // guard.
                    let exit_vars = gexit_vars
                        .iter_set_bits(..)
                        .map(InstIdx::from)
                        .collect::<Vec<_>>();
                    let label = self.be.i_guard(&mut ra, b, iidx, x, &exit_vars)?;
                    let exit_vlocs = ra.vlocs_from_iidxs(&exit_vars);
                    self.asmguards.push(AsmGuard {
                        geidx: *geidx,
                        block: b_self.unwrap(),
                        label,
                        exit_vars,
                        exit_vlocs,
                        copy_in: gcopy
                            .iter_set_bits(..)
                            .map(InstIdx::from)
                            .collect::<Vec<_>>(),
                        stack_off: ra.stack_off(),
                    });
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
                Inst::MemCpy(x) => {
                    self.be.i_memcpy(&mut ra, b, iidx, x)?;
                }
                Inst::MemSet(x) => {
                    self.be.i_memset(&mut ra, b, iidx, x)?;
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
                Inst::SDiv(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_sdiv(&mut ra, b, iidx, x)?;
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
                Inst::SMax(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_smax(&mut ra, b, iidx, x)?;
                    }
                }
                Inst::SMin(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_smin(&mut ra, b, iidx, x)?;
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
                Inst::Term(Term(_)) => (),
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
                Inst::UIToFP(x) => {
                    if ra.is_used(iidx) {
                        self.be.i_uitofp(&mut ra, b, iidx, x)?;
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
                self.log_inst(b, iidx, "");
            }
        }

        // We deal with [Arg] instructions specially: they're best thought of as
        // pseudo-instructions in the sense that they define variables but don't directly
        // generate code themselves.
        ra.set_entry_vlocs_at_start(&mut self.be, entry_vlocs);
        if logging {
            for (iidx, _inst) in insts_iter {
                let pp_vlocs = entry_vlocs[usize::from(iidx)]
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                self.log_inst(b, iidx, &format!(" [{pp_vlocs}]"));
            }
        }

        Ok(ra.stack_off())
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
#[test_stubs]
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

    /// Convert a yksmp stackmap to one or more `VarLoc`s. Since stack maps do not currently encode
    /// a register fill, any registers returned by this function will be given the fill `reg_fill`.
    fn smp_to_vloc(
        smp_locs: &SmallVec<[yksmp::Location; 1]>,
        reg_fill: RegFill,
    ) -> VarLocs<Self::Reg>;
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
            ExeCodeBuf,
            IndexVec<CompiledGuardIdx, J2CompiledGuard<Self::Reg>>,
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

    /// Move a value of `bitw` on the stack from `src_stack_off` to `dst_stack_off` using `tmp_reg`.
    fn move_stack_val(
        &mut self,
        bitw: u32,
        src_stack_off: u32,
        dst_stack_off: u32,
        tmp_reg: Self::Reg,
    );

    // Functions for the start and end of various kinds of traces. Note that some functions are
    // used by more than one (x, y) trace kind.

    /// Produce code for the completed body of a (ControlPoint, Coupler | Return) trace. It has
    /// consumed `stack_off` additional bytes of stack space. The label returned must be attached
    /// to the first instruction after the stack is adjusted.
    fn controlpoint_coupler_or_return_start(
        &mut self,
        stack_off: u32,
    ) -> Result<Self::Label, CompilationError>;

    /// Produce code for the backwards jump at the end of a (ControlPoint, Loop) trace.
    fn controlpoint_loop_end(&mut self) -> Result<Self::Label, CompilationError>;

    /// Produce code for the completed body of a (ControlPoint, Loop) trace. It has consumed
    /// `stack_off` additional bytes of stack space. `post_stack_label` must be attached to the
    /// first instruction after the stack is adjusted.
    fn controlpoint_loop_start(&mut self, post_stack_label: Self::Label, stack_off: u32);

    /// Produce code for the jump to `tgt_ctr` at the end of a (*, Coupler) trace.
    fn star_coupler_end(
        &mut self,
        tgt_ctr: &Arc<J2CompiledTrace<Self::Reg>>,
    ) -> Result<(), CompilationError>;

    /// Produce code for the completed body of a (Guard, Coupler) trace. It has consumed
    /// `stack_off` additional bytes of stack space.
    fn guard_coupler_start(&mut self, stack_off: u32);

    /// Produce code for the end of a (*, Return) trace. The safepoint of the `return` is
    /// `exit_safepoint`.
    fn star_return_end(
        &mut self,
        exit_safepoint: &'static DeoptSafepoint,
    ) -> Result<(), CompilationError>;

    // Functions for guards.

    /// Produce code for the end of a guard: return a label for the instruction to be patched when
    /// a side-trace is produced.
    fn guard_end(
        &mut self,
        trid: TraceId,
        gidx: CompiledGuardIdx,
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
        deopt_frames: SmallVec<[DeoptFrame; 2]>,
        deopt_vars: Vec<DeoptVar<Self::Reg>>,
        switch: Option<Switch>,
    );

    // Functions for each HIR instruction.

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

    fn i_floor(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Floor,
    ) -> Result<(), CompilationError>;

    fn i_fmul(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FMul,
    ) -> Result<(), CompilationError>;

    fn i_fneg(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &FNeg,
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

    /// The instruction should use [super::regalloc::RegCnstr::KeepAlive] for the values in
    /// `exit_vars`.
    ///
    /// The label returned should be the jump instruction to the guard body.
    fn i_guard(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &Guard,
        exit_vars: &[InstIdx],
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

    fn i_memcpy(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &MemCpy,
    ) -> Result<(), CompilationError>;

    fn i_memset(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &MemSet,
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

    fn i_sdiv(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &SDiv,
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

    fn i_smax(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &SMax,
    ) -> Result<(), CompilationError>;

    fn i_smin(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &SMin,
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

    fn i_uitofp(
        &mut self,
        ra: &mut RegAlloc<Self>,
        b: &Block,
        iidx: InstIdx,
        inst: &UIToFP,
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
struct AsmGuard<'a, AB: HirToAsmBackend + ?Sized> {
    geidx: GuardExtraIdx,
    label: AB::Label,
    /// The block that contained the associated guard.
    block: &'a Block,
    /// The post-guard-body list of variables live at the point of the guard's exit. Must be sorted
    /// and not contain duplicates.
    exit_vars: Vec<InstIdx>,
    /// For each variable in [Self::exit_Vars], its corresponding [VarLocs].
    exit_vlocs: Vec<VarLocs<AB::Reg>>,
    /// The instructions that should be copied into the guard body when it is created. Must be
    /// sorted, not contain duplicates, and have an empty intersection with [Self::exit_vars].
    copy_in: Vec<InstIdx>,
    /// The stack offset of the register allocator at the entry point of the guard.
    stack_off: u32,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        compile::{
            j2::{
                compiled_trace::{CompiledGuardIdx, DeoptFrame, DeoptVar},
                hir::Mod,
                hir_parser::str_to_mod,
                regalloc::{RegCnstr, RegCnstrFill, TestRegIter},
            },
            jitc_yk::aot_ir,
        },
        location::{HotLocation, HotLocationKind},
        mt::TraceId,
    };
    use fm::{FMBuilder, FMatcher};

    use lazy_static::lazy_static;
    use parking_lot::Mutex;
    use regex::Regex;
    use std::sync::Arc;
    use strum::{Display, EnumCount, FromRepr};

    #[derive(Copy, Clone, Debug, Display, EnumCount, FromRepr, PartialEq)]
    #[repr(u8)]
    enum TestReg {
        R0,
        R1,
        R2,
        R3,
        Undefined,
    }

    const GP_REGS: [TestReg; 4] = [TestReg::R0, TestReg::R1, TestReg::R2, TestReg::R3];

    impl RegT for TestReg {
        type RegIdx = TestRegIdx;
        const MAX_REGIDX: TestRegIdx = TestRegIdx::from_usize_unchecked(TestReg::COUNT);

        fn undefined() -> Self {
            TestReg::Undefined
        }

        fn from_regidx(idx: Self::RegIdx) -> Self {
            TestReg::from_repr(idx.raw()).unwrap()
        }

        fn regidx(&self) -> Self::RegIdx {
            TestRegIdx::from(*self as usize)
        }

        fn is_caller_saved(&self) -> bool {
            todo!()
        }

        fn iter_test_regs() -> impl TestRegIter<Self> {
            TestRegTestIter::new()
        }

        fn from_str(s: &str) -> Option<Self> {
            match s {
                "R0" => Some(Self::R0),
                "R1" => Some(Self::R1),
                "R2" => Some(Self::R2),
                "R3" => Some(Self::R3),
                _ => None,
            }
        }
    }

    index_vec::define_index_type! {
        pub(crate) struct TestRegIdx = u8;
    }

    struct TestRegTestIter<Reg> {
        gp_regs: Box<dyn Iterator<Item = Reg>>,
    }

    impl TestRegTestIter<TestReg> {
        fn new() -> Self {
            Self {
                gp_regs: Box::new(
                    [TestReg::R0, TestReg::R1, TestReg::R2, TestReg::R3]
                        .iter()
                        .cloned(),
                ),
            }
        }
    }

    impl TestRegIter<TestReg> for TestRegTestIter<TestReg> {
        fn next_reg(&mut self, ty: &Ty) -> Option<TestReg> {
            match ty {
                Ty::Double | Ty::Float => todo!(),
                Ty::Func(_func_ty) => todo!(),
                Ty::Int(bitw) => {
                    if *bitw <= 64 {
                        self.gp_regs.next()
                    } else {
                        todo!()
                    }
                }
                Ty::Ptr(addrspace) => {
                    assert_eq!(*addrspace, 0);
                    self.gp_regs.next()
                }
                Ty::Void => todo!(),
            }
        }
    }

    index_vec::define_index_type! {
        struct TestLabelIdx = u32;
        IMPL_RAW_CONVERSIONS = true;
    }

    struct TestHirToAsm<'a> {
        m: &'a Mod<TestReg>,
        log: Vec<String>,
    }

    impl<'a> TestHirToAsm<'a> {
        fn new(m: &'a Mod<TestReg>) -> Self {
            Self { m, log: Vec::new() }
        }
    }

    impl<'a> HirToAsmBackend for TestHirToAsm<'a> {
        type Label = TestLabelIdx;
        type Reg = TestReg;
        type BuildTest = String;

        fn build_test(self, _labels: &[Self::Label]) -> Self::BuildTest {
            self.log.join("\n")
        }

        fn iter_possible_regs(&self, b: &Block, iidx: InstIdx) -> impl Iterator<Item = Self::Reg> {
            match b.inst_ty(self.m, iidx) {
                Ty::Double | Ty::Float => todo!(),
                Ty::Func(_func_ty) => todo!(),
                Ty::Int(_) | Ty::Ptr(_) => GP_REGS.iter().cloned(),
                Ty::Void => todo!(),
            }
        }

        fn log(&mut self, s: String) {
            self.log.push(format!("; {s}"));
        }

        fn const_needs_tmp_reg(
            &self,
            _reg: Self::Reg,
            _c: &ConstKind,
        ) -> Option<impl Iterator<Item = Self::Reg>> {
            None::<std::iter::Empty<Self::Reg>>
        }

        fn arrange_fill(
            &mut self,
            reg: Self::Reg,
            src_fill: RegFill,
            dst_bitw: u32,
            dst_fill: RegFill,
        ) {
            self.log.push(format!(
                "arrange_fill: reg={reg:?}, from={src_fill:?}, dst_bitw={dst_bitw}, to={dst_fill:?}"
            ));
        }

        fn copy_reg(
            &mut self,
            from_reg: Self::Reg,
            to_reg: Self::Reg,
        ) -> Result<(), CompilationError> {
            self.log.push(format!("copy_reg: {to_reg:?}={from_reg:?}"));
            Ok(())
        }

        fn align_spill(&self, stack_off: u32, bitw: u32) -> u32 {
            stack_off + (bitw / 8).next_multiple_of(8)
        }

        fn spill(
            &mut self,
            reg: Self::Reg,
            in_fill: RegFill,
            stack_off: u32,
            bitw: u32,
        ) -> Result<(), CompilationError> {
            self.log.push(format!(
                "spill: reg={reg:?}, in_fill={in_fill:?}, stack_off={stack_off}, bitw={bitw}"
            ));
            Ok(())
        }

        fn guard_coupler_start(&mut self, stack_off: u32) {
            self.log
                .push(format!("guard_coupler_start: stack_off={stack_off}"));
        }

        fn guard_end(
            &mut self,
            _trid: TraceId,
            _gidx: CompiledGuardIdx,
        ) -> Result<Self::Label, CompilationError> {
            Ok(TestLabelIdx::new(0))
        }

        fn guard_completed(
            &mut self,
            _start_label: Self::Label,
            _patch_label: Self::Label,
            _stack_off: u32,
            _bid: aot_ir::BBlockId,
            _deopt_frames: SmallVec<[DeoptFrame; 2]>,
            deopt_vars: Vec<DeoptVar<Self::Reg>>,
            _switch: Option<Switch>,
        ) {
            self.log.push(format!(
                "guard_completed:\n{}",
                deopt_vars
                    .iter()
                    .map(|x| format!(
                        "  fromvlocs=[{}]\n  tovlocs=[{}]",
                        x.fromvlocs
                            .iter()
                            .map(|x| format!("{x:?}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                        x.tovlocs
                            .iter()
                            .map(|x| format!("{x:?}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ));
        }

        fn i_dynptradd(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            DynPtrAdd {
                ptr,
                num_elems,
                elem_size,
            }: &DynPtrAdd,
        ) -> Result<(), CompilationError> {
            let [ptrr, nelemsr, outr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *ptr,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Input {
                        in_iidx: *num_elems,
                        in_fill: RegCnstrFill::Zeroed,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Output {
                        out_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        can_be_same_as_input: true,
                    },
                ],
            )?;
            self.log.push(format!(
                "dynptradd: {outr:?}={ptrr:?} + ({nelemsr:?}*{elem_size})"
            ));
            Ok(())
        }

        fn i_guard(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            Guard { cond, .. }: &Guard,
            exit_vars: &[InstIdx],
        ) -> Result<Self::Label, CompilationError> {
            let [_condr, _] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *cond,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::KeepAlive { iidxs: exit_vars },
                ],
            )?;
            self.log.push(format!(
                "i_guard: [{}]",
                exit_vars
                    .iter()
                    .map(|x| format!("%{x:?}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
            Ok(TestLabelIdx::new(0))
        }

        fn i_load(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            Load { ptr, .. }: &Load,
        ) -> Result<(), CompilationError> {
            let [ptrr, outr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *ptr,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Output {
                        out_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        can_be_same_as_input: true,
                    },
                ],
            )?;
            self.log.push(format!("load: {outr:?}=*{ptrr:?}"));
            Ok(())
        }

        fn i_ptradd(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            PtrAdd { ptr, off, .. }: &PtrAdd,
        ) -> Result<(), CompilationError> {
            let [inr, outr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *ptr,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Output {
                        out_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        can_be_same_as_input: true,
                    },
                ],
            )?;
            self.log.push(format!("ptradd: {outr:?}={inr:?} + {off}"));
            Ok(())
        }

        fn i_store(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            Store { ptr, val, .. }: &Store,
        ) -> Result<(), CompilationError> {
            let [ptrr, valr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *ptr,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Input {
                        in_iidx: *val,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                ],
            )?;
            self.log.push(format!("store: *{ptrr:?}={valr:?}"));
            Ok(())
        }
    }

    lazy_static! {
        /// Use `{{name}}` to match non-literal strings in tests.
        static ref PTN_RE: Regex = {
            Regex::new(r"\{\{.+?\}\}").unwrap()
        };

        static ref PTN_RE_IGNORE: Regex = {
            Regex::new(r"\{\{_}\}").unwrap()
        };

        static ref TEXT_RE: Regex = {
            Regex::new(r"[a-zA-Z0-9\._]+").unwrap()
        };
    }

    fn fmatcher(ptn: &str) -> FMatcher<'_> {
        FMBuilder::new(ptn)
            .unwrap()
            .name_matcher(PTN_RE.clone(), TEXT_RE.clone())
            .name_matcher_ignore(PTN_RE_IGNORE.clone(), TEXT_RE.clone())
            .build()
            .unwrap()
    }

    /// Enable simple tests of hir_to_asm.
    ///
    /// This function takes a module `s` in and runs [HirToAsm] it with our "test" backend above.
    /// It then runs each line of the log through `log_filter`, only keeping lines where
    /// `log_filter` returns `true`. It recombines the log and then matches it against the [fm]
    /// pattern `ptn`.
    fn build_and_test<F>(s: &str, log_filter: F, ptns: &[&str])
    where
        F: Fn(&str) -> bool,
    {
        let m = str_to_mod::<TestReg>(s);

        let hl = Arc::new(Mutex::new(HotLocation {
            kind: HotLocationKind::Tracing(TraceId::testing()),
            tracecompilation_errors: 0,
            #[cfg(feature = "ykd")]
            debug_str: None,
        }));

        let be = TestHirToAsm::new(&m);
        let log = HirToAsm::new(&m, hl, be).build_test().unwrap();
        let log = log
            .lines()
            .filter(|s| log_filter(s))
            .collect::<Vec<_>>()
            .join("\n");
        let mut failures = Vec::with_capacity(ptns.len());
        for ptn in ptns {
            match fmatcher(ptn).matches(&log) {
                Ok(_) => return,
                Err(e) => failures.push(format!("{e}")),
            }
        }

        panic!("{}", failures.join("\n\n"));
    }

    #[test]
    fn gbody_moves() {
        // Nothing can be moved into the guard body.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          guard true, %0, [%1], [[[reg("R0", undefined)]]]
          term [%0, %1]
        "#,
            |_| true,
            &[r#"
          ...
          i_guard: [%1]
          ...
        "#],
        );

        // Move one instruction into the guard body.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: ptr = ptradd %1, 8
          guard true, %0, [%2], [[[reg("R0", undefined)]]]
          term [%0, %1]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1]
          i_guard: [%1]
          ...
          ; %0: i1 = arg [Reg("R0", Undefined)]
          guard_coupler_start: stack_off=0
          ; term [%1]
          spill: reg=R0, in_fill=Undefined, stack_off=8, bitw=64
          ptradd: R0=R0 + 8
          ; %1: ptr = ptradd %0, 8
          arrange_fill: reg=R0, from=Undefined, dst_bitw=64, to=Undefined
          copy_reg: R0=R1
          ; %0: ptr = arg [Reg("R1", Undefined)]
          guard_completed:
            fromvlocs=[Stack(8)]
            tovlocs=[Reg(R0, Undefined)]
        "#],
        );

        // Move two unrelated instructions into the guard body.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: ptr = ptradd %1, 8
          %3: ptr = ptradd %1, 16
          guard true, %0, [%2, %3], [[[reg("R0", undefined)]], [[reg("R1", undefined)]]]
          term [%0, %1]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1]
          i_guard: [%1]
          ...
          ; term [%1, %2]
          spill: reg=R0, in_fill=Undefined, stack_off=16, bitw=64
          ptradd: R0=R0 + 16
          ; %2: ptr = ptradd %0, 16
          spill: reg=R1, in_fill=Undefined, stack_off=8, bitw=64
          ptradd: R1=R0 + 8
          ; %1: ptr = ptradd %0, 8
          arrange_fill: reg=R0, from=Undefined, dst_bitw=64, to=Undefined
          copy_reg: R0=R1
          ; %0: ptr = arg [Reg("R1", Undefined)]
          guard_completed:
            fromvlocs=[Stack(8)]
            tovlocs=[Reg(R0, Undefined)]
            fromvlocs=[Stack(16)]
            tovlocs=[Reg(R1, Undefined)]
        "#],
        );

        // Move two recursively-related instructions into the guard body.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: i64 = arg [reg]
          %3: ptr = ptradd %1, 8
          %4: ptr = dynptradd %3, %2, 8
          guard true, %0, [%4], [[[reg("R0", undefined)]]]
          term [%0, %1, %2]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1, %2]
          i_guard: [%1, %2]
          ...
          ; term [%3]
          spill: reg=R0, in_fill=Undefined, stack_off=8, bitw=64
          dynptradd: R0=R0 + (R1*8)
          ; %3: ptr = dynptradd %2, %1, 8
          ptradd: R0=R0 + 8
          ; %2: ptr = ptradd %0, 8
          arrange_fill: reg=R1, from=Undefined, dst_bitw=64, to=Zeroed
          copy_reg: R1=R2
          arrange_fill: reg=R0, from=Undefined, dst_bitw=64, to=Undefined
          copy_reg: R0=R1
          ; %1: i64 = arg [Reg("R2", Undefined)]
          ; %0: ptr = arg [Reg("R1", Undefined)]
          guard_completed:
            fromvlocs=[Stack(8)]
            tovlocs=[Reg(R0, Undefined)]
        "#],
        );

        // Move something that we don't use at the point of the guard but then use later. This
        // doesn't hurt, but does mean we compute things twice.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: ptr = ptradd %1, 8
          blackbox %2
          guard true, %0, [%2], [[[reg("R0", undefined)]]]
          term [%0, %1]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1]
          i_guard: [%1]
          ; guard true, %0, [%2]
          ; blackbox %2
          ptradd: R2=R1 + 8
          ; %2: ptr = ptradd %1, 8
          ...
          ; term [%1]
          ...
          ptradd: R0=R0 + 8
          ; %1: ptr = ptradd %0, 8
          ...
        "#],
        );

        // Move a (safe to move!) `load` into a guard body.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: i8 = load %1
          guard true, %0, [%2], [[[reg("R0", undefined)]]]
          term [%0, %1]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1]
          i_guard: [%1]
          ; guard true, %0, [%2]
          ; %2: i8 = load %1
          ...
          ; term [%1]
          ...
          load: R0=*R0
          ; %1: i8 = load %0
          ...
        "#],
        );

        // Don't move a (not safe to move!) `load` into a guard body.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: i8 = arg [reg]
          %3: i8 = load %1
          store %2, %1
          guard true, %0, [%3], [[[reg("R0", undefined)]]]
          term [%0, %1, %2]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1, %2]
          i_guard: [%3]
          ; guard true, %0, [%3]
          store: *R1=R2
          ; store %2, %1
          load: R3=*R1
          ; %3: i8 = load %1
          ...
        "#],
        );
    }

    #[test]
    fn gbody_deopt_vars() {
        // Test `VarLoc::Stack`.
        build_and_test(
            r#"
          %0: i1 = arg [reg]
          %1: ptr = arg [reg]
          %2: ptr = ptradd %1, 8
          guard true, %0, [%2], [[[stack(24)]]]
          term [%0, %1]
        "#,
            |_| true,
            &[r#"
          ; term [%0, %1]
          i_guard: [%1]
          ...
          ; %0: i1 = arg [Reg("R0", Undefined)]
          guard_coupler_start: stack_off=0
          ; term [%1]
          spill: reg=R0, in_fill=Undefined, stack_off=8, bitw=64
          ptradd: R0=R0 + 8
          ; %1: ptr = ptradd %0, 8
          arrange_fill: reg=R0, from=Undefined, dst_bitw=64, to=Undefined
          copy_reg: R0=R1
          ; %0: ptr = arg [Reg("R1", Undefined)]
          guard_completed:
            fromvlocs=[Stack(8)]
            tovlocs=[Stack(24)]
        "#],
        );
    }
}
