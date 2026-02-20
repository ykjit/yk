//! AOT IR to HIR translator.
//!
//! This is a relatively simple translation from AOT IR to [super::hir], assuming that the input
//! trace derives from (or is otherwise compatible with) [crate::trace::swt].
//!
//! Translation is intertwined with optimisation so that `idempotent_promote` works: we need to be
//! able to tell what's constant to decide whether to promote at the point of an
//! `idempotent_promote` or inline. That means that this module doesn't directly record
//! instructions itself: they are immediately pushed to an optimiser. When we need to look up a HIR
//! instruction, we thus need to look it up in the optimiser, since that is the source of truth for
//! the output of the translation.
//!
//! The intertwining of optimisation has an important consequence: this module cannot assume that
//! there is a unique relationship between an AOT instruction / variable and its HIR equivalent. A
//! pushed instruction might be proven to be equivalent to a previous instruction; guards might be
//! optimised away entirely; and so on.

use crate::{
    compile::{
        CompilationError, CompiledTrace, GuardId,
        j2::{
            J2,
            compiled_trace::{
                CompiledGuardIdx, DeoptFrame, DeoptVar, J2CompiledTrace, J2TraceStart,
            },
            hir,
            opt::{OptT, fullopt::FullOpt, noopt::NoOpt},
            regalloc::{RegT, VarLoc, VarLocs},
        },
        jitc_yk::{AOT_MOD, aot_ir::*, arbbitint::ArbBitInt},
    },
    location::HotLocation,
    log::{IRPhase, log_ir, should_log_any_ir, should_log_ir, stats::TimingState},
    mt::{MT, TraceId},
    trace::TraceAction,
};
#[cfg(test)]
use index_vec::IndexVec;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use std::{assert_matches, collections::HashMap, iter::Peekable, marker::PhantomData, sync::Arc};

/// The symbol name of the global variable pointers array.
const GLOBAL_PTR_ARRAY_SYM: &str = "__yk_globalvar_ptrs";

pub(super) struct AotToHir<Reg: RegT> {
    mt: Arc<MT>,
    j2: Arc<J2>,
    /// The AOT IR.
    am: &'static Module,
    hl: Arc<Mutex<HotLocation>>,
    ta_iter: Peekable<TraceActionIterator>,
    /// What was the previous [BBlockId] fully processed by [TraceActionIterator]? Note: this is a
    /// bit more subtle than "the value before the most recent `next`". It really means "the last
    /// value before `p_block` or equivalent fully ran". As that suggests, this is rather fragile:
    /// there must be, and we should aim to find soon, a better way to do this.
    prev_bid: Option<BBlockId>,
    trid: TraceId,
    bkind: BuildKind,
    /// The data passed to successive calls to `yk_promote`. Note: some of this may have been
    /// passed in outlined code and must be ignored in such parts of a trace.
    promotions: Box<[u8]>,
    /// How much of [Self::promotions] have we consumed so far?
    promotions_off: usize,
    /// The virtual address of the global variable pointer array.
    ///
    /// This is an array added to the LLVM AOT module by ykllvm containing a pointer to each global
    /// variable in the AOT module. The indices of the elements correspond with
    /// [aot_ir::GlobalDeclIdx]s. Note: this array is not available during testing, since tests are
    /// not built with ykllvm.
    globals: &'static [*const ()],
    opt: Box<dyn OptT>,
    /// Initially set to `None` until we find the locations for this trace's arguments.
    frames: Vec<Frame>,
    /// If logging is enabled, create a map of addresses -> names to make IR printing nicer.
    addr_name_map: Option<HashMap<usize, Option<String>>>,
    /// The JIT IR this struct builds.
    phantom: PhantomData<Reg>,
    /// The strings used by yk_debug_str
    debug_strs: Vec<String>,
    /// The next debug string to process, as an index into [Self::debug_strs].
    debug_strs_seen: usize,
}

impl<Reg: RegT + 'static> AotToHir<Reg> {
    pub(super) fn new(
        mt: &Arc<MT>,
        j2: &Arc<J2>,
        am: &'static Module,
        hl: Arc<Mutex<HotLocation>>,
        ta_iter: Box<dyn crate::trace::AOTTraceIterator>,
        trid: TraceId,
        bkind: BuildKind,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
    ) -> Self {
        let globals = {
            let ptr = j2.dlsym(GLOBAL_PTR_ARRAY_SYM, false).unwrap().0 as *const *const ();
            assert!(!ptr.is_null());
            unsafe { std::slice::from_raw_parts(ptr, am.global_decls_len()) }
        };

        let opt = if mt.opt_level() == 0 {
            Box::new(NoOpt::new()) as Box<dyn OptT>
        } else {
            Box::new(FullOpt::new())
        };

        Self {
            mt: Arc::clone(mt),
            j2: Arc::clone(j2),
            am,
            hl,
            ta_iter: TraceActionIterator::new(ta_iter).peekable(),
            prev_bid: None,
            trid,
            bkind,
            promotions,
            promotions_off: 0,
            globals,
            opt,
            frames: Vec::new(),
            addr_name_map: should_log_any_ir().then_some(HashMap::new()),
            phantom: PhantomData,
            debug_strs,
            debug_strs_seen: 0,
        }
    }

    pub(super) fn build(mut self) -> Result<hir::Mod<Reg>, CompilationError> {
        if should_log_ir(IRPhase::AOT) {
            log_ir(&format!(
                "--- Begin aot ---\n{}\n--- End aot ---\n",
                &*AOT_MOD
            ));
        }

        self.mt.stats.timing_state(TimingState::Compiling);

        // Process the start of the trace.
        let bmk = match &self.bkind {
            BuildKind::Coupler { .. } | BuildKind::Loop => {
                // The control point call will be found in the immediate predecessor of first block
                // we see. That means we first see a `MappedAOTBlock` with index `bb`...
                let Some(Ok(TraceAction::MappedAOTBBlock { funcidx, bbidx })) = self.ta_iter.peek()
                else {
                    panic!()
                };
                assert!(*bbidx > 0);
                // ...and the block we really want thus has index `bb - 1`.
                let ta = &TraceAction::MappedAOTBBlock {
                    funcidx: *funcidx,
                    bbidx: *bbidx - 1,
                };
                let cp_bid = self.ta_to_bid(ta).unwrap();
                let entry_safepoint = self.p_start_loop(&cp_bid)?;
                assert_matches!(
                    self.ta_iter.peek(),
                    Some(&Ok(TraceAction::MappedAOTBBlock { .. }))
                );
                match &self.bkind {
                    BuildKind::Coupler { tgt_ctr } => {
                        let tgt_ctr = Arc::clone(tgt_ctr)
                            .as_any()
                            .downcast::<J2CompiledTrace<Reg>>()
                            .unwrap();
                        BuildModKind::Coupler {
                            entry_safepoint,
                            tgt_ctr,
                        }
                    }
                    BuildKind::Loop => BuildModKind::Loop { entry_safepoint },
                    _ => unreachable!(),
                }
            }
            BuildKind::Side {
                src_ctr,
                src_gid,
                tgt_ctr,
            } => {
                let src_ctr = Arc::clone(src_ctr)
                    .as_any()
                    .downcast::<J2CompiledTrace<Reg>>()
                    .unwrap();
                let src_gidx = CompiledGuardIdx::from(usize::from(*src_gid));
                let prev_bid = src_ctr.bid(src_gidx);
                self.prev_bid = Some(prev_bid);
                let tgt_ctr = Arc::clone(tgt_ctr)
                    .as_any()
                    .downcast::<J2CompiledTrace<Reg>>()
                    .unwrap();
                let args_vlocs = self.p_start_side(&src_ctr, src_gidx, &tgt_ctr)?;
                if let Some(hir::Switch {
                    iid,
                    seen_bbidxs: seen_blocks,
                }) = src_ctr.switch(src_gidx)
                {
                    // In the past, we saw cases where parts of switch blocks were retraced. I
                    // suspect that was an hwt artefact, but I'm not 100% sure, so guard against it
                    // happening here until we understand if it's still possible or not.
                    let next_ta = self.ta_iter.peek().unwrap().as_ref().unwrap().clone();
                    assert_ne!(self.ta_to_bid(&next_ta), Some(prev_bid));

                    self.p_switch(iid.clone(), prev_bid, self.am.inst(iid), Some(seen_blocks))?;
                }
                BuildModKind::Side {
                    prev_bid,
                    args_vlocs,
                    src_ctr,
                    src_gidx,
                    tgt_ctr,
                }
            }
        };

        self.prev_bid = match &bmk {
            BuildModKind::Coupler { .. } | BuildModKind::Loop { .. } => None,
            BuildModKind::Side { prev_bid, .. } => Some(*prev_bid),
        };

        // If we encounter a return in a side-trace, [return_safepoint] will be `Some`, and no
        // further processing of the trace should occur.
        let return_safepoint = self.p_blocks()?;
        if return_safepoint.is_none() {
            assert_eq!(self.promotions_off, self.promotions.len());
            assert_eq!(self.frames.len(), 1);
            let exit_safepoint = match &bmk {
                BuildModKind::Loop { entry_safepoint } => entry_safepoint,
                BuildModKind::Coupler { tgt_ctr, .. } | BuildModKind::Side { tgt_ctr, .. } => {
                    match &tgt_ctr.trace_start {
                        J2TraceStart::ControlPoint {
                            entry_safepoint, ..
                        } => entry_safepoint,
                        J2TraceStart::Guard { .. } => todo!(),
                    }
                }
            };
            let term_vars = exit_safepoint
                .lives
                .iter()
                .map(|x| self.frames[0].get_local(&*self.opt, &x.to_inst_id()))
                .collect::<Vec<_>>();
            self.opt.feed_void(hir::Inst::Term(hir::Term(term_vars)))?;
        }

        let tyidx_int1 = self.opt.tyidx_int1();
        let tyidx_ptr0 = self.opt.tyidx_ptr0();
        let tyidx_void = self.opt.tyidx_void();

        let (trace_start, trace_end, tys) = match bmk {
            BuildModKind::Coupler {
                entry_safepoint,
                tgt_ctr,
            } => {
                let (entry, tys) = self.opt.build()?;
                (
                    hir::TraceStart::ControlPoint { entry_safepoint },
                    hir::TraceEnd::Coupler { entry, tgt_ctr },
                    tys,
                )
            }
            BuildModKind::Loop { entry_safepoint } => match return_safepoint {
                None => {
                    let (entry, peel, tys) = self.opt.build_with_peel()?;
                    (
                        hir::TraceStart::ControlPoint { entry_safepoint },
                        hir::TraceEnd::Loop { entry, peel },
                        tys,
                    )
                }
                Some(exit_safepoint) => {
                    let (entry, tys) = self.opt.build()?;
                    (
                        hir::TraceStart::ControlPoint { entry_safepoint },
                        hir::TraceEnd::Return {
                            entry,
                            exit_safepoint,
                        },
                        tys,
                    )
                }
            },
            BuildModKind::Side {
                args_vlocs,
                src_ctr,
                src_gidx,
                tgt_ctr,
                ..
            } => {
                let (entry, tys) = self.opt.build()?;
                match return_safepoint {
                    None => (
                        hir::TraceStart::Guard {
                            args_vlocs,
                            src_ctr,
                            src_gidx,
                        },
                        hir::TraceEnd::Coupler { entry, tgt_ctr },
                        tys,
                    ),
                    Some(exit_safepoint) => (
                        hir::TraceStart::Guard {
                            args_vlocs,
                            src_ctr,
                            src_gidx,
                        },
                        hir::TraceEnd::Return {
                            entry,
                            exit_safepoint,
                        },
                        tys,
                    ),
                }
            }
        };

        let m = hir::Mod {
            trid: self.trid,
            trace_start,
            trace_end,
            tys,
            tyidx_int1,
            tyidx_ptr0,
            tyidx_void,
            addr_name_map: self.addr_name_map,
            #[cfg(test)]
            smaps: IndexVec::new(),
        };

        let ds = if let Some(x) = &self.hl.lock().debug_str {
            format!(": {}", x.as_str())
        } else {
            "".to_owned()
        };

        if should_log_ir(IRPhase::DebugStrs) {
            log_ir(&format!(
                "--- Begin debugstrs{ds} ---\n; {}\n{}\n--- End debugstrs ---\n",
                m.json_info().split("\n").collect::<Vec<_>>().join("\n; "),
                self.debug_strs.join("\n")
            ));
        }

        if should_log_ir(IRPhase::PreOpt) {
            log_ir(&format!(
                "--- Begin jit-pre-opt{ds} ---\n{}\n--- End jit-pre-opt ---\n",
                m
            ));
        }

        #[cfg(test)]
        m.assert_well_formed();

        Ok(m)
    }

    fn peek_next_bbid(&mut self) -> Option<BBlockId> {
        self.ta_iter
            .peek()
            .and_then(|x| x.as_ref().ok())
            .cloned()
            .and_then(|x| self.ta_to_bid(&x))
    }

    fn next_pc(&self, pc: InstId) -> InstId {
        InstId::new(
            pc.funcidx(),
            pc.bbidx(),
            BBlockInstIdx::new(usize::from(pc.iidx()) + 1),
        )
    }

    fn push_guard(
        &mut self,
        bid: BBlockId,
        iid: InstId,
        expect_true: bool,
        cond_iidx: hir::InstIdx,
        guard_safepoint: &'static DeoptSafepoint,
        switch: Option<hir::Switch>,
    ) -> Result<(), CompilationError> {
        self.frames.last_mut().unwrap().pc_safepoint = Some(guard_safepoint);

        // If the condition variable is referenced in the guard's exit vars, we'll change it to
        // reference a const -- but we construct this as-needed.
        let mut cond_inverse_iidx = None;

        let mut deopt_frames = SmallVec::with_capacity(self.frames.len());
        let mut deopt_vars = Vec::with_capacity(
            self.frames
                .iter()
                .map(|x| x.pc_safepoint.unwrap().lives.len())
                .sum(),
        );
        for i in 0..self.frames.len() {
            let Frame {
                pc, pc_safepoint, ..
            } = &self.frames[i];
            let pc_safepoint = pc_safepoint.unwrap();
            let pc = if i + 1 < self.frames.len() {
                pc.clone().unwrap()
            } else {
                iid.clone()
            };
            for op in pc_safepoint.lives.iter() {
                let mut iidx = self.frames[i].get_local(&*self.opt, &op.to_inst_id());
                if iidx == cond_iidx {
                    if cond_inverse_iidx.is_none() {
                        let tyidx = self.opt.push_ty(hir::Ty::Int(1))?;
                        cond_inverse_iidx = Some(self.const_to_iidx(
                            tyidx,
                            hir::ConstKind::Int(ArbBitInt::from_u64(
                                1,
                                !u64::from(expect_true) & 0b1,
                            )),
                        )?);
                    }
                    iidx = cond_inverse_iidx.unwrap();
                }
                deopt_vars.push(iidx);
            }
            deopt_frames.push(hir::Frame {
                pc,
                pc_safepoint,
                #[cfg(test)]
                smapidx: hir::StackMapIdx::new(0),
            });
        }

        let hinst = hir::Guard {
            expect: expect_true,
            cond: cond_iidx,
            geidx: hir::GuardExtraIdx::MAX,
        };
        let gextra = hir::GuardExtra {
            bid,
            switch,
            deopt_vars,
            deopt_frames,
        };

        self.opt.feed_guard(hinst, gextra)?;
        Ok(())
    }

    /// This overwrites previous `(iid, inst)` mappings, which is necessary for unrolling to work.
    fn push_inst_and_link_local(
        &mut self,
        iid: InstId,
        inst: impl Into<hir::Inst>,
    ) -> Result<hir::InstIdx, CompilationError> {
        let iidx = self.opt.feed(inst.into())?;
        if self.addr_name_map.is_some()
            && let hir::Inst::Const(hir::Const {
                tyidx: _,
                kind: hir::ConstKind::Ptr(addr),
            }) = self.opt.inst(iidx)
        {
            self.addr_name_map
                .as_mut()
                .map(|x| x.insert(*addr, self.j2.dladdr(*addr)));
        }
        self.frames.last_mut().unwrap().set_local(iid, iidx);
        Ok(iidx)
    }

    /// Same as [push_inst_and_link_local] but for void instructions.
    fn push_void_inst_and_link_local(
        &mut self,
        iid: InstId,
        inst: impl Into<hir::Inst>,
    ) -> Result<Option<hir::InstIdx>, CompilationError> {
        let iidx = self.opt.feed_void(inst.into())?;
        if let Some(id) = iidx {
            self.frames.last_mut().unwrap().set_local(iid, id);
        }
        Ok(iidx)
    }

    fn const_to_iidx(
        &mut self,
        tyidx: hir::TyIdx,
        kind: hir::ConstKind,
    ) -> Result<hir::InstIdx, CompilationError> {
        // We could, if we want, do some sort of caching for constants so that we don't end up with
        // as many duplicate instructions.
        self.opt.feed(hir::Const { tyidx, kind }.into())
    }

    /// Translate a [TraceAction] to a [BBlockId]. If `ta` is not a mappable block, this will
    /// return `None`.
    fn ta_to_bid(&self, ta: &TraceAction) -> Option<BBlockId> {
        match ta {
            TraceAction::MappedAOTBBlock {
                funcidx: fidx,
                bbidx,
            } => {
                let fidx = FuncIdx::from(*fidx);
                if !self.am.func(fidx).is_declaration() {
                    Some(BBlockId::new(fidx, BBlockIdx::new(*bbidx)))
                } else {
                    None
                }
            }
            TraceAction::UnmappableBBlock => None,
            TraceAction::Promotion => unreachable!(),
        }
    }

    /// Process the current point of promotion data to a constant of type `ty`. This will update
    /// [Self::promotions_off].
    fn promotion_data_to_const(&mut self, ty: &Ty) -> Result<hir::InstIdx, CompilationError> {
        let (bitw, iidx) = match ty {
            Ty::Integer(x) => {
                let bitw = x.bitw();
                let v = match bitw {
                    1..=8 => u64::from(self.promotions[self.promotions_off]),
                    9..=16 => u64::from(u16::from_ne_bytes([
                        self.promotions[self.promotions_off],
                        self.promotions[self.promotions_off + 1],
                    ])),
                    17..=32 => u64::from(u32::from_ne_bytes([
                        self.promotions[self.promotions_off],
                        self.promotions[self.promotions_off + 1],
                        self.promotions[self.promotions_off + 2],
                        self.promotions[self.promotions_off + 3],
                    ])),
                    33..=64 => u64::from_ne_bytes([
                        self.promotions[self.promotions_off],
                        self.promotions[self.promotions_off + 1],
                        self.promotions[self.promotions_off + 2],
                        self.promotions[self.promotions_off + 3],
                        self.promotions[self.promotions_off + 4],
                        self.promotions[self.promotions_off + 5],
                        self.promotions[self.promotions_off + 6],
                        self.promotions[self.promotions_off + 7],
                    ]),
                    _ => todo!("{}", x.bitw()),
                };
                assert_eq!(ty.bitw(), x.bitw());
                let tyidx = self.opt.push_ty(hir::Ty::Int(bitw))?;
                (
                    bitw,
                    self.const_to_iidx(
                        tyidx,
                        hir::ConstKind::Int(ArbBitInt::from_u64(x.bitw(), v)),
                    )?,
                )
            }
            Ty::Void => todo!(),
            Ty::Ptr => {
                let v = match size_of::<usize>() {
                    8 => usize::from_ne_bytes([
                        self.promotions[self.promotions_off],
                        self.promotions[self.promotions_off + 1],
                        self.promotions[self.promotions_off + 2],
                        self.promotions[self.promotions_off + 3],
                        self.promotions[self.promotions_off + 4],
                        self.promotions[self.promotions_off + 5],
                        self.promotions[self.promotions_off + 6],
                        self.promotions[self.promotions_off + 7],
                    ]),
                    x => todo!("{x}"),
                };
                let ty = hir::Ty::Ptr(0);
                let bitw = ty.bitw();
                let tyidx = self.opt.push_ty(ty)?;
                (bitw, self.const_to_iidx(tyidx, hir::ConstKind::Ptr(v))?)
            }
            Ty::Func(_func_ty) => todo!(),
            Ty::Struct(_struct_ty) => todo!(),
            Ty::Float(_float_ty) => todo!(),
            Ty::Unimplemented(_) => todo!(),
        };
        self.promotions_off += usize::try_from(bitw.div_ceil(8)).unwrap();
        Ok(iidx)
    }

    fn vloc_arg_to_const(&mut self, vloc: &VarLoc<Reg>) -> Result<hir::InstIdx, CompilationError> {
        match vloc {
            VarLoc::Stack(_) => todo!(),
            VarLoc::StackOff(_) => todo!(),
            VarLoc::Reg(_, _) => todo!(),
            VarLoc::Const(kind) => match kind {
                hir::ConstKind::Double(_) => {
                    let tyidx = self.opt.push_ty(hir::Ty::Double)?;
                    self.opt.feed_arg(
                        hir::Const {
                            tyidx,
                            kind: kind.clone(),
                        }
                        .into(),
                    )
                }
                hir::ConstKind::Float(_) => todo!(),
                hir::ConstKind::Int(x) => {
                    let tyidx = self.opt.push_ty(hir::Ty::Int(x.bitw()))?;
                    self.opt.feed_arg(
                        hir::Const {
                            tyidx,
                            kind: kind.clone(),
                        }
                        .into(),
                    )
                }
                hir::ConstKind::Ptr(_) => {
                    let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                    self.opt.feed_arg(
                        hir::Const {
                            tyidx,
                            kind: kind.clone(),
                        }
                        .into(),
                    )
                }
            },
        }
    }

    /// Process the start of a (ControlPoint, Coupler | Loop | Return) trace.
    fn p_start_loop(
        &mut self,
        cp_bid: &BBlockId,
    ) -> Result<&'static DeoptSafepoint, CompilationError> {
        let cp_blk = self.am.bblock(cp_bid);
        let cp_iidx = BBlockInstIdx::new(
            cp_blk
                .insts
                .iter()
                .position(|x| x.is_control_point(self.am))
                .unwrap(),
        );
        let safepoint = cp_blk.insts[cp_iidx].safepoint().unwrap();
        assert!(self.frames.is_empty());
        self.frames.push(Frame {
            args: SmallVec::new(),
            locals: HashMap::new(),
            pc: Some(InstId::new(cp_bid.funcidx(), cp_bid.bbidx(), cp_iidx)),
            pc_safepoint: None,
            prev_pc: None,
        });

        for op in safepoint.lives.iter() {
            let tyidx = self.p_ty(op.type_(self.am))?;
            let iidx = self.opt.feed_arg(hir::Arg { tyidx }.into())?;
            self.frames
                .last_mut()
                .unwrap()
                .set_local(op.to_inst_id(), iidx);
        }

        Ok(safepoint)
    }

    /// Process the beginning of a (Guard, Coupler | Return) trace.
    fn p_start_side(
        &mut self,
        src_ctr: &Arc<J2CompiledTrace<Reg>>,
        src_gidx: CompiledGuardIdx,
        _tgt_ctr: &Arc<J2CompiledTrace<Reg>>,
    ) -> Result<Vec<VarLocs<Reg>>, CompilationError> {
        assert!(self.frames.is_empty());
        let guard = src_ctr.guard(src_gidx);
        let mut entry_vars = Vec::with_capacity(guard.deopt_vars.len());
        let mut deopt_vars_off = 0;
        for DeoptFrame { pc, pc_safepoint } in &guard.deopt_frames {
            let mut locals = HashMap::with_capacity(pc_safepoint.lives.len());
            for (iid, DeoptVar { fromvlocs, .. }) in
                pc_safepoint.lives.iter().map(|x| x.to_inst_id()).zip(
                    &guard.deopt_vars[deopt_vars_off..deopt_vars_off + pc_safepoint.lives.len()],
                )
            {
                let tyidx = self.p_ty(self.am.inst(&iid).def_type(self.am).unwrap())?;
                let iidx = if fromvlocs
                    .iter()
                    .any(|vloc| matches!(vloc, VarLoc::Const(_)))
                {
                    assert_eq!(fromvlocs.len(), 1);
                    self.vloc_arg_to_const(fromvlocs.iter().nth(0).unwrap())?
                } else {
                    self.opt.feed_arg(hir::Arg { tyidx }.into())?
                };
                locals.insert(iid.clone(), iidx);
                entry_vars.push(fromvlocs.clone());
            }
            self.frames.push(Frame {
                // aot_ir::Arg instructions come at the start of functions; by definition any frame
                // we're encountering will be past that point, so we can get away with pretending
                // there aren't any AOT arguments for any of the frames.
                args: smallvec![],
                locals,
                pc: Some(pc.clone()),
                pc_safepoint: Some(pc_safepoint),
                prev_pc: None,
            });
            deopt_vars_off += pc_safepoint.lives.len();
        }
        Ok(entry_vars)
    }

    /// Process a type.
    fn p_ty(&mut self, ty: &Ty) -> Result<hir::TyIdx, CompilationError> {
        let tyidx = match ty {
            Ty::Void => hir::Ty::Void,
            Ty::Integer(x) => hir::Ty::Int(x.bitw()),
            Ty::Ptr => {
                // FIXME: AOT IR doesn't yet tell us what the address space is, so we guess "0".
                hir::Ty::Ptr(0)
            }
            Ty::Func(fty) => {
                let rtn_tyidx = self.p_ty(self.am.type_(fty.ret_ty()))?;
                let mut args_tyidxs = SmallVec::with_capacity(fty.arg_tyidxs().len());
                for arg_ty in fty.arg_tyidxs() {
                    args_tyidxs.push(self.p_ty(self.am.type_(*arg_ty))?);
                }
                let fty = hir::FuncTy {
                    rtn_tyidx,
                    args_tyidxs,
                    has_varargs: fty.is_vararg(),
                };
                hir::Ty::Func(Box::new(fty))
            }
            Ty::Struct(_ty) => todo!(),
            Ty::Float(FloatTy::Double) => hir::Ty::Double,
            Ty::Float(FloatTy::Float) => hir::Ty::Float,
            Ty::Unimplemented(_) => todo!(),
        };
        self.opt.push_ty(tyidx)
    }

    /// Process an [Operand] and return the [hir::InstIdx] it references. Note: this can insert
    /// instructions into [self.opt]!
    fn p_operand(&mut self, op: &Operand) -> Result<hir::InstIdx, CompilationError> {
        match op {
            Operand::Const(cidx) => {
                let c = self.am.const_(*cidx).unwrap_val();
                let bytes = c.bytes();
                match self.am.type_(c.tyidx()) {
                    Ty::Integer(x) => {
                        // FIXME: It would be better if the AOT IR had converted these integers in advance
                        // rather than doing this dance here.
                        let v = match x.bitw() {
                            1 | 8 => {
                                debug_assert_eq!(bytes.len(), 1);
                                u64::from(bytes[0])
                            }
                            16 => {
                                debug_assert_eq!(bytes.len(), 2);
                                u64::from(u16::from_ne_bytes([bytes[0], bytes[1]]))
                            }
                            32 => {
                                debug_assert_eq!(bytes.len(), 4);
                                u64::from(u32::from_ne_bytes([
                                    bytes[0], bytes[1], bytes[2], bytes[3],
                                ]))
                            }
                            64 => {
                                debug_assert_eq!(bytes.len(), 8);
                                u64::from_ne_bytes([
                                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
                                    bytes[6], bytes[7],
                                ])
                            }
                            _ => todo!("{}", x.bitw()),
                        };
                        let tyidx = self.opt.push_ty(hir::Ty::Int(x.bitw()))?;
                        self.const_to_iidx(
                            tyidx,
                            hir::ConstKind::Int(ArbBitInt::from_u64(x.bitw(), v)),
                        )
                    }
                    Ty::Float(FloatTy::Double) => {
                        debug_assert_eq!(bytes.len(), 8);
                        let v = f64::from_ne_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]);
                        let tyidx = self.opt.push_ty(hir::Ty::Double)?;
                        self.const_to_iidx(tyidx, hir::ConstKind::Double(v))
                    }
                    Ty::Float(FloatTy::Float) => {
                        // FIXME: Floats are currently stored in AOT as doubles
                        // https://github.com/ykjit/yk/issues/1876
                        debug_assert_eq!(bytes.len(), 8);
                        let v = f64::from_ne_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]);
                        let v = v as f32;
                        let tyidx = self.opt.push_ty(hir::Ty::Float)?;
                        self.const_to_iidx(tyidx, hir::ConstKind::Float(v))
                    }
                    Ty::Ptr => {
                        debug_assert_eq!(bytes.len(), 8);
                        let v = u64::from_ne_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]);
                        let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                        self.const_to_iidx(tyidx, hir::ConstKind::Ptr(usize::try_from(v).unwrap()))
                    }
                    x => todo!("{x:?}"),
                }
            }
            Operand::Local(iid) => Ok(self.frames.last().unwrap().get_local(&*self.opt, iid)),
            Operand::Global(gidx) => {
                let gl = self.am.global_decl(*gidx);
                let (addr, inst) = if gl.is_threadlocal() {
                    let addr = self.j2.dlsym(gl.name(), true).unwrap().0;
                    assert!(!addr.is_null());
                    (addr.addr(), self.opt.feed(hir::ThreadLocal(addr).into()))
                } else {
                    let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                    let addr = self.globals[usize::from(*gidx)].addr();
                    (addr, self.const_to_iidx(tyidx, hir::ConstKind::Ptr(addr)))
                };
                self.addr_name_map
                    .as_mut()
                    .map(|x| x.insert(addr, Some(gl.name().to_owned())));
                inst
            }
            Operand::Func(fidx) => {
                let func = self.am.func(*fidx);
                let addr = self.j2.dlsym(func.name(), false).unwrap().0.addr();
                self.addr_name_map
                    .as_mut()
                    .map(|x| x.insert(addr, Some(func.name().to_owned())));
                let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                self.const_to_iidx(tyidx, hir::ConstKind::Ptr(addr))
            }
        }
    }

    /// Returns `Some(safepoint)` if an early return was encountered: parent code should stop
    /// examining the trace at this point.
    fn p_blocks(&mut self) -> Result<Option<&'static DeoptSafepoint>, CompilationError> {
        loop {
            let (pc, bid, blk) = {
                let mut pc = self.frames.last().unwrap().pc.clone().unwrap();
                let mut bid = BBlockId::new(pc.funcidx(), pc.bbidx());
                let mut blk = self.am.bblock(&bid);
                if usize::from(pc.iidx()) == blk.insts.len() {
                    let Some(ta) = self.ta_iter.next() else {
                        return Ok(None);
                    };
                    let cnd_bid = self.ta_to_bid(&ta?).unwrap();
                    blk = self.am.bblock(&cnd_bid);
                    let mut iidx = 0;
                    pc = InstId::new(cnd_bid.funcidx(), cnd_bid.bbidx(), BBlockInstIdx::new(iidx));
                    self.frames.last_mut().unwrap().pc = Some(pc.clone());
                    while iidx < blk.insts.len() {
                        let inst = &blk.insts[BBlockInstIdx::new(iidx)];
                        if let Inst::Phi { .. } = inst {
                            self.p_phi(pc, bid.bbidx(), cnd_bid, inst)?;
                            iidx += 1;
                            pc = InstId::new(
                                cnd_bid.funcidx(),
                                cnd_bid.bbidx(),
                                BBlockInstIdx::new(iidx),
                            );
                            self.frames.last_mut().unwrap().pc = Some(pc.clone());
                        } else {
                            break;
                        }
                    }
                    bid = cnd_bid;
                }
                (pc, bid, blk)
            };

            let inst = &blk.insts[pc.iidx()];
            match inst {
                Inst::Nop => todo!(),
                Inst::Alloca { .. } => todo!(),
                Inst::BinaryOp { .. } => self.p_binop(pc.clone(), inst)?,
                Inst::Br { .. } => (),
                Inst::Call { .. } => {
                    if self.p_call(pc.clone(), bid, inst)? {
                        continue;
                    }
                }
                Inst::Cast { .. } => self.p_cast(pc.clone(), inst)?,
                Inst::CondBr { .. } => self.p_condbr(pc.clone(), bid, inst)?,
                Inst::DebugStr { .. } => self.p_debugstr(pc.clone())?,
                Inst::ExtractValue { .. } => todo!(),
                Inst::FCmp { .. } => self.p_fcmp(pc.clone(), inst)?,
                Inst::FNeg { .. } => self.p_fneg(pc.clone(), inst)?,
                Inst::ICmp { .. } => self.p_icmp(pc.clone(), inst)?,
                Inst::IndirectCall { .. } => {
                    if self.p_icall(pc.clone(), bid, inst)? {
                        continue;
                    }
                }
                Inst::InsertValue { .. } => todo!(),
                Inst::Load { .. } => self.p_load(pc.clone(), inst)?,
                Inst::LoadArg { .. } => self.p_loadarg(pc.clone(), inst)?,
                Inst::Phi { .. } => unreachable!(),
                Inst::Promote { .. } => self.p_promote(pc.clone(), bid, inst)?,
                Inst::PtrAdd { .. } => self.p_ptradd(pc.clone(), inst)?,
                Inst::Ret { .. } => {
                    if let Some(x) = self.p_return(pc.clone(), inst)? {
                        // We encountered an early return.
                        return Ok(Some(x));
                    }
                }
                Inst::Select { .. } => self.p_select(pc.clone(), inst)?,
                Inst::Store { .. } => self.p_store(pc.clone(), inst)?,
                Inst::Switch { .. } => self.p_switch(pc.clone(), bid, inst, None)?,
                Inst::Unimplemented {
                    tyidx: _,
                    llvm_inst_str,
                } => {
                    return Err(CompilationError::General(format!(
                        "Unimplemented: '{}'",
                        llvm_inst_str.trim()
                    )));
                }
            }

            let frame = self.frames.last_mut().unwrap();
            let pc = frame.pc.clone().unwrap();
            frame.prev_pc = Some(pc.clone());
            frame.pc = Some(InstId::new(
                pc.funcidx(),
                pc.bbidx(),
                BBlockInstIdx::new(usize::from(pc.iidx()) + 1),
            ));
        }
    }

    fn p_binop(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::BinaryOp { lhs, binop, rhs } = inst else {
            panic!()
        };
        let lhs = self.p_operand(lhs)?;
        let rhs = self.p_operand(rhs)?;
        let inst: hir::Inst = match binop {
            BinOp::Add => hir::Add {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                nuw: false,
                nsw: false,
            }
            .into(),
            BinOp::And => hir::And {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
            BinOp::AShr => hir::AShr {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                exact: false,
            }
            .into(),
            BinOp::FAdd => hir::FAdd {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
            BinOp::FDiv => hir::FDiv {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
            BinOp::FMul => hir::FMul {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
            BinOp::FRem => todo!(),
            BinOp::FSub => hir::FSub {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
            BinOp::LShr => hir::LShr {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                exact: false,
            }
            .into(),
            BinOp::Mul => hir::Mul {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                nuw: false,
                nsw: false,
            }
            .into(),
            BinOp::Or => hir::Or {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                disjoint: false,
            }
            .into(),
            BinOp::SDiv => hir::SDiv {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                exact: false,
            }
            .into(),
            BinOp::SRem => hir::SRem {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
            BinOp::Shl => hir::Shl {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                nuw: false,
                nsw: false,
            }
            .into(),
            BinOp::Sub => hir::Sub {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                nuw: false,
                nsw: false,
            }
            .into(),
            BinOp::UDiv => hir::UDiv {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
                exact: false,
            }
            .into(),
            BinOp::URem => todo!(),
            BinOp::Xor => hir::Xor {
                tyidx: self.p_ty(inst.def_type(self.am).unwrap())?,
                lhs,
                rhs,
            }
            .into(),
        };
        self.push_inst_and_link_local(iid, inst).map(|_| ())
    }

    fn p_call(
        &mut self,
        iid: InstId,
        bid: BBlockId,
        inst: &'static Inst,
    ) -> Result<bool, CompilationError> {
        let Inst::Call {
            callee,
            args,
            safepoint,
        } = inst
        else {
            panic!()
        };

        // Ignore control point, or LLVM debug, calls.
        if inst.is_control_point(self.am) || inst.is_debug_call(self.am) {
            return Ok(false);
        }

        let func = self.am.func(*callee);
        // Ignore calls the software tracer makes to record blocks.
        #[cfg(tracer_swt)]
        if func.name() == "__yk_trace_basicblock" {
            return Ok(false);
        }

        let mut jargs = SmallVec::with_capacity(args.len());
        for x in args {
            jargs.push(self.p_operand(x)?);
        }

        if func.is_idempotent() {
            let Ty::Func(fty) = self.am.type_(func.tyidx()) else {
                panic!()
            };
            if jargs
                .iter()
                .all(|x| matches!(self.opt.inst(*x), hir::Inst::Const(_)))
            {
                let const_iidx = self.promotion_data_to_const(self.am.type_(fty.ret_ty()))?;
                self.frames.last_mut().unwrap().set_local(iid, const_iidx);
                self.outline_until(bid)?;
                return Ok(false);
            }
            self.promotions_off += usize::try_from(self.am.type_(fty.ret_ty()).bytew()).unwrap();
        }

        if !func.is_declaration()
            && !func.is_outline()
            // FIXME: We currently don't handle va_start.
            // It would be better if ykllvm marked functions containing `llvm.va_start.p*` with
            // `yk_outline` (at least until we can inline calls to that intrinsic).
            && !func.contains_call_to(self.am, "llvm.va_start.p0")
            // Is this a recursive call?
            && !self.frames.iter().any(|f| f.pc.as_ref().unwrap().funcidx() == *callee)
        {
            // Inlinable call.
            self.frames.last_mut().unwrap().pc_safepoint = Some(safepoint.as_ref().unwrap());
            self.frames.push(Frame {
                args: jargs,
                locals: HashMap::new(),
                pc: Some(InstId::new(
                    *callee,
                    BBlockIdx::new(0),
                    BBlockInstIdx::new(0),
                )),
                pc_safepoint: None,
                prev_pc: None,
            });
            let next_ta = &self.ta_iter.next().unwrap()?;
            let next_bid = self.ta_to_bid(next_ta).unwrap();
            assert_eq!(next_bid.funcidx(), *callee);
            assert_eq!(next_bid.bbidx(), BBlockIdx::new(0));
            Ok(true)
        } else {
            // Non-inlinable call. These come in two distinct flavours:
            //   1. LLVM intrinsics. We handle each of these individually.
            //   2. User-level calls. We emit a call instruction then skip any blocks we encounter
            //      (which could be zero, or many, and may include recursive calls) until that
            //      function returns.

            let ftyidx = self.p_ty(self.am.type_(func.tyidx()))?;

            // Handle LLVM intrinsics.
            if func.name().starts_with("llvm.") {
                self.p_llvm_intrinsic(iid, ftyidx, func.name(), jargs)?;
                return Ok(false);
            }

            // Handle user-level functions.
            let opt_fname = format!("__yk_opt_{}", func.name());
            let (fname, addr) = if let Some(x) = self.j2.dlsym(&opt_fname, false) {
                (opt_fname.as_str(), x.0.addr())
            } else {
                (
                    func.name(),
                    self.j2.dlsym(func.name(), false).unwrap().0.addr(),
                )
            };
            self.addr_name_map
                .as_mut()
                .map(|x| x.insert(addr, Some(fname.to_owned())));
            let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
            let tgt_iidx = self.const_to_iidx(tyidx, hir::ConstKind::Ptr(addr))?;

            let inst = hir::Call {
                tgt: tgt_iidx,
                func_tyidx: ftyidx,
                args: jargs,
            }
            .into();
            let hir::Ty::Func(box hir::FuncTy { rtn_tyidx, .. }) = self.opt.ty(ftyidx) else {
                panic!()
            };
            if *self.opt.ty(*rtn_tyidx) == hir::Ty::Void {
                self.opt.feed_void(inst)?;
            } else {
                self.push_inst_and_link_local(iid, inst)?;
            }
            self.outline_until(bid)?;
            Ok(false)
        }
    }

    fn p_icall(
        &mut self,
        iid: InstId,
        bid: BBlockId,
        inst: &'static Inst,
    ) -> Result<bool, CompilationError> {
        let Inst::IndirectCall {
            ftyidx,
            callop,
            args,
            safepoint: _,
        } = inst
        else {
            panic!()
        };

        let ftyidx = self.p_ty(self.am.type_(*ftyidx))?;
        let tgt_iidx = self.p_operand(callop)?;
        let mut jargs = SmallVec::with_capacity(args.len());
        for x in args {
            jargs.push(self.p_operand(x)?);
        }
        let inst = hir::Call {
            tgt: tgt_iidx,
            func_tyidx: ftyidx,
            args: jargs,
        };
        let hir::Ty::Func(box hir::FuncTy { rtn_tyidx, .. }) = self.opt.ty(ftyidx) else {
            panic!()
        };
        if *self.opt.ty(*rtn_tyidx) == hir::Ty::Void {
            self.opt.feed_void(inst.into())?;
        } else {
            self.push_inst_and_link_local(iid, inst)?;
        }
        self.outline_until(bid)?;
        Ok(false)
    }

    /// Outline until the successor block to `bid` is encountered. Returns `Err` if irregular
    /// control flow is detected.
    fn outline_until(&mut self, cur_bid: BBlockId) -> Result<(), CompilationError> {
        // Now we skip over all the blocks in this call.
        let tgt_bid = match self.am.bblock(&cur_bid).insts().last().unwrap() {
            Inst::Br { succ } => {
                // We can only stop outlining when we see the successor block and we are not in
                // the middle of recursion.
                BBlockId::new(cur_bid.funcidx(), *succ)
            }
            Inst::CondBr { .. } => {
                // Currently, the successor of a call is always an unconditional branch due to
                // the block spitting pass. However, there's a FIXME in that pass which could
                // lead to conditional branches showing up here. Leave a todo here so we know
                // when this happens.
                todo!()
            }
            _ => panic!(),
        };

        let mut prev_bid = cur_bid;
        let mut recurse = 0;
        loop {
            let ta = {
                let Some(Ok(ta)) = self.ta_iter.peek() else {
                    return Err(CompilationError::General(
                "irregular control flow detected (trace ended with outline successor pending)"
                    .into(),
            ));
                };
                ta.to_owned()
            };
            let cnd_bid = self.ta_to_bid(&ta).unwrap();
            self.check_correct_successor(prev_bid, cnd_bid)?;

            if recurse == 0 && cnd_bid == tgt_bid {
                break;
            }

            if cnd_bid.funcidx() == tgt_bid.funcidx() {
                if cnd_bid.is_entry() {
                    assert!(!self.am.bblock(&cnd_bid).is_return());
                    recurse += 1;
                } else if self.am.bblock(&cnd_bid).is_return() {
                    assert!(recurse > 0);
                    recurse -= 1;
                }
            }

            for inst in self.am.bblock(&cnd_bid).insts() {
                match inst {
                    Inst::Call { callee, .. } => {
                        let func = self.am.func(*callee);
                        if func.is_idempotent() {
                            let Ty::Func(fty) = self.am.type_(func.tyidx()) else {
                                panic!()
                            };
                            self.promotions_off +=
                                usize::try_from(self.am.type_(fty.ret_ty()).bytew()).unwrap();
                        }
                    }
                    Inst::Promote { tyidx, .. } => {
                        self.promotions_off +=
                            usize::try_from(self.am.type_(*tyidx).bytew()).unwrap();
                    }
                    Inst::DebugStr { .. } => {
                        self.debug_strs_seen += 1;
                    }
                    _ => (),
                }
            }

            prev_bid = cnd_bid;
            self.ta_iter.next();
        }
        Ok(())
    }

    /// Check that `cnd_bid` is a successor to `prev_bid` returning `Err` otherwise.
    fn check_correct_successor(
        &self,
        prev_bid: BBlockId,
        cnd_bid: BBlockId,
    ) -> Result<(), CompilationError> {
        if !cnd_bid.static_intraprocedural_successor_of(&prev_bid, self.am)
            && !cnd_bid.is_entry()
            && !self.am.bblock(&prev_bid).is_return()
        {
            // longjmp, or similar, has occurred.
            return Err(CompilationError::General(
                "irregular control flow detected (unexpected successor)".into(),
            ));
        }
        Ok(())
    }

    fn p_llvm_intrinsic(
        &mut self,
        iid: InstId,
        ftyidx: hir::TyIdx,
        name: &str,
        jargs: SmallVec<[hir::InstIdx; 1]>,
    ) -> Result<(), CompilationError> {
        assert!(name.starts_with("llvm."));
        match name.split_once(".").unwrap().1.split_once(".").unwrap().0 {
            "abs" => {
                let [src, int_min_poison]: [hir::InstIdx; 2] = jargs.into_vec().try_into().unwrap();
                let int_min_poison = if let hir::Inst::Const(hir::Const {
                    kind: hir::ConstKind::Int(x),
                    ..
                }) = &self.opt.inst(int_min_poison)
                {
                    x.to_zero_ext_u8().unwrap() != 0
                } else {
                    panic!()
                };
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::Abs {
                    tyidx: fty.rtn_tyidx,
                    val: src,
                    int_min_poison,
                };
                self.push_inst_and_link_local(iid, hinst).map(|_| ())
            }
            "ctpop" => {
                let [src]: [hir::InstIdx; 1] = jargs.into_vec().try_into().unwrap();
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::CtPop {
                    tyidx: fty.rtn_tyidx,
                    val: src,
                };
                self.push_inst_and_link_local(iid, hinst).map(|_| ())
            }
            "floor" => {
                let [src]: [hir::InstIdx; 1] = jargs.into_vec().try_into().unwrap();
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::Floor {
                    tyidx: fty.rtn_tyidx,
                    val: src,
                };
                self.push_inst_and_link_local(iid, hinst).map(|_| ())
            }
            "lifetime" => Ok(()),
            "memcpy" => {
                let [dst, src, len, volatile]: [hir::InstIdx; 4] =
                    jargs.into_vec().try_into().unwrap();
                let is_volatile = if let hir::Inst::Const(hir::Const {
                    kind: hir::ConstKind::Int(x),
                    ..
                }) = self.opt.inst(volatile)
                {
                    assert_eq!(x.bitw(), 1);
                    x.to_zero_ext_u8().unwrap() == 1
                } else {
                    panic!()
                };
                let hinst = hir::MemCpy {
                    dst,
                    src,
                    len,
                    is_volatile,
                };
                self.opt.feed_void(hinst.into()).map(|_| ())
            }
            "memset" => {
                let [dst, val, len, volatile]: [hir::InstIdx; 4] =
                    jargs.into_vec().try_into().unwrap();
                let is_volatile = if let hir::Inst::Const(hir::Const {
                    kind: hir::ConstKind::Int(x),
                    ..
                }) = self.opt.inst(volatile)
                {
                    assert_eq!(x.bitw(), 1);
                    x.to_zero_ext_u8().unwrap() == 1
                } else {
                    panic!()
                };
                let hinst = hir::MemSet {
                    dst,
                    val,
                    len,
                    is_volatile,
                };
                self.opt.feed_void(hinst.into()).map(|_| ())
            }
            "smax" => {
                let [lhs, rhs]: [hir::InstIdx; 2] = jargs.into_vec().try_into().unwrap();
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::SMax {
                    tyidx: fty.rtn_tyidx,
                    lhs,
                    rhs,
                };
                self.push_inst_and_link_local(iid, hinst).map(|_| ())
            }
            "smin" => {
                let [lhs, rhs]: [hir::InstIdx; 2] = jargs.into_vec().try_into().unwrap();
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::SMin {
                    tyidx: fty.rtn_tyidx,
                    lhs,
                    rhs,
                };
                self.push_inst_and_link_local(iid, hinst).map(|_| ())
            }
            n => todo!("{name} ('{n}')"),
        }
    }

    fn p_cast(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::Cast {
            cast_kind,
            val,
            dest_tyidx,
        } = inst
        else {
            panic!()
        };

        let val = self.p_operand(val)?;
        let tyidx = self.p_ty(self.am.type_(*dest_tyidx))?;
        let hinst: hir::Inst = match cast_kind {
            CastKind::SExt => hir::SExt { tyidx, val }.into(),
            CastKind::ZeroExtend => hir::ZExt { tyidx, val }.into(),
            CastKind::Trunc => hir::Trunc {
                tyidx,
                val,
                nuw: false,
                nsw: false,
            }
            .into(),
            CastKind::SIToFP => hir::SIToFP { tyidx, val }.into(),
            CastKind::FPExt => hir::FPExt { tyidx, val }.into(),
            CastKind::FPToSI => hir::FPToSI { tyidx, val }.into(),
            CastKind::BitCast => hir::BitCast { tyidx, val }.into(),
            CastKind::PtrToInt => hir::PtrToInt { tyidx, val }.into(),
            CastKind::IntToPtr => hir::IntToPtr { tyidx, val }.into(),
            CastKind::UIToFP => hir::UIToFP {
                tyidx,
                val,
                nneg: false,
            }
            .into(),
        };
        self.push_inst_and_link_local(iid, hinst).map(|_| ())
    }

    fn p_condbr(
        &mut self,
        iid: InstId,
        bid: BBlockId,
        inst: &'static Inst,
    ) -> Result<(), CompilationError> {
        let Inst::CondBr {
            cond,
            true_bb,
            false_bb,
            safepoint,
        } = inst
        else {
            panic!()
        };

        let next_bid = self.peek_next_bbid().unwrap();
        assert_eq!(
            next_bid.funcidx(),
            iid.funcidx(),
            "Control flow has diverged"
        );
        assert!(next_bid.bbidx() == *true_bb || next_bid.bbidx() == *false_bb);

        let cond_iidx = self.p_operand(cond)?;
        self.push_guard(
            bid,
            self.next_pc(iid),
            next_bid.bbidx() == *true_bb,
            cond_iidx,
            safepoint,
            None,
        )
    }

    fn p_debugstr(&mut self, iid: InstId) -> Result<(), CompilationError> {
        assert!(self.debug_strs_seen < self.debug_strs.len());
        let msg = self.debug_strs[self.debug_strs_seen].clone();
        self.debug_strs_seen += 1;
        self.push_void_inst_and_link_local(iid, hir::DebugStr(msg))?
            .unwrap();
        Ok(())
    }

    fn p_fcmp(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::FCmp {
            tyidx: _,
            lhs,
            pred,
            rhs,
        } = inst
        else {
            panic!()
        };
        let lhs = self.p_operand(lhs)?;
        let rhs = self.p_operand(rhs)?;
        let pred = match pred {
            FloatPredicate::False => hir::FPred::False,
            FloatPredicate::OrderedEqual => hir::FPred::Oeq,
            FloatPredicate::OrderedGreater => hir::FPred::Ogt,
            FloatPredicate::OrderedGreaterEqual => hir::FPred::Oge,
            FloatPredicate::OrderedLess => hir::FPred::Olt,
            FloatPredicate::OrderedLessEqual => hir::FPred::Ole,
            FloatPredicate::OrderedNotEqual => hir::FPred::One,
            FloatPredicate::Ordered => hir::FPred::Ord,
            FloatPredicate::Unordered => hir::FPred::Uno,
            FloatPredicate::UnorderedEqual => hir::FPred::Ueq,
            FloatPredicate::UnorderedGreater => hir::FPred::Ugt,
            FloatPredicate::UnorderedGreaterEqual => hir::FPred::Uge,
            FloatPredicate::UnorderedLess => hir::FPred::Ult,
            FloatPredicate::UnorderedLessEqual => hir::FPred::Ule,
            FloatPredicate::UnorderedNotEqual => hir::FPred::Une,
            FloatPredicate::True => hir::FPred::True,
        };
        self.push_inst_and_link_local(iid, hir::FCmp { pred, lhs, rhs })
            .map(|_| ())
    }

    fn p_fneg(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::FNeg { val } = inst else { panic!() };
        let tyidx = self.p_ty(val.type_(self.am))?;
        let val = self.p_operand(val)?;
        self.push_inst_and_link_local(iid, hir::FNeg { tyidx, val })
            .map(|_| ())
    }

    fn p_icmp(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::ICmp {
            tyidx: _,
            lhs,
            pred,
            rhs,
        } = inst
        else {
            panic!()
        };
        let lhs = self.p_operand(lhs)?;
        let rhs = self.p_operand(rhs)?;
        let pred = match pred {
            Predicate::Equal => hir::IPred::Eq,
            Predicate::NotEqual => hir::IPred::Ne,
            Predicate::UnsignedGreater => hir::IPred::Ugt,
            Predicate::UnsignedGreaterEqual => hir::IPred::Uge,
            Predicate::UnsignedLess => hir::IPred::Ult,
            Predicate::UnsignedLessEqual => hir::IPred::Ule,
            Predicate::SignedGreater => hir::IPred::Sgt,
            Predicate::SignedGreaterEqual => hir::IPred::Sge,
            Predicate::SignedLess => hir::IPred::Slt,
            Predicate::SignedLessEqual => hir::IPred::Sle,
        };
        self.push_inst_and_link_local(
            iid,
            hir::ICmp {
                pred,
                lhs,
                rhs,
                samesign: false,
            },
        )
        .map(|_| ())
    }

    fn p_load(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::Load {
            ptr,
            tyidx,
            volatile,
        } = inst
        else {
            panic!()
        };
        let tyidx = self.p_ty(self.am.type_(*tyidx))?;
        let ptr = self.p_operand(ptr)?;
        self.push_inst_and_link_local(
            iid,
            hir::Load {
                tyidx,
                ptr,
                is_volatile: *volatile,
            },
        )
        .map(|_| ())
    }

    fn p_loadarg(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::LoadArg { arg_idx, ty_idx: _ } = inst else {
            panic!()
        };
        let iidx = self.frames.last().unwrap().args[*arg_idx];
        self.frames.last_mut().unwrap().set_local(iid, iidx);
        Ok(())
    }

    fn p_phi(
        &mut self,
        iid: InstId,
        prev_bidx: BBlockIdx,
        _bid: BBlockId,
        inst: &Inst,
    ) -> Result<(), CompilationError> {
        let Inst::Phi {
            tyidx: _,
            incoming_bbs,
            incoming_vals,
        } = inst
        else {
            panic!()
        };
        let v = &incoming_vals[incoming_bbs.iter().position(|x| *x == prev_bidx).unwrap()];
        let op = self.p_operand(v)?;
        self.frames.last_mut().unwrap().set_local(iid, op);
        Ok(())
    }

    fn p_promote(
        &mut self,
        iid: InstId,
        bid: BBlockId,
        inst: &'static Inst,
    ) -> Result<(), CompilationError> {
        let Inst::Promote {
            tyidx,
            val,
            safepoint,
        } = inst
        else {
            panic!()
        };

        let val_iidx = self.p_operand(val)?;
        self.frames
            .last_mut()
            .unwrap()
            .set_local(iid.clone(), val_iidx);
        let _bitw = self.opt.inst_bitw(&*self.opt, val_iidx);
        let const_iidx = self.promotion_data_to_const(self.am.type_(*tyidx))?;

        let icmp = self.opt.feed(
            hir::ICmp {
                pred: hir::IPred::Eq,
                lhs: val_iidx,
                rhs: const_iidx,
                samesign: false,
            }
            .into(),
        )?;
        self.push_guard(bid, self.next_pc(iid.clone()), true, icmp, safepoint, None)
    }

    fn p_ptradd(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::PtrAdd {
            tyidx: _,
            ptr,
            const_off,
            dyn_elem_counts,
            dyn_elem_sizes,
        } = inst
        else {
            panic!()
        };
        let mut ptr = self.p_operand(ptr)?;
        if *const_off != 0 {
            ptr = self.opt.feed(
                hir::PtrAdd {
                    ptr,
                    // LLVM only allows 32 bit offsets, so this should never fail.
                    off: i32::try_from(*const_off).unwrap(),
                    in_bounds: false,
                    nusw: false,
                    nuw: false,
                }
                .into(),
            )?;
        };

        for (num_elems, elem_size) in dyn_elem_counts.iter().zip(dyn_elem_sizes) {
            let num_elems = self.p_operand(num_elems)?;
            // If the element count is not the same width as LLVM's GEP index type, then we have to
            // sign extend it up (or truncate it down) to the right size. We've not yet
            // seen this in the wild.
            assert_eq!(
                self.opt.inst_bitw(&*self.opt, num_elems),
                u32::from(self.am.ptr_off_bitsize())
            );
            let elem_size = u32::try_from(*elem_size).map_err(|_| {
                CompilationError::LimitExceeded("PtrAdd elem_size doesn't fit in u32".into())
            })?;
            ptr = self.opt.feed(
                hir::DynPtrAdd {
                    ptr,
                    num_elems,
                    elem_size,
                }
                .into(),
            )?;
        }

        self.frames.last_mut().unwrap().set_local(iid, ptr);
        Ok(())
    }

    /// Return `Some(safepoint)` if this is an early return: this must stop further examination of
    /// the trace.
    fn p_return(
        &mut self,
        _iid: InstId,
        inst: &Inst,
    ) -> Result<Option<&'static DeoptSafepoint>, CompilationError> {
        let Inst::Ret { val } = inst else { panic!() };

        let val = match val {
            Some(x) => Some(self.p_operand(x)?),
            None => None,
        };

        let frame = self.frames.pop().unwrap();
        if !self.frames.is_empty() {
            if let Some(val_iidx) = val {
                let frame = self.frames.last_mut().unwrap();
                frame.set_local(frame.pc.clone().unwrap(), val_iidx);
            }
            Ok(None)
        } else {
            // We've returned out of the function that started tracing. Stop processing any
            // remaining blocks and emit a return instruction that naturally returns from a
            // compiled trace into the interpreter.
            let safepoint = frame.pc_safepoint.unwrap();
            // We currently don't support passing values back during early returns.
            assert!(val.is_none());
            self.opt.feed_void(hir::Term(Vec::new()).into())?;
            Ok(Some(safepoint))
        }
    }

    fn p_select(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::Select {
            cond,
            trueval,
            falseval,
        } = inst
        else {
            panic!()
        };
        let tyidx = self.p_ty(inst.def_type(self.am).unwrap())?;
        let cond = self.p_operand(cond)?;
        let truev = self.p_operand(trueval)?;
        let falsev = self.p_operand(falseval)?;
        self.push_inst_and_link_local(
            iid,
            hir::Select {
                tyidx,
                cond,
                truev,
                falsev,
            },
        )
        .map(|_| ())
    }

    fn p_store(&mut self, _iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::Store { val, tgt, volatile } = inst else {
            panic!()
        };
        let ptr = self.p_operand(tgt)?;
        let val = self.p_operand(val)?;
        self.opt
            .feed_void(
                hir::Store {
                    ptr,
                    val,
                    is_volatile: *volatile,
                }
                .into(),
            )
            .map(|_| ())
    }

    fn p_switch(
        &mut self,
        iid: InstId,
        bid: BBlockId,
        inst: &'static Inst,
        seen_bbidxs: Option<&[BBlockIdx]>,
    ) -> Result<(), CompilationError> {
        let Inst::Switch {
            test_val,
            default_dest: _,
            case_values,
            case_dests,
            safepoint,
        } = inst
        else {
            panic!()
        };
        assert_eq!(case_values.len(), case_dests.len());
        assert!(!case_values.is_empty());

        let tyidx = self.p_ty(test_val.type_(self.am))?;
        let bitw = self.opt.ty(tyidx).bitw();
        let next_ta = self
            .ta_iter
            .peek()
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        let next_bbidx = self.ta_to_bid(&next_ta).unwrap().bbidx();

        let (expect, check_vals) = match case_dests.iter().position(|x| *x == next_bbidx) {
            Some(_i) => {
                // A non-default case
                (
                    true,
                    // Multiple `case_values` might map to the same block, so we need to guard all
                    // of the values which point to that block.
                    case_dests
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| *x == &next_bbidx)
                        .map(|(i, _)| case_values[i])
                        .collect::<Vec<_>>(),
                )
            }
            None => {
                // We've hit a `default` case. We need to guard all of the `case_values` we've not
                // yet seen.
                (
                    false,
                    case_dests
                        .iter()
                        .zip(case_values)
                        .filter(|(x, _)| !seen_bbidxs.is_some_and(|y| y.contains(*x)))
                        .map(|(_, x)| *x)
                        .collect::<Vec<_>>(),
                )
            }
        };

        if !check_vals.is_empty() {
            let val_iidx = self.p_operand(test_val)?;
            let mut icmp = {
                let const_iidx = self.const_to_iidx(
                    tyidx,
                    hir::ConstKind::Int(ArbBitInt::from_u64(bitw, check_vals[0])),
                )?;
                self.opt.feed(
                    hir::ICmp {
                        pred: hir::IPred::Eq,
                        lhs: val_iidx,
                        rhs: const_iidx,
                        samesign: false,
                    }
                    .into(),
                )?
            };

            // If there is more than one value that we need to check, then we generate, in essence
            // a fold.
            let i1_tyidx = self.opt.push_ty(hir::Ty::Int(1))?;
            for v in &check_vals[1..] {
                let const_iidx =
                    self.const_to_iidx(tyidx, hir::ConstKind::Int(ArbBitInt::from_u64(bitw, *v)))?;
                let next_icmp = self.opt.feed(
                    hir::ICmp {
                        pred: hir::IPred::Eq,
                        lhs: val_iidx,
                        rhs: const_iidx,
                        samesign: false,
                    }
                    .into(),
                )?;
                icmp = self.opt.feed(
                    hir::Or {
                        tyidx: i1_tyidx,
                        lhs: icmp,
                        rhs: next_icmp,
                        disjoint: false,
                    }
                    .into(),
                )?;
            }

            let seen_bbidxs = if let Some(seen_bbidxs) = seen_bbidxs {
                seen_bbidxs
                    .iter()
                    .cloned()
                    .chain([next_bbidx])
                    .collect::<Vec<_>>()
            } else {
                vec![next_bbidx]
            };
            self.push_guard(
                bid,
                iid.clone(),
                expect,
                icmp,
                safepoint,
                Some(hir::Switch { iid, seen_bbidxs }),
            )?;
        }

        Ok(())
    }
}

/// The information needed to build a HIR trace.
pub(super) enum BuildKind {
    Coupler {
        tgt_ctr: Arc<dyn CompiledTrace>,
    },
    Loop,
    Side {
        src_ctr: Arc<dyn CompiledTrace>,
        src_gid: GuardId,
        tgt_ctr: Arc<dyn CompiledTrace>,
    },
}

/// An intermediate enum to bridge [BuildKind] and [hir::ModKind]. It allows us to gradually build
/// up some intermediate state that isn't present in [BuildKind] but which is needed by
/// [hir::ModKind], while keeping the latter enum simple.
enum BuildModKind<Reg: RegT> {
    Coupler {
        entry_safepoint: &'static DeoptSafepoint,
        tgt_ctr: Arc<J2CompiledTrace<Reg>>,
    },
    Loop {
        entry_safepoint: &'static DeoptSafepoint,
    },
    Side {
        prev_bid: BBlockId,
        args_vlocs: Vec<VarLocs<Reg>>,
        src_ctr: Arc<J2CompiledTrace<Reg>>,
        src_gidx: CompiledGuardIdx,
        tgt_ctr: Arc<J2CompiledTrace<Reg>>,
    },
}

/// An inlined frame.
#[derive(Debug)]
struct Frame {
    /// This frame's arguments. This is not mutated after frame creation.
    args: SmallVec<[hir::InstIdx; 1]>,
    locals: HashMap<InstId, hir::InstIdx>,
    pc: Option<InstId>,
    /// The current safepoint for this frame. This has no initial value at frame entry, and is
    /// updated at every call site.
    pc_safepoint: Option<&'static DeoptSafepoint>,
    prev_pc: Option<InstId>,
}

impl Frame {
    /// Lookup the AOT variable `iid` relative to `opt`.
    fn get_local(&self, opt: &dyn OptT, iid: &InstId) -> hir::InstIdx {
        opt.equiv_iidx(self.locals[iid])
    }

    /// Set the AOT variable `iid` mapping to HIR `iidx`. This is allowed to override previous
    /// bindings (which occur due to unrolling).
    fn set_local(&mut self, iid: InstId, iidx: hir::InstIdx) {
        self.locals.insert(iid, iidx);
    }
}

struct TraceActionIterator {
    ta_iter: Peekable<Box<dyn crate::trace::AOTTraceIterator>>,
}

impl TraceActionIterator {
    fn new(ta_iter: Box<dyn crate::trace::AOTTraceIterator>) -> Self {
        Self {
            ta_iter: ta_iter.peekable(),
        }
    }
}

impl Iterator for TraceActionIterator {
    type Item = Result<TraceAction, CompilationError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.ta_iter
            .next()
            .map(|x| x.map_err(|e| CompilationError::General(e.to_string())))
    }
}
