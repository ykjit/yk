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
            compiled_trace::{J2CompiledTrace, J2CompiledTraceKind},
            hir::{self, GuardRestore, GuardRestoreIdx},
            opt::{OptT, noopt::NoOpt, opt::Opt},
            regalloc::{RegT, VarLoc, VarLocs},
        },
        jitc_yk::{AOT_MOD, aot_ir::*, arbbitint::ArbBitInt},
    },
    location::HotLocation,
    log::{IRPhase, log_ir, should_log_any_ir, should_log_ir, stats::TimingState},
    mt::{MT, TraceId},
    trace::TraceAction,
};
use index_vec::IndexVec;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use std::{
    assert_matches::assert_matches,
    collections::HashMap,
    ffi::{CString, c_void},
    iter::Peekable,
    marker::PhantomData,
    sync::Arc,
};

/// The symbol name of the global variable pointers array.
const GLOBAL_PTR_ARRAY_SYM: &str = "__yk_globalvar_ptrs";

pub(super) struct AotToHir<Reg: RegT> {
    mt: Arc<MT>,
    /// The AOT IR.
    am: &'static Module,
    hl: Arc<Mutex<HotLocation>>,
    ta_iter: Peekable<TraceActionIterator>,
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
    guard_restores: IndexVec<GuardRestoreIdx, GuardRestore>,
    /// Initially set to `None` until we find the locations for this trace's arguments.
    frames: Vec<Frame>,
    /// Cache dlsym() lookups.
    dlsym_cache: HashMap<String, Option<*const c_void>>,
    /// If logging is enabled, create a map of addresses -> names to make IR printing nicer.
    addr_name_map: Option<HashMap<usize, String>>,
    /// The JIT IR this struct builds.
    phantom: PhantomData<Reg>,
}

impl<Reg: RegT + 'static> AotToHir<Reg> {
    pub(super) fn new(
        mt: &Arc<MT>,
        am: &'static Module,
        hl: Arc<Mutex<HotLocation>>,
        ta_iter: Box<dyn crate::trace::AOTTraceIterator>,
        trid: TraceId,
        bkind: BuildKind,
        promotions: Box<[u8]>,
        _debug_strs: Vec<String>,
    ) -> Self {
        let globals = {
            let cn = CString::new(GLOBAL_PTR_ARRAY_SYM).unwrap();
            let ptr = unsafe { libc::dlsym(std::ptr::null_mut(), cn.as_c_str().as_ptr()) }
                as *const *const ();
            assert!(!ptr.is_null());
            unsafe { std::slice::from_raw_parts(ptr, am.global_decls_len()) }
        };

        let opt = if mt.opt_level() == 0 {
            Box::new(NoOpt::new()) as Box<dyn OptT>
        } else {
            Box::new(Opt::new())
        };

        Self {
            mt: Arc::clone(mt),
            am,
            hl,
            ta_iter: TraceActionIterator::new(ta_iter).peekable(),
            trid,
            bkind,
            promotions,
            promotions_off: 0,
            globals,
            opt,
            guard_restores: IndexVec::new(),
            frames: Vec::new(),
            dlsym_cache: HashMap::new(),
            addr_name_map: should_log_any_ir().then_some(HashMap::new()),
            phantom: PhantomData,
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
            BuildKind::Coupler => todo!(),
            BuildKind::Loop => {
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
                BuildModKind::Loop { entry_safepoint }
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
                let src_gridx = hir::GuardRestoreIdx::from(usize::from(*src_gid));
                let prev_bid = src_ctr.bid(src_gridx);
                let tgt_ctr = Arc::clone(tgt_ctr)
                    .as_any()
                    .downcast::<J2CompiledTrace<Reg>>()
                    .unwrap();
                let entry_vlocs = self.p_start_side(&src_ctr, src_gridx, &tgt_ctr)?;
                if let Some(hir::Switch {
                    iid,
                    seen_bbidxs: seen_blocks,
                }) = src_ctr.switch(src_gridx)
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
                    entry_vlocs,
                    src_ctr,
                    src_gridx,
                    tgt_ctr,
                }
            }
        };

        let mut prev_bid = match &bmk {
            BuildModKind::Coupler => todo!(),
            BuildModKind::Loop { .. } => None,
            BuildModKind::Side { prev_bid, .. } => Some(*prev_bid),
        };

        // If we encounter a return in a side-trace, [early_return] will be set to true, and no
        // further processing of the trace should occur.
        let mut early_return = false;
        while let Some(ta) = self.ta_iter.next() {
            if let Some(bid) = self.ta_to_bid(&ta?) {
                if self.p_block(prev_bid, bid)? {
                    // We encountered an early return.
                    early_return = true;
                    break;
                }
                prev_bid = Some(bid);
            }
        }

        assert_eq!(self.promotions_off, self.promotions.len());

        if !early_return {
            assert_eq!(self.frames.len(), 1);
            let exit_safepoint = match &bmk {
                BuildModKind::Coupler => todo!(),
                BuildModKind::Loop { entry_safepoint } => entry_safepoint,
                BuildModKind::Side { tgt_ctr, .. } => match &tgt_ctr.kind {
                    J2CompiledTraceKind::Loop {
                        entry_safepoint, ..
                    } => entry_safepoint,
                    J2CompiledTraceKind::Side { .. } => todo!(),
                    #[cfg(test)]
                    J2CompiledTraceKind::Test => unreachable!(),
                },
            };
            let exit_vars = exit_safepoint
                .lives
                .iter()
                .map(|x| self.frames[0].get_local(&*self.opt, &x.to_inst_id()))
                .collect::<Vec<_>>();
            self.opt.push_inst(hir::Inst::Exit(hir::Exit(exit_vars)))?;
        }

        let (entry, tys) = self.opt.build();
        let mk = match bmk {
            BuildModKind::Coupler => todo!(),
            BuildModKind::Loop { entry_safepoint } => hir::ModKind::Loop {
                entry_safepoint,
                entry,
                inner: None,
            },
            BuildModKind::Side {
                entry_vlocs,
                src_ctr,
                src_gridx,
                tgt_ctr,
                ..
            } => hir::ModKind::Side {
                entry,
                entry_vlocs,
                src_ctr,
                src_gridx,
                tgt_ctr,
            },
        };

        let m = hir::Mod {
            trid: self.trid,
            kind: mk,
            tys,
            addr_name_map: self.addr_name_map,
            guard_restores: self.guard_restores,
        };

        let ds = if let Some(x) = &self.hl.lock().debug_str {
            format!(": {}", x.as_str())
        } else {
            "".to_owned()
        };

        if should_log_ir(IRPhase::DebugStrs) {
            todo!();
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

    /// Convert a name to an address: this is effectively a caching front-end to [libc::dlsym].
    fn dlsym(&mut self, sym: &str) -> Option<*const c_void> {
        if let Some(x) = self.dlsym_cache.get(sym) {
            return *x;
        }
        let cn = CString::new(sym).unwrap();
        let ptr =
            unsafe { libc::dlsym(std::ptr::null_mut(), cn.as_c_str().as_ptr()) } as *const c_void;
        let r = if ptr.is_null() { None } else { Some(ptr) };
        self.dlsym_cache.insert(sym.to_owned(), r);
        r
    }

    fn peek_next_bbid(&mut self) -> Option<BBlockId> {
        self.ta_iter
            .peek()
            .and_then(|x| x.as_ref().ok())
            .cloned()
            .and_then(|x| self.ta_to_bid(&x))
    }

    fn push_guard(
        &mut self,
        bid: BBlockId,
        _iid: InstId,
        expect_true: bool,
        cond_iidx: hir::InstIdx,
        guard_safepoint: &'static DeoptSafepoint,
        switch: Option<hir::Switch>,
    ) -> Result<(), CompilationError> {
        self.frames.last_mut().unwrap().call_safepoint = Some(guard_safepoint);
        let mut deopt_frames = SmallVec::with_capacity(self.frames.len());
        for (
            i,
            frame @ Frame {
                call_iid,
                func,
                call_safepoint,
                ..
            },
        ) in self.frames.iter().enumerate()
        {
            let safepoint = if i + 1 < self.frames.len() {
                call_safepoint.unwrap()
            } else {
                guard_safepoint
            };
            deopt_frames.push(hir::Frame {
                safepoint,
                call_iid: call_iid.to_owned(),
                func: *func,
                exit_vars: safepoint
                    .lives
                    .iter()
                    .map(|x| frame.get_local(&*self.opt, &x.to_inst_id()))
                    .collect::<Vec<_>>(),
            });
        }

        // In many cases, the last variable in a guards' list of variables is the condition
        // variable: by definition, we know that if the guard is taken the value will be
        // `!expect_true`. We could leave this for the optimiser, but it's cheaper to do it here.
        if let Operand::Local(last_iid) = guard_safepoint.lives.last().unwrap()
            && cond_iidx == self.frames.last().unwrap().get_local(&*self.opt, last_iid)
        {
            let tyidx = self.opt.push_ty(hir::Ty::Int(1))?;
            let ciidx = self.const_to_iidx(
                tyidx,
                hir::ConstKind::Int(ArbBitInt::from_u64(1, !u64::from(expect_true) & 0b1)),
            )?;
            // This `let` is only needed because type inference goes a bit wonky, at least on
            // rust-1.91.
            let last: &mut hir::Frame = deopt_frames.last_mut().unwrap();
            *last.exit_vars.last_mut().unwrap() = ciidx;
        }

        // This is temporary, since we currently don't put any instructions in the guard body:
        // when we do, entry_vars and exit_vars will, in general, be different to each other.
        let entry_vars = deopt_frames
            .iter()
            .flat_map(|hir::Frame { exit_vars, .. }| exit_vars.to_owned())
            .collect::<Vec<_>>();
        let gridx = self.guard_restores.push(hir::GuardRestore {
            exit_frames: deopt_frames,
        });
        let hinst = hir::Guard {
            expect: expect_true,
            cond: cond_iidx,
            entry_vars,
            gridx,
            bid,
            switch: switch.map(Box::new),
        };

        // We now try pushing the guard instruction...
        let iidx = self.opt.push_inst(hinst.into())?;
        // ...but if it turned into a non-guard then it means the guard was optimised away and we
        // should remove the corresponding [GuardRestore].
        match self.opt.inst(iidx) {
            hir::Inst::Guard(hir::Guard {
                gridx: pushed_gridx,
                ..
            }) => {
                // If this fails, it means the optimiser has returned a reference to an older
                // guard. That seems a reasonable thing to do, but I haven't thought it through
                // yet.
                assert_eq!(gridx, *pushed_gridx);
            }
            _ => {
                self.guard_restores
                    .remove(self.guard_restores.len_idx() - 1);
            }
        }
        Ok(())
    }

    /// This overwrites previous `(iid, inst)` mappings, which is necessary for unrolling to work.
    fn push_inst_and_link_local(
        &mut self,
        iid: InstId,
        inst: hir::Inst,
    ) -> Result<hir::InstIdx, CompilationError> {
        let iidx = self.opt.push_inst(inst)?;
        self.frames.last_mut().unwrap().set_local(iid, iidx);
        Ok(iidx)
    }

    fn const_to_iidx(
        &mut self,
        tyidx: hir::TyIdx,
        kind: hir::ConstKind,
    ) -> Result<hir::InstIdx, CompilationError> {
        // We could, if we want, do some sort of caching for constants so that we don't end up with
        // as many duplicate instructions.
        self.opt.push_inst(hir::Const { tyidx, kind }.into())
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
        self.promotions_off += usize::try_from(bitw.next_multiple_of(8) / 8).unwrap();
        Ok(iidx)
    }

    fn vloc_to_const(&mut self, vloc: &VarLoc<Reg>) -> Result<hir::InstIdx, CompilationError> {
        match vloc {
            VarLoc::Stack(_) => todo!(),
            VarLoc::StackOff(_) => todo!(),
            VarLoc::Reg(_) => todo!(),
            VarLoc::Const(kind) => match kind {
                hir::ConstKind::Double(_) => todo!(),
                hir::ConstKind::Float(_) => todo!(),
                hir::ConstKind::Int(x) => {
                    let tyidx = self.opt.push_ty(hir::Ty::Int(x.bitw()))?;
                    self.opt.push_inst(
                        hir::Const {
                            tyidx,
                            kind: kind.clone(),
                        }
                        .into(),
                    )
                }
                hir::ConstKind::Ptr(_) => {
                    let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                    self.opt.push_inst(
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

    /// Process the start of a loop trace.
    fn p_start_loop(
        &mut self,
        cp_bid: &BBlockId,
    ) -> Result<&'static DeoptSafepoint, CompilationError> {
        let cp_blk = self.am.bblock(cp_bid);
        let inst = cp_blk
            .insts
            .iter()
            .find(|x| x.is_control_point(self.am))
            .unwrap();
        let safepoint = inst.safepoint().unwrap();
        assert!(self.frames.is_empty());
        self.frames.push(Frame {
            call_iid: None,
            func: cp_bid.funcidx(),
            call_safepoint: None,
            args: SmallVec::new(),
            locals: HashMap::new(),
        });

        for op in safepoint.lives.iter() {
            let tyidx = self.p_ty(op.type_(self.am))?;
            self.push_inst_and_link_local(op.to_inst_id(), hir::Arg { tyidx }.into())?;
        }

        Ok(safepoint)
    }

    /// Process the beginning of a side-trace.
    fn p_start_side(
        &mut self,
        src_ctr: &Arc<J2CompiledTrace<Reg>>,
        src_gridx: hir::GuardRestoreIdx,
        _tgt_ctr: &Arc<J2CompiledTrace<Reg>>,
    ) -> Result<Vec<VarLocs<Reg>>, CompilationError> {
        assert!(self.frames.is_empty());
        let dframes = src_ctr.deopt_frames(src_gridx);
        let mut entry_vars = Vec::new();
        for dframe in dframes {
            assert_eq!(dframe.vars.len(), dframe.safepoint.lives.len());
            let mut locals = HashMap::with_capacity(dframe.vars.len());
            for (iid, _bitw, fromvlocs, _tovlocs) in &dframe.vars {
                let tyidx = self.p_ty(self.am.inst(iid).def_type(self.am).unwrap())?;
                let iidx = if fromvlocs
                    .iter()
                    .any(|vloc| matches!(vloc, VarLoc::Const(_)))
                {
                    assert_eq!(fromvlocs.len(), 1);
                    self.vloc_to_const(fromvlocs.iter().nth(0).unwrap())?
                } else {
                    self.opt.push_inst(hir::Arg { tyidx }.into())?
                };
                locals.insert(iid.clone(), iidx);
                entry_vars.push(fromvlocs.clone());
            }
            self.frames.push(Frame {
                call_iid: dframe.call_iid.clone(),
                func: dframe.func,
                call_safepoint: Some(dframe.safepoint),
                // aot_ir::Arg instructions come at the start of functions; by definition any frame
                // we're encountering will be past that point, so we can get away with pretending
                // there aren't any AOT arguments for any of the frames.
                args: smallvec![],
                locals,
            });
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
                    let addr = self.dlsym(gl.name()).unwrap();
                    assert!(!addr.is_null());
                    (
                        addr.addr(),
                        self.opt.push_inst(hir::ThreadLocal(addr).into()),
                    )
                } else {
                    let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                    let addr = self.globals[usize::from(*gidx)].addr();
                    (addr, self.const_to_iidx(tyidx, hir::ConstKind::Ptr(addr)))
                };
                self.addr_name_map
                    .as_mut()
                    .map(|x| x.insert(addr, gl.name().to_owned()));
                inst
            }
            Operand::Func(fidx) => {
                let func = self.am.func(*fidx);
                let addr = self.dlsym(func.name()).unwrap().addr();
                self.addr_name_map
                    .as_mut()
                    .map(|x| x.insert(addr, func.name().to_owned()));
                let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
                self.const_to_iidx(tyidx, hir::ConstKind::Ptr(addr))
            }
        }
    }

    /// Returns `true` if an early return was encountered: parent code should stop examining the
    /// trace at this point.
    fn p_block(
        &mut self,
        prev_bid: Option<BBlockId>,
        bid: BBlockId,
    ) -> Result<bool, CompilationError> {
        let blk = self.am.bblock(&bid);
        for (i, inst) in blk.insts.iter().enumerate() {
            let iid = InstId::new(bid.funcidx(), bid.bbidx(), BBlockInstIdx::new(i));
            match inst {
                Inst::Nop => todo!(),
                Inst::Alloca { .. } => todo!(),
                Inst::BinaryOp { .. } => self.p_binop(iid, inst)?,
                Inst::Br { .. } => (),
                Inst::Call { .. } => self.p_call(iid, bid, inst)?,
                Inst::Cast { .. } => self.p_cast(iid, inst)?,
                Inst::CondBr { .. } => self.p_condbr(iid, bid, inst)?,
                Inst::DebugStr { .. } => todo!(),
                Inst::ExtractValue { .. } => todo!(),
                Inst::FCmp { .. } => self.p_fcmp(iid, inst)?,
                Inst::FNeg { val: _ } => todo!(),
                Inst::ICmp { .. } => self.p_icmp(iid, inst)?,
                Inst::IndirectCall { .. } => self.p_icall(iid, bid, inst)?,
                Inst::InsertValue { .. } => todo!(),
                Inst::Load { .. } => self.p_load(iid, inst)?,
                Inst::LoadArg { .. } => self.p_loadarg(iid, inst)?,
                Inst::Phi { .. } => self.p_phi(iid, prev_bid.unwrap().bbidx(), bid, inst)?,
                Inst::Promote { .. } => self.p_promote(iid, bid, inst)?,
                Inst::PtrAdd { .. } => self.p_ptradd(iid, inst)?,
                Inst::Ret { .. } => {
                    if self.p_ret(iid, inst)? {
                        // We encountered an early return in a side-trace.
                        return Ok(true);
                    }
                }
                Inst::Select { .. } => self.p_select(iid, inst)?,
                Inst::Store { .. } => self.p_store(iid, inst)?,
                Inst::Switch { .. } => self.p_switch(iid, bid, inst, None)?,
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
        }
        Ok(false)
    }

    fn p_binop(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::BinaryOp { lhs, binop, rhs } = inst else {
            panic!()
        };
        let lhs = self.p_operand(lhs)?;
        let rhs = self.p_operand(rhs)?;
        let inst = match binop {
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
    ) -> Result<(), CompilationError> {
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
            return Ok(());
        }

        let func = self.am.func(*callee);
        // Ignore calls the software tracer makes to record blocks.
        #[cfg(tracer_swt)]
        if func.name() == "__yk_trace_basicblock" {
            return Ok(());
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
                return self.outline_until(bid);
            }
            self.promotions_off += usize::try_from(self.am.type_(fty.ret_ty()).bytew()).unwrap();
        }

        if !func.is_declaration()
            && !func.is_outline()
            // FIXME: We currently don't handle va_start
            && !func.contains_call_to(self.am, "llvm.va_start")
            // Is this a recursive call?
            && !self.frames.iter().any(|f| f.func == *callee)
        {
            // Inlinable call.
            self.frames.last_mut().unwrap().call_safepoint = Some(safepoint.as_ref().unwrap());
            self.frames.push(Frame {
                call_iid: Some(iid),
                func: *callee,
                call_safepoint: None,
                args: jargs,
                locals: HashMap::new(),
            });
        } else {
            // Non-inlinable call. These come in two distinct flavours:
            //   1. LLVM intrinsics. We handle each of these individually.
            //   2. User-level calls. We emit a call instruction then skip any blocks we encounter
            //      (which could be zero, or many, and may include recursive calls) until that
            //      function returns.

            let ftyidx = self.p_ty(self.am.type_(func.tyidx()))?;

            // Handle LLVM intrinsics.
            if func.name().starts_with("llvm.") {
                return self.p_llvm_intrinsic(iid, ftyidx, func.name(), jargs);
            }

            // Handle user-level functions.
            let addr = self.dlsym(func.name()).unwrap().addr();
            self.addr_name_map
                .as_mut()
                .map(|x| x.insert(addr, func.name().to_owned()));
            let tyidx = self.opt.push_ty(hir::Ty::Ptr(0))?;
            let tgt_iidx = self.const_to_iidx(tyidx, hir::ConstKind::Ptr(addr))?;
            self.push_inst_and_link_local(
                iid,
                hir::Call {
                    tgt: tgt_iidx,
                    func_tyidx: ftyidx,
                    args: jargs,
                }
                .into(),
            )?;
            self.outline_until(bid)?;
        }
        Ok(())
    }

    fn p_icall(
        &mut self,
        iid: InstId,
        bid: BBlockId,
        inst: &'static Inst,
    ) -> Result<(), CompilationError> {
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
        self.push_inst_and_link_local(
            iid,
            hir::Call {
                tgt: tgt_iidx,
                func_tyidx: ftyidx,
                args: jargs,
            }
            .into(),
        )?;
        self.outline_until(bid)?;

        Ok(())
    }

    /// Outline until the successor block to `bid` is encountered. Returns `Err` if irregular
    /// control flow is detected.
    fn outline_until(&mut self, cur_bid: BBlockId) -> Result<(), CompilationError> {
        // Now we skip over all the blocks in this call.
        let next_bid = match self.am.bblock(&cur_bid).insts().last().unwrap() {
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
        let mut recurse = 0;
        loop {
            let ta = {
                let Some(Ok(ta)) = self.ta_iter.peek() else {
                    panic!()
                };
                ta.to_owned()
            };
            let cnd_bid = self.ta_to_bid(&ta).unwrap();
            if cnd_bid.funcidx() == cur_bid.funcidx() {
                if cnd_bid.is_entry() {
                    recurse += 1;
                } else if self.am.bblock(&cnd_bid).is_return() {
                    assert!(recurse > 0);
                    recurse -= 1;
                }

                if recurse == 0 && cnd_bid == next_bid {
                    break;
                }
            }

            for inst in self.am.bblock(&cnd_bid).insts() {
                match inst {
                    Inst::Call { callee, .. } => {
                        let func = self.am.func(*callee);
                        if func.is_idempotent() {
                            todo!();
                        }
                    }
                    Inst::Promote { tyidx, .. } => {
                        self.promotions_off +=
                            usize::try_from(self.am.type_(*tyidx).bytew()).unwrap();
                    }
                    _ => (),
                }
            }

            self.ta_iter.next();
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
                let [src, is_int_min_poison]: [hir::InstIdx; 2] =
                    jargs.into_vec().try_into().unwrap();
                let is_int_min_poison = if let hir::Inst::Const(hir::Const {
                    kind: hir::ConstKind::Int(x),
                    ..
                }) = &self.opt.inst(is_int_min_poison)
                {
                    x.to_zero_ext_u8().unwrap() != 0
                } else {
                    panic!()
                };
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::Abs {
                    tyidx: fty.rtn_tyidx,
                    val: src,
                    is_int_min_poison,
                };
                self.push_inst_and_link_local(iid, hinst.into()).map(|_| ())
            }
            "ctpop" => {
                let [src]: [hir::InstIdx; 1] = jargs.into_vec().try_into().unwrap();
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::CtPop {
                    tyidx: fty.rtn_tyidx,
                    val: src,
                };
                self.push_inst_and_link_local(iid, hinst.into()).map(|_| ())
            }
            "lifetime" => Ok(()),
            "memcpy" => {
                let [dst, src, len, volatile]: [hir::InstIdx; 4] =
                    jargs.into_vec().try_into().unwrap();
                let volatile = if let hir::Inst::Const(hir::Const {
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
                    volatile,
                };
                self.push_inst_and_link_local(iid, hinst.into()).map(|_| ())
            }
            "smax" => {
                let [lhs, rhs]: [hir::InstIdx; 2] = jargs.into_vec().try_into().unwrap();
                let fty = self.opt.func_ty(ftyidx);
                let hinst = hir::SMax {
                    tyidx: fty.rtn_tyidx,
                    lhs,
                    rhs,
                };
                self.push_inst_and_link_local(iid, hinst.into()).map(|_| ())
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
        let hinst = match cast_kind {
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
            CastKind::BitCast => todo!(),
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

        let next_bb = self.peek_next_bbid().unwrap();
        assert_eq!(
            next_bb.funcidx(),
            iid.funcidx(),
            "Control flow has diverged"
        );
        assert!(next_bb.bbidx() == *true_bb || next_bb.bbidx() == *false_bb);

        let cond_iidx = self.p_operand(cond)?;
        self.push_guard(
            bid,
            iid,
            next_bb.bbidx() == *true_bb,
            cond_iidx,
            safepoint,
            None,
        )
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
        self.push_inst_and_link_local(iid, hir::FCmp { pred, lhs, rhs }.into())
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
            }
            .into(),
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
            }
            .into(),
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

        let icmp = self.opt.push_inst(
            hir::ICmp {
                pred: hir::IPred::Eq,
                lhs: val_iidx,
                rhs: const_iidx,
                samesign: false,
            }
            .into(),
        )?;
        self.push_guard(bid, iid.clone(), true, icmp, safepoint, None)
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
            ptr = self.opt.push_inst(
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
        // self.push_inst_and_link_local(iid, hir::Load::new(ty, ptr, *volatile).into())
        //     .map(|_| ())

        for (num_elems, elem_size) in dyn_elem_counts.iter().zip(dyn_elem_sizes) {
            let num_elems = self.p_operand(num_elems)?;
            // If the element count is not the same width as LLVM's GEP index type, then we have to
            // sign extend it up (or truncate it down) to the right size. We've not yet
            // seen this in the wild.
            // if num_elems.byte_size(&self.jit_mod) * 8 != usize::from(self.aot_mod.ptr_off_bitsize())
            // {
            //     todo!();
            // }
            let elem_size = u32::try_from(*elem_size).map_err(|_| {
                CompilationError::LimitExceeded("PtrAdd elem_size doesn't fit in u32".into())
            })?;
            ptr = self.opt.push_inst(
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

    /// Return `true` if this is an early return that should stop examining the trace further.
    fn p_ret(&mut self, _iid: InstId, inst: &Inst) -> Result<bool, CompilationError> {
        let Inst::Ret { val } = inst else { panic!() };

        let val = match val {
            Some(x) => Some(self.p_operand(x)?),
            None => None,
        };

        let frame = self.frames.pop().unwrap();
        if !self.frames.is_empty() {
            let last_frame = self.frames.last_mut().unwrap();
            last_frame.call_safepoint = None;
            if let Some(val_iidx) = val {
                last_frame.set_local(frame.call_iid.unwrap(), val_iidx);
            }
            Ok(false)
        } else {
            if let BuildKind::Side { .. } = self.bkind {
                todo!();
            }
            // We've returned out of the function that started tracing. Stop processing any
            // remaining blocks and emit a return instruction that naturally returns from a
            // compiled trace into the interpreter.
            let safepoint = frame.call_safepoint.unwrap();
            // We currently don't support passing values back during early returns.
            assert!(val.is_none());
            self.opt.push_inst(hir::Return { safepoint }.into())?;
            Ok(true)
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
            }
            .into(),
        )
        .map(|_| ())
    }

    fn p_store(&mut self, iid: InstId, inst: &Inst) -> Result<(), CompilationError> {
        let Inst::Store { val, tgt, volatile } = inst else {
            panic!()
        };
        let ptr = self.p_operand(tgt)?;
        let val = self.p_operand(val)?;
        self.push_inst_and_link_local(
            iid,
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
                self.opt.push_inst(
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
                let next_icmp = self.opt.push_inst(
                    hir::ICmp {
                        pred: hir::IPred::Eq,
                        lhs: val_iidx,
                        rhs: const_iidx,
                        samesign: false,
                    }
                    .into(),
                )?;
                icmp = self.opt.push_inst(
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
    #[allow(unused)]
    Coupler,
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
    #[allow(unused)]
    Coupler,
    Loop {
        entry_safepoint: &'static DeoptSafepoint,
    },
    Side {
        prev_bid: BBlockId,
        entry_vlocs: Vec<VarLocs<Reg>>,
        src_ctr: Arc<J2CompiledTrace<Reg>>,
        src_gridx: hir::GuardRestoreIdx,
        tgt_ctr: Arc<J2CompiledTrace<Reg>>,
    },
}

/// An inlined frame.
#[derive(Debug)]
struct Frame {
    /// If this is an inlined frame (i.e. for all but the bottom frame), this is the [InstId] of
    /// the call instruction. This is used to link the return value when the inlined frame is
    /// popped.
    call_iid: Option<InstId>,
    func: FuncIdx,
    /// This frame's arguments. This is not mutated after frame creation.
    args: SmallVec<[hir::InstIdx; 1]>,
    /// The current safepoint for this frame. This has no initial value at frame entry, and is
    /// updated at every call site.
    call_safepoint: Option<&'static DeoptSafepoint>,
    locals: HashMap<InstId, hir::InstIdx>,
}

impl Frame {
    /// Lookup the AOT variable `iid` relative to `opt`.
    fn get_local(&self, opt: &dyn OptT, iid: &InstId) -> hir::InstIdx {
        opt.map_iidx(self.locals[iid])
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
