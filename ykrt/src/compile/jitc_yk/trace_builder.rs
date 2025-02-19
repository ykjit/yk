//! The JIT IR trace builder.
//!
//! This takes in an (AOT IR, execution trace) pair and constructs a JIT IR trace from it.

use super::aot_ir::{self, BBlockId, BinOp, Module};
use super::YkSideTraceInfo;
use super::{
    arbbitint::ArbBitInt,
    jit_ir::{self, Const, Operand, PackedOperand, ParamIdx, TraceKind},
    AOT_MOD,
};
use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::CompilationError,
    mt::CompiledTraceId,
    trace::{AOTTraceIterator, AOTTraceIteratorError, TraceAction},
};
use std::{collections::HashMap, ffi::CString, marker::PhantomData, sync::Arc};

/// Given an execution trace and AOT IR, creates a JIT IR trace.
pub(crate) struct TraceBuilder<Register: Send + Sync> {
    /// The AOT IR.
    aot_mod: &'static Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    /// Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstID, jit_ir::Operand>,
    /// BBlock containing the current control point (i.e. the control point that started this trace).
    cp_block: Option<aot_ir::BBlockId>,
    /// Index of the first [ParameterInst].
    first_paraminst_idx: usize,
    /// Inlined calls.
    ///
    /// For a valid trace, this always contains at least one element, otherwise the trace returned
    /// out of the function that started tracing, which is problematic.
    frames: Vec<InlinedFrame>,
    /// The block at which to stop outlining.
    outline_target_blk: Option<BBlockId>,
    /// Current count of recursive calls to the function in which outlining was started. Will be 0
    /// if `outline_target_blk` is None.
    recursion_count: usize,
    /// Values promoted to trace-level constants. Values are stored as native-endian sequences of
    /// bytes: the AOT code must be examined to determine the size of a given a value at a given
    /// point. Currently (and probably forever) only values that are a multiple of 8 bits are
    /// supported.
    promotions: Box<[u8]>,
    /// The trace's current position in the promotions array.
    promote_idx: usize,
    phantom: PhantomData<Register>,
    /// The dynamically recorded debug strings, in the order that the corresponding
    /// `yk_debug_str()` calls were encountered in the trace.
    debug_strs: Vec<String>,
    /// The trace's current position in the [Self::debug_strs] vector.
    debug_str_idx: usize,
}

impl<Register: Send + Sync + 'static> TraceBuilder<Register> {
    /// Create a trace builder.
    ///
    /// Arguments:
    ///  - `aot_mod`: The AOT IR module that the trace flows through.
    ///  - `mtrace`: The mapped trace.
    ///  - `promotions`: Values promoted to constants during runtime.
    ///  - `debug_archors`: Debug strs recorded during runtime.
    fn new(
        tracekind: TraceKind,
        ctr_id: CompiledTraceId,
        aot_mod: &'static Module,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
    ) -> Result<Self, CompilationError> {
        Ok(Self {
            aot_mod,
            jit_mod: jit_ir::Module::new(tracekind, ctr_id, aot_mod.global_decls_len())?,
            local_map: HashMap::new(),
            cp_block: None,
            first_paraminst_idx: 0,
            // We have to insert a placeholder frame to represent the place we started tracing, as
            // we don't know where that is yet. We'll update it as soon as we do.
            frames: vec![InlinedFrame {
                funcidx: None,
                callinst: None,
                safepoint: None,
                args: Vec::new(),
            }],
            outline_target_blk: None,
            recursion_count: 0,
            promotions,
            promote_idx: 0,
            phantom: PhantomData,
            debug_strs,
            debug_str_idx: 0,
        })
    }

    // Given a mapped block, find the AOT block ID, or return `None` if it is unmapped.
    fn lookup_aot_block(&self, tb: &TraceAction) -> Option<aot_ir::BBlockId> {
        match tb {
            TraceAction::MappedAOTBBlock { func_name, bb } => {
                let func_name = func_name.to_str().unwrap(); // safe: func names are valid UTF-8.
                let func = self.aot_mod.funcidx(func_name);
                Some(aot_ir::BBlockId::new(func, aot_ir::BBlockIdx::new(*bb)))
            }
            TraceAction::UnmappableBBlock => None,
            TraceAction::Promotion => todo!(),
        }
    }

    /// Create the prolog of the trace.
    fn create_trace_header(
        &mut self,
        blk: &'static aot_ir::BBlock,
    ) -> Result<(), CompilationError> {
        // Find the control point call to retrieve the live variables from its safepoint.
        //
        // FIXME: Stash the location at IR lowering time, instead of searching at runtime.
        let mut safepoint = None;
        let mut inst_iter = blk.insts.iter().enumerate().rev();
        for (_, inst) in inst_iter.by_ref() {
            // Is it a call to the control point, then insert loads for the trace inputs. These
            // directly reference registers or stack slots in the parent frame and thus don't
            // necessarily result in machine code during codegen.
            if inst.is_control_point(self.aot_mod) {
                safepoint = Some(inst.safepoint().unwrap());
                break;
            }
        }
        // If we don't find a safepoint here something has gone wrong with the AOT IR.
        let safepoint = safepoint.unwrap();
        let (rec, _) = AOT_STACKMAPS
            .as_ref()
            .unwrap()
            .get(usize::try_from(safepoint.id).unwrap());

        debug_assert!(safepoint.lives.len() == rec.live_vars.len());
        for idx in 0..safepoint.lives.len() {
            let aot_op = &safepoint.lives[idx];
            let input_tyidx = self.handle_type(aot_op.type_(self.aot_mod))?;

            // Get the location for this input variable.
            let var = &rec.live_vars[idx];
            if var.len() > 1 {
                todo!("Deal with multi register locations");
            }
            let param_inst = jit_ir::ParamInst::new(ParamIdx::try_from(idx)?, input_tyidx).into();
            self.jit_mod.push(param_inst)?;
            self.jit_mod.push_param(var.get(0).unwrap().clone());
            self.local_map.insert(
                aot_op.to_inst_id(),
                jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
            );
            self.jit_mod
                .push_body_start_var(jit_ir::Operand::Var(self.jit_mod.last_inst_idx()));
        }
        self.jit_mod.push(jit_ir::Inst::TraceHeaderStart)?;
        Ok(())
    }

    /// Process (skip) promotions and debug strings inside an otherwise outlined block.
    fn process_promotions_and_debug_strs_only(
        &mut self,
        bid: &aot_ir::BBlockId,
    ) -> Result<(), CompilationError> {
        let blk = self.aot_mod.bblock(bid);

        for inst in blk.insts.iter() {
            match inst {
                aot_ir::Inst::Promote {
                    val: aot_ir::Operand::LocalVariable(_),
                    tyidx,
                    ..
                } => {
                    // Consume the correct number of bytes from the promoted values array.
                    let width_bits = match self.aot_mod.type_(*tyidx) {
                        aot_ir::Ty::Integer(x) => x.bitw(),
                        _ => unreachable!(),
                    };
                    let width_bytes = usize::try_from(width_bits.div_ceil(8)).unwrap();
                    self.promote_idx += width_bytes;
                }
                aot_ir::Inst::DebugStr { .. } => {
                    // Skip this debug string.
                    self.debug_str_idx += 1;
                }
                _ => (),
            }
        }
        Ok(())
    }

    /// Walk over a traced AOT block, translating the constituent instructions into the JIT module.
    fn process_block(
        &mut self,
        bid: &aot_ir::BBlockId,
        prevbb: &Option<aot_ir::BBlockId>,
        nextbb: Option<aot_ir::BBlockId>,
    ) -> Result<(), CompilationError> {
        // We have already checked this (and aborted building the trace) at the time we encountered
        // an `Inst::Ret` that would make the frame stack empty.
        assert!(!self.frames.is_empty());

        // unwrap safe: can't trace a block not in the AOT module.
        let blk = self.aot_mod.bblock(bid);

        // Decide how to translate each AOT instruction.
        for (iidx, inst) in blk.insts.iter().enumerate() {
            match inst {
                aot_ir::Inst::Br { .. } => Ok(()),
                aot_ir::Inst::Load {
                    ptr,
                    tyidx,
                    volatile,
                } => self.handle_load(bid, iidx, ptr, tyidx, *volatile),
                // FIXME: ignore remaining instructions after a call.
                aot_ir::Inst::Call { callee, args, .. } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_call(inst, bid, iidx, callee, args, nextinst)
                }
                aot_ir::Inst::IndirectCall {
                    ftyidx,
                    callop,
                    args,
                } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_indirectcall(inst, bid, iidx, ftyidx, callop, args, nextinst)
                }
                aot_ir::Inst::Store { tgt, val, volatile } => {
                    self.handle_store(bid, iidx, tgt, val, *volatile)
                }
                aot_ir::Inst::PtrAdd {
                    ptr,
                    const_off,
                    dyn_elem_counts,
                    dyn_elem_sizes,
                    ..
                } => {
                    if self.cp_block.as_ref() == Some(bid) && iidx == self.first_paraminst_idx {
                        // We've reached the trace inputs part of the control point block. There's
                        // no point in copying these instructions over and we can just skip to the
                        // next block.
                        return Ok(());
                    }
                    self.handle_ptradd(bid, iidx, ptr, *const_off, dyn_elem_counts, dyn_elem_sizes)
                }
                aot_ir::Inst::BinaryOp { lhs, binop, rhs } => {
                    self.handle_binop(bid, iidx, *binop, lhs, rhs)
                }
                aot_ir::Inst::ICmp { lhs, pred, rhs, .. } => {
                    self.handle_icmp(bid, iidx, lhs, pred, rhs)
                }
                aot_ir::Inst::CondBr {
                    cond,
                    true_bb,
                    safepoint,
                    ..
                } => self.handle_condbr(bid, safepoint, nextbb.as_ref().unwrap(), cond, true_bb),
                aot_ir::Inst::Cast {
                    cast_kind,
                    val,
                    dest_tyidx,
                } => self.handle_cast(bid, iidx, cast_kind, val, dest_tyidx),
                aot_ir::Inst::Ret { val } => self.handle_ret(bid, iidx, val),
                aot_ir::Inst::Switch {
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                    safepoint,
                } => self.handle_switch(
                    bid,
                    iidx,
                    safepoint,
                    nextbb.as_ref().unwrap(),
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                ),
                aot_ir::Inst::Phi {
                    incoming_bbs,
                    incoming_vals,
                    ..
                } => {
                    debug_assert_eq!(prevbb.as_ref().unwrap().funcidx(), bid.funcidx());
                    self.handle_phi(
                        bid,
                        iidx,
                        &prevbb.as_ref().unwrap().bbidx(),
                        incoming_bbs,
                        incoming_vals,
                    )
                }
                aot_ir::Inst::Select {
                    cond,
                    trueval,
                    falseval,
                } => self.handle_select(bid, iidx, cond, trueval, falseval),
                aot_ir::Inst::LoadArg { arg_idx, .. } => {
                    // Map passed in arguments to their respective LoadArg instructions.
                    let jitop = &self.frames.last().unwrap().args[*arg_idx];
                    let aot_iid =
                        aot_ir::InstID::new(bid.funcidx(), bid.bbidx(), aot_ir::InstIdx::new(iidx));
                    self.local_map.insert(aot_iid, jitop.unpack(&self.jit_mod));
                    Ok(())
                }
                aot_ir::Inst::FCmp { lhs, pred, rhs, .. } => {
                    self.handle_fcmp(bid, iidx, lhs, pred, rhs)
                }
                aot_ir::Inst::Promote {
                    val,
                    tyidx,
                    safepoint,
                } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_promote(bid, iidx, val, safepoint, tyidx, nextinst)
                }
                aot_ir::Inst::FNeg { val } => self.handle_fneg(bid, iidx, val),
                aot_ir::Inst::DebugStr { .. } => {
                    let nextinst = blk.insts.last().unwrap();
                    self.handle_debug_str(bid, iidx, nextinst)
                }
                _ => todo!("{:?}", inst),
            }?;
        }
        Ok(())
    }

    /// Link the AOT IR to the last instruction pushed into the JIT IR.
    ///
    /// This must be called after adding a JIT IR instruction which has a return value.
    fn link_iid_to_last_inst(&mut self, bid: &aot_ir::BBlockId, aot_inst_idx: usize) {
        let aot_iid = aot_ir::InstID::new(
            bid.funcidx(),
            bid.bbidx(),
            aot_ir::InstIdx::new(aot_inst_idx),
        );
        // The unwrap is safe because we've already inserted an element at this index and proven
        // that the index is in bounds.
        self.local_map
            .insert(aot_iid, jit_ir::Operand::Var(self.jit_mod.last_inst_idx()));
    }

    fn copy_inst(
        &mut self,
        jit_inst: jit_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        if jit_inst.def_type(&self.jit_mod).is_some() {
            // If the AOT instruction defines a new value, then add it to the local map.
            self.jit_mod.push(jit_inst)?;
            self.link_iid_to_last_inst(bid, aot_inst_idx);
        } else {
            self.jit_mod.push(jit_inst)?;
        }
        Ok(())
    }

    /// Translate a global variable use.
    fn handle_global(
        &mut self,
        idx: aot_ir::GlobalDeclIdx,
    ) -> Result<jit_ir::GlobalDeclIdx, CompilationError> {
        let aot_global = self.aot_mod.global_decl(idx);
        let jit_global = jit_ir::GlobalDecl::new(
            CString::new(aot_global.name()).unwrap(),
            aot_global.is_threadlocal(),
            idx,
        );
        self.jit_mod.insert_global_decl(jit_global)
    }

    /// Translate a constant value.
    fn handle_const(
        &mut self,
        aot_const: &aot_ir::Const,
    ) -> Result<jit_ir::Const, CompilationError> {
        let aot_const = aot_const.unwrap_val();
        let bytes = aot_const.bytes();
        match self.aot_mod.type_(aot_const.tyidx()) {
            aot_ir::Ty::Integer(x) => {
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
                        u64::from(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    }
                    64 => {
                        debug_assert_eq!(bytes.len(), 8);
                        u64::from_ne_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    _ => todo!("{}", x.bitw()),
                };
                let jit_tyidx = self.jit_mod.insert_ty(jit_ir::Ty::Integer(x.bitw()))?;
                Ok(jit_ir::Const::Int(
                    jit_tyidx,
                    ArbBitInt::from_u64(x.bitw(), v),
                ))
            }
            aot_ir::Ty::Float(fty) => {
                let jit_tyidx = self.jit_mod.insert_ty(jit_ir::Ty::Float(fty.clone()))?;
                // unwrap cannot fail if the AOT IR is valid.
                let val = f64::from_ne_bytes(bytes[0..8].try_into().unwrap());
                Ok(jit_ir::Const::Float(jit_tyidx, val))
            }
            aot_ir::Ty::Ptr => {
                let val: usize;
                #[cfg(target_arch = "x86_64")]
                {
                    debug_assert_eq!(bytes.len(), 8);
                    val = usize::from_ne_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]);
                }
                Ok(jit_ir::Const::Ptr(val))
            }
            x => todo!("{x:?}"),
        }
    }

    /// Translate an operand.
    fn handle_operand(
        &mut self,
        op: &aot_ir::Operand,
    ) -> Result<jit_ir::Operand, CompilationError> {
        match op {
            aot_ir::Operand::LocalVariable(iid) => Ok(self.local_map[iid].clone()),
            aot_ir::Operand::Const(cidx) => {
                let jit_const = self.handle_const(self.aot_mod.const_(*cidx))?;
                Ok(jit_ir::Operand::Const(
                    self.jit_mod.insert_const(jit_const)?,
                ))
            }
            aot_ir::Operand::Global(gd_idx) => {
                let load = jit_ir::LookupGlobalInst::new(self.handle_global(*gd_idx)?)?;
                self.jit_mod.push_and_make_operand(load.into())
            }
            _ => todo!("{}", op.display(self.aot_mod)),
        }
    }

    /// Translate a type.
    fn handle_type(&mut self, aot_type: &aot_ir::Ty) -> Result<jit_ir::TyIdx, CompilationError> {
        let jit_ty = match aot_type {
            aot_ir::Ty::Void => jit_ir::Ty::Void,
            aot_ir::Ty::Integer(x) => jit_ir::Ty::Integer(x.bitw()),
            aot_ir::Ty::Ptr => jit_ir::Ty::Ptr,
            aot_ir::Ty::Func(ft) => {
                let mut jit_args = Vec::new();
                for aot_arg_tyidx in ft.arg_tyidxs() {
                    let jit_ty = self.handle_type(self.aot_mod.type_(*aot_arg_tyidx))?;
                    jit_args.push(jit_ty);
                }
                let jit_retty = self.handle_type(self.aot_mod.type_(ft.ret_ty()))?;
                jit_ir::Ty::Func(jit_ir::FuncTy::new(jit_args, jit_retty, ft.is_vararg()))
            }
            aot_ir::Ty::Struct(_st) => todo!(),
            aot_ir::Ty::Float(ft) => {
                let inner = match ft {
                    aot_ir::FloatTy::Float => jit_ir::FloatTy::Float,
                    aot_ir::FloatTy::Double => jit_ir::FloatTy::Double,
                };
                jit_ir::Ty::Float(inner)
            }
            aot_ir::Ty::Unimplemented(s) => jit_ir::Ty::Unimplemented(s.to_owned()),
        };
        self.jit_mod.insert_ty(jit_ty)
    }

    /// Translate a function.
    fn handle_func(
        &mut self,
        aot_idx: aot_ir::FuncIdx,
    ) -> Result<jit_ir::FuncDeclIdx, CompilationError> {
        let aot_func = self.aot_mod.func(aot_idx);
        let jit_func = jit_ir::FuncDecl::new(
            aot_func.name().to_owned(),
            self.handle_type(self.aot_mod.type_(aot_func.tyidx()))?,
        );
        self.jit_mod.insert_func_decl(jit_func)
    }

    /// Translate binary operations such as add, sub, mul, etc.
    fn handle_binop(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        binop: aot_ir::BinOp,
        lhs: &aot_ir::Operand,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let lhs = self.handle_operand(lhs)?;
        let rhs = self.handle_operand(rhs)?;
        let inst = jit_ir::BinOpInst::new(lhs, binop, rhs).into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    /// Create a guard.
    ///
    /// The guard fails if `cond` is not `expect`.
    fn create_guard(
        &mut self,
        bid: &aot_ir::BBlockId,
        cond: &jit_ir::Operand,
        expect: bool,
        safepoint: &'static aot_ir::DeoptSafepoint,
    ) -> Result<jit_ir::GuardInst, CompilationError> {
        // Assign this branch's stackmap to the current frame.
        self.frames.last_mut().unwrap().safepoint = Some(safepoint);

        // Collect the safepoint IDs and live variables from this conditional branch and the
        // previous frames to store inside the guard.
        // Unwrap-safe as each frame at this point must have a safepoint associated with it.
        let mut live_vars = Vec::new(); // (AOT var, JIT var) pairs
        let mut callframes = Vec::new();
        for frame in &self.frames {
            let safepoint = frame.safepoint.unwrap();
            // All the `unwrap`s are safe as we've filled in the necessary information during trace
            // building.
            callframes.push(jit_ir::InlinedFrame::new(
                frame.callinst.clone(),
                frame.funcidx.unwrap(),
                frame.safepoint.unwrap(),
                // We don't need to copy the arguments since they are only required for LoadArg
                // instructions which we won't see at the beginning of a sidetrace.
                Vec::new(),
            ));

            // Collect live variables.
            for op in safepoint.lives.iter() {
                match op {
                    aot_ir::Operand::LocalVariable(iid) => {
                        match self.local_map[iid] {
                            jit_ir::Operand::Var(liidx) => {
                                // If, as often happens, a guard has in its live set the boolean
                                // variable used as the condition, we can convert this into a
                                // constant. For example if we have e.g.:
                                //
                                // ```
                                // %10: i1 = ...
                                // guard true, %10 [...: %10]
                                // ```
                                //
                                // then if the guard fails we know %10 was false and the guard is
                                // therefore equivalent to:
                                //
                                // ```
                                // %10: i1 = ...
                                // guard true, %10 [...: 0i1]
                                // ```
                                //
                                // Rewriting it in this form makes code further down the chain
                                // simpler, because it means it doesn't have to be clever when
                                // analysing a guard's live variables.
                                match cond {
                                    Operand::Var(cond_idx) if *cond_idx == liidx => {
                                        let cidx = if expect {
                                            self.jit_mod.false_constidx()
                                        } else {
                                            self.jit_mod.true_constidx()
                                        };
                                        live_vars.push((
                                            iid.clone(),
                                            PackedOperand::new(&jit_ir::Operand::Const(cidx)),
                                        ));
                                    }
                                    _ => {
                                        live_vars.push((
                                            iid.clone(),
                                            PackedOperand::new(&jit_ir::Operand::Var(liidx)),
                                        ));
                                    }
                                }
                            }
                            jit_ir::Operand::Const(cidx) => {
                                live_vars.push((
                                    iid.clone(),
                                    PackedOperand::new(&jit_ir::Operand::Const(cidx)),
                                ));
                            }
                        }
                    }
                    _ => panic!(), // IR malformed.
                }
            }
        }

        let gi = jit_ir::GuardInfo::new(bid.clone(), live_vars, callframes, safepoint.id);
        let gi_idx = self.jit_mod.push_guardinfo(gi).unwrap();

        Ok(jit_ir::GuardInst::new(cond.clone(), expect, gi_idx))
    }

    /// Translate a conditional `Br` instruction.
    fn handle_condbr(
        &mut self,
        bid: &aot_ir::BBlockId,
        safepoint: &'static aot_ir::DeoptSafepoint,
        next_bb: &aot_ir::BBlockId,
        cond: &aot_ir::Operand,
        true_bb: &aot_ir::BBlockIdx,
    ) -> Result<(), CompilationError> {
        let jit_cond = self.handle_operand(cond)?;
        let guard = self.create_guard(bid, &jit_cond, *true_bb == next_bb.bbidx(), safepoint)?;
        self.jit_mod.push(guard.into())?;
        Ok(())
    }

    fn handle_ret(
        &mut self,
        _bid: &aot_ir::BBlockId,
        _aot_inst_idx: usize,
        val: &Option<aot_ir::Operand>,
    ) -> Result<(), CompilationError> {
        // If this `unwrap` fails, it means that early return detection in `mt.rs` is not working
        // as expected.
        let frame = self.frames.pop().unwrap();
        if !self.frames.is_empty() {
            if let Some(val) = val {
                let op = self.handle_operand(val)?;
                self.local_map.insert(frame.callinst.unwrap(), op);
            }
            Ok(())
        } else {
            // We've returned out of the function that started tracing, which isn't allowed.
            Err(CompilationError::General(
                "returned from function that started tracing".into(),
            ))
        }
    }

    /// Translate a `ICmp` instruction.
    fn handle_icmp(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        lhs: &aot_ir::Operand,
        pred: &aot_ir::Predicate,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst =
            jit_ir::ICmpInst::new(self.handle_operand(lhs)?, *pred, self.handle_operand(rhs)?)
                .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_fcmp(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        lhs: &aot_ir::Operand,
        pred: &aot_ir::FloatPredicate,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst =
            jit_ir::FCmpInst::new(self.handle_operand(lhs)?, *pred, self.handle_operand(rhs)?)
                .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    /// Translate a `Load` instruction.
    fn handle_load(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ptr: &aot_ir::Operand,
        tyidx: &aot_ir::TyIdx,
        volatile: bool,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::LoadInst::new(
            self.handle_operand(ptr)?,
            self.handle_type(self.aot_mod.type_(*tyidx))?,
            volatile,
        )
        .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_indirectcall(
        &mut self,
        inst: &'static aot_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ftyidx: &aot_ir::TyIdx,
        callop: &aot_ir::Operand,
        args: &[aot_ir::Operand],
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        debug_assert!(!inst.is_debug_call(self.aot_mod));

        // Convert AOT args to JIT args.
        let mut jit_args = Vec::new();
        for arg in args {
            jit_args.push(self.handle_operand(arg)?);
        }

        // While we can work out if this is a mappable call or not by loooking at the next block,
        // this unfortunately doesn't tell us whether the call has been annotated with
        // "yk_outline". For now we have to be conservative here and outline all indirect calls.
        // One solution would be to stop including the AOT IR for functions annotated with
        // "yk_outline". Any mappable, indirect call is then guaranteed to be inline safe.
        self.outline_until_after_call(bid, nextinst);

        let jit_callop = self.handle_operand(callop)?;
        let jit_tyidx = self.handle_type(self.aot_mod.type_(*ftyidx))?;
        let inst =
            jit_ir::IndirectCallInst::new(&mut self.jit_mod, jit_tyidx, jit_callop, jit_args)?;
        let idx = self.jit_mod.push_indirect_call(inst)?;
        self.copy_inst(jit_ir::Inst::IndirectCall(idx), bid, aot_inst_idx)
    }

    fn handle_call(
        &mut self,
        inst: &'static aot_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        callee: &aot_ir::FuncIdx,
        args: &[aot_ir::Operand],
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        // Ignore special functions that we neither want to inline nor copy.
        if inst.is_debug_call(self.aot_mod) {
            return Ok(());
        }
        // Ignore software tracer calls. Software tracer inserts
        // `__yk_trace_basicblock` instruction calls into the beginning of
        // every basic block. These calls can be ignored as they are
        // only used to collect runtime information for the tracer itself.
        if AOT_MOD.func(*callee).name() == "__yk_trace_basicblock" {
            return Ok(());
        }

        if inst.is_control_point(self.aot_mod) {
            return Ok(());
        }

        // Convert AOT args to JIT args.
        let mut jit_args = Vec::new();
        for arg in args {
            match self.handle_operand(arg)? {
                jit_ir::Operand::Const(c) => {
                    // We don't want to do constant propagation here as it makes our life harder
                    // creating guards. Instead we simply create a `Const` instruction here and
                    // reference that.
                    let inst = jit_ir::Inst::Const(c);
                    self.jit_mod.push(inst)?;
                    let op = jit_ir::Operand::Var(self.jit_mod.last_inst_idx());
                    jit_args.push(op);
                }
                op => jit_args.push(op),
            }
        }

        // Check if this is a recursive call by scanning the call stack for the callee.
        let is_recursive = self.frames.iter().any(|f| f.funcidx == Some(*callee));

        if inst.is_mappable_call(self.aot_mod)
            && !self.aot_mod.func(*callee).is_outline()
            && !is_recursive
        {
            // This is a mappable call that we want to inline.
            debug_assert!(inst.safepoint().is_some());
            // Assign safepoint to the current frame.
            // Unwrap is safe as there's always at least one frame.
            self.frames.last_mut().unwrap().safepoint = inst.safepoint();
            // Create a new frame for the inlined call and pass in the arguments of the caller.
            let aot_iid = aot_ir::InstID::new(
                bid.funcidx(),
                bid.bbidx(),
                aot_ir::InstIdx::new(aot_inst_idx),
            );
            self.frames.push(InlinedFrame {
                funcidx: Some(*callee),
                callinst: Some(aot_iid),
                safepoint: None,
                args: jit_args.iter().map(PackedOperand::new).collect::<Vec<_>>(),
            });
            Ok(())
        } else {
            // This call can't be inlined. It is either unmappable (a declaration or an indirect
            // call) or the compiler annotated it with `yk_outline`.
            self.outline_until_after_call(bid, nextinst);
            let jit_func_decl_idx = self.handle_func(*callee)?;
            let inst =
                jit_ir::DirectCallInst::new(&mut self.jit_mod, jit_func_decl_idx, jit_args)?.into();
            self.copy_inst(inst, bid, aot_inst_idx)
        }
    }

    fn handle_store(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        tgt: &aot_ir::Operand,
        val: &aot_ir::Operand,
        volatile: bool,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::StoreInst::new(
            self.handle_operand(tgt)?,
            self.handle_operand(val)?,
            volatile,
        )
        .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_ptradd(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ptr: &aot_ir::Operand,
        const_off: isize,
        dyn_elem_counts: &[aot_ir::Operand],
        dyn_elem_sizes: &[usize],
    ) -> Result<(), CompilationError> {
        // First apply the constant offset.
        let mut jit_ptr = self.handle_operand(ptr)?;
        if const_off != 0 {
            let off_i32 = i32::try_from(const_off).map_err(|_| {
                CompilationError::LimitExceeded("ptradd constant offset doesn't fit in i32".into())
            })?;
            jit_ptr = self
                .jit_mod
                .push_and_make_operand(jit_ir::PtrAddInst::new(jit_ptr, off_i32).into())?;
        } else {
            jit_ptr = match jit_ptr {
                Operand::Var(iidx) => self
                    .jit_mod
                    .push_and_make_operand(jit_ir::Inst::Copy(iidx))?,
                _ => todo!(),
            }
        }

        // Now apply any dynamic indices.
        //
        // Each offset is the number of elements multiplied by the byte size of an element.
        for (num_elems, elem_size) in dyn_elem_counts.iter().zip(dyn_elem_sizes) {
            let num_elems = self.handle_operand(num_elems)?;
            // If the element count is not the same width as LLVM's GEP index type, then we have to
            // sign extend it up (or truncate it down) to the right size. To date I've been unable
            // to get clang to emit code that would require an extend or truncate, so for now it's
            // a todo.
            if num_elems.byte_size(&self.jit_mod) * 8 != self.aot_mod.ptr_off_bitsize().into() {
                todo!();
            }
            let elem_size = u16::try_from(*elem_size).map_err(|_| {
                CompilationError::LimitExceeded("ptradd elem size doesn't fit in u16".into())
            })?;
            jit_ptr = self.jit_mod.push_and_make_operand(
                jit_ir::DynPtrAddInst::new(jit_ptr, num_elems, elem_size).into(),
            )?;
        }
        self.link_iid_to_last_inst(bid, aot_inst_idx);
        Ok(())
    }

    fn handle_cast(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        cast_kind: &aot_ir::CastKind,
        val: &aot_ir::Operand,
        dest_tyidx: &aot_ir::TyIdx,
    ) -> Result<(), CompilationError> {
        let inst = match cast_kind {
            aot_ir::CastKind::SExt => jit_ir::SExtInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::ZeroExtend => jit_ir::ZExtInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::Trunc => jit_ir::TruncInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::SIToFP => jit_ir::SIToFPInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::FPExt => jit_ir::FPExtInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::FPToSI => jit_ir::FPToSIInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
            aot_ir::CastKind::BitCast => jit_ir::BitCastInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_tyidx))?,
            )
            .into(),
        };
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_switch(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        safepoint: &'static aot_ir::DeoptSafepoint,
        next_bb: &aot_ir::BBlockId,
        test_val: &aot_ir::Operand,
        _default_dest: &aot_ir::BBlockIdx,
        case_values: &[u64],
        case_dests: &[aot_ir::BBlockIdx],
    ) -> Result<(), CompilationError> {
        if case_values.is_empty() {
            // Degenerate switch. Not sure it can even happen.
            panic!();
        }

        let jit_tyidx = self.handle_type(test_val.type_(self.aot_mod))?;
        let bitw = self.jit_mod.type_(jit_tyidx).bitw().unwrap();

        // Find out which case we traced.
        let guard = match case_dests.iter().position(|&cd| cd == next_bb.bbidx()) {
            Some(cidx) => {
                // A non-default case was traced.
                let val = case_values[cidx];
                let bb = case_dests[cidx];

                // Build the constant value to guard.
                let jit_const = jit_ir::Const::Int(jit_tyidx, ArbBitInt::from_u64(bitw, val));
                let jit_const_opnd = jit_ir::Operand::Const(self.jit_mod.insert_const(jit_const)?);

                // Perform the comparison.
                let jit_test_val = self.handle_operand(test_val)?;
                let cmp_inst =
                    jit_ir::ICmpInst::new(jit_test_val, jit_ir::Predicate::Equal, jit_const_opnd);
                let jit_cond = self.jit_mod.push_and_make_operand(cmp_inst.into())?;

                // Guard the result of the comparison.
                self.create_guard(bid, &jit_cond, bb == next_bb.bbidx(), safepoint)?
            }
            None => {
                // The default case was traced.
                //
                // We need a guard that expresses that `test_val` was not any of the `case_values`.
                // We encode this in the JIT IR like:
                //
                //     member = test_val == case_val1 || ... || test_val == case_valN
                //     guard member == 0
                //
                // OPT: This could be much more efficient if we introduced a special "none of these
                // values" guard. It could codegen the comparisons into a bitfield and efficiently
                // bitwise OR a whole word's worth of comparison outcomes at once.
                //
                // OPT: Also depending on the shape of the cases you may be able to optimise. e.g.
                // if they are a consecutive run, you could do a range check instead of all of
                // these comparisons.
                let mut cmps_opnds = Vec::new();
                for cv in case_values {
                    // Build a constant of the case value.
                    let jit_const = jit_ir::Const::Int(jit_tyidx, ArbBitInt::from_u64(bitw, *cv));
                    let jit_const_opnd =
                        jit_ir::Operand::Const(self.jit_mod.insert_const(jit_const)?);

                    // Do the comparison.
                    let jit_test_val = self.handle_operand(test_val)?;
                    let cmp = jit_ir::ICmpInst::new(
                        jit_test_val,
                        jit_ir::Predicate::Equal,
                        jit_const_opnd,
                    );
                    cmps_opnds.push(self.jit_mod.push_and_make_operand(cmp.into())?);
                }

                // OR together all the equality tests.
                let mut jit_cond = None;
                for cmp in cmps_opnds {
                    if jit_cond.is_none() {
                        jit_cond = Some(cmp);
                    } else {
                        // unwrap can't fail due to the above.
                        let lhs = jit_cond.take().unwrap();
                        let or = jit_ir::BinOpInst::new(lhs, BinOp::Or, cmp);
                        jit_cond = Some(self.jit_mod.push_and_make_operand(or.into())?);
                    }
                }

                // Guard the result of ORing all the comparisons together.
                // unwrap can't fail: we already disregarded degenerate switches with no
                // non-default cases.
                self.create_guard(bid, &jit_cond.unwrap(), false, safepoint)?
            }
        };
        self.copy_inst(guard.into(), bid, aot_inst_idx)
    }

    fn handle_phi(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        prev_bb: &aot_ir::BBlockIdx,
        incoming_bbs: &[aot_ir::BBlockIdx],
        incoming_vals: &[aot_ir::Operand],
    ) -> Result<(), CompilationError> {
        // If the IR is well-formed the indexing and unwrap() here will not fail.
        let chosen_val = &incoming_vals[incoming_bbs.iter().position(|bb| bb == prev_bb).unwrap()];
        let aot_iit = aot_ir::InstID::new(
            bid.funcidx(),
            bid.bbidx(),
            aot_ir::InstIdx::new(aot_inst_idx),
        );
        let op = match self.handle_operand(chosen_val)? {
            jit_ir::Operand::Const(c) => {
                // We don't want to do constant propagation here as it makes our life harder
                // creating guards. Instead we simply create a `Const` instruction here and
                // reference that.
                let inst = jit_ir::Inst::Const(c);
                self.jit_mod.push(inst)?;
                jit_ir::Operand::Var(self.jit_mod.last_inst_idx())
            }
            op => op,
        };
        self.local_map.insert(aot_iit, op);
        Ok(())
    }

    fn handle_select(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        cond: &aot_ir::Operand,
        trueval: &aot_ir::Operand,
        falseval: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::SelectInst::new(
            self.handle_operand(cond)?,
            self.handle_operand(trueval)?,
            self.handle_operand(falseval)?,
        )
        .into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    /// Turn outlining until the specified call has been consumed from the trace.
    ///
    /// `terminst` is the terminating instruction immediately after the call, guaranteed to be
    /// present due to ykllvm's block splitting pass.
    ///
    /// `bid` is the [BBlockId] containing the call.
    fn outline_until_after_call(&mut self, bid: &BBlockId, terminst: &'static aot_ir::Inst) {
        match terminst {
            aot_ir::Inst::Br { succ } => {
                // We can only stop outlining when we see the succesor block and we are not in
                // the middle of recursion.
                let succbid = BBlockId::new(bid.funcidx(), *succ);
                self.outline_target_blk = Some(succbid);
                self.recursion_count = 0;
            }
            aot_ir::Inst::CondBr { .. } => {
                // Currently, the successor of a call is always an unconditional branch due to
                // the block spitting pass. However, there's a FIXME in that pass which could
                // lead to conditional branches showing up here. Leave a todo here so we know
                // when this happens.
                todo!()
            }
            _ => panic!(),
        }
    }

    fn handle_promote(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        val: &aot_ir::Operand,
        safepoint: &'static aot_ir::DeoptSafepoint,
        tyidx: &aot_ir::TyIdx,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        self.outline_until_after_call(bid, nextinst);
        match self.handle_operand(val)? {
            jit_ir::Operand::Var(ref_iidx) => {
                self.jit_mod.push(jit_ir::Inst::Copy(ref_iidx))?;
                self.link_iid_to_last_inst(bid, aot_inst_idx);

                // Insert a guard to ensure the trace only runs if the value we encounter is the
                // same each time.
                let tyidx = self.handle_type(self.aot_mod.type_(*tyidx))?;
                // Create the constant from the runtime value.
                let ty = self.jit_mod.type_(tyidx);
                let c = match ty {
                    jit_ir::Ty::Void => unreachable!(),
                    jit_ir::Ty::Integer(bitw) => {
                        let bytew = ty.byte_size().unwrap();
                        let v = match *bitw {
                            64 => u64::from_ne_bytes(
                                self.promotions[self.promote_idx..self.promote_idx + bytew]
                                    .try_into()
                                    .unwrap(),
                            ),
                            32 => u64::from(u32::from_ne_bytes(
                                self.promotions[self.promote_idx..self.promote_idx + bytew]
                                    .try_into()
                                    .unwrap(),
                            )),
                            x => todo!("{x}"),
                        };
                        self.promote_idx += bytew;
                        Const::Int(tyidx, ArbBitInt::from_u64(*bitw, v))
                    }
                    jit_ir::Ty::Ptr => todo!(),
                    jit_ir::Ty::Func(_) => todo!(),
                    jit_ir::Ty::Float(_) => todo!(),
                    jit_ir::Ty::Unimplemented(_) => todo!(),
                };
                let cidx = self.jit_mod.insert_const(c)?;
                let cmp_instr = jit_ir::ICmpInst::new(
                    jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
                    aot_ir::Predicate::Equal,
                    jit_ir::Operand::Const(cidx),
                );
                let jit_cond = self.jit_mod.push_and_make_operand(cmp_instr.into())?;
                let guard = self.create_guard(bid, &jit_cond, true, safepoint)?;
                self.copy_inst(guard.into(), bid, aot_inst_idx)
            }
            jit_ir::Operand::Const(_cidx) => todo!(),
        }
    }

    fn handle_fneg(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        val: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::FNegInst::new(self.handle_operand(val)?).into();
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    fn handle_debug_str(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        self.outline_until_after_call(bid, nextinst);
        let msg = self.debug_strs[self.debug_str_idx].to_owned();
        let inst =
            jit_ir::Inst::DebugStr(jit_ir::DebugStrInst::new(self.jit_mod.push_debug_str(msg)?));
        self.copy_inst(inst, bid, aot_inst_idx)?;
        self.debug_str_idx += 1;
        Ok(())
    }

    /// Entry point for building an IR trace.
    ///
    /// Consumes the trace builder, returning a JIT module.
    ///
    /// # Panics
    ///
    /// If `ta_iter` produces no elements.
    fn build(
        mut self,
        ta_iter: Box<dyn AOTTraceIterator>,
    ) -> Result<jit_ir::Module, CompilationError> {
        // Collect the trace first so we can peek at the last element to find the control point
        // block during side-tracing.
        // FIXME: This is a workaround so we can peek at the last block in a trace in order to
        // extract information from the control point when we are side-tracing. Ideally, we extract
        // this information at AOT compile time and serialise it into the module (or block), so we
        // don't have to do this. Note, we also can't use `collect` here since that won't catch the
        // `TraceTooLong` error early enough and we run out of memory.
        let tas = ta_iter
            .map(|x| {
                x.map_err(|e| match e {
                    AOTTraceIteratorError::TraceTooLong => {
                        CompilationError::LimitExceeded("Trace too long.".into())
                    }
                    x => CompilationError::General(x.to_string()),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The previous processed block.
        let mut prev_bid = None;

        if let TraceKind::Sidetrace(sti) = self.jit_mod.tracekind() {
            let sti = Arc::clone(sti)
                .as_any()
                .downcast::<YkSideTraceInfo<Register>>()
                .unwrap();
            // Set the previous block to the last block in the parent trace. Required in order to
            // process phi nodes.
            prev_bid = Some(sti.bid.clone());

            // Setup the trace builder for side-tracing.
            let lastblk = match &tas.last() {
                Some(b) => self.lookup_aot_block(b),
                _ => todo!(),
            };
            // Create loads for the live variables that will be passed into the side-trace from the
            // parent trace.
            for (idx, (aotid, loc)) in sti.lives().iter().enumerate() {
                let aotinst = self.aot_mod.inst(aotid);
                let aotty = aotinst.def_type(self.aot_mod).unwrap();
                let tyidx = self.handle_type(aotty)?;
                let param_inst = jit_ir::ParamInst::new(ParamIdx::try_from(idx)?, tyidx).into();
                self.jit_mod.push(param_inst)?;
                self.jit_mod.push_param(loc.clone());
                self.local_map.insert(
                    aotid.clone(),
                    jit_ir::Operand::Var(self.jit_mod.last_inst_idx()),
                );
            }
            self.cp_block = lastblk;
            self.frames = sti
                .callframes()
                .iter()
                .map(|x| InlinedFrame {
                    funcidx: Some(x.funcidx),
                    callinst: x.callinst.clone(),
                    safepoint: Some(x.safepoint),
                    args: x.args.clone(),
                })
                .collect::<Vec<_>>();

            // When side-tracing a switch guard failure, we need to reprocess the switch statement
            // (and only the switch statement) in order to emit a guard at the beginning of the
            // side-trace to check that the case requiring execution is that case the trace
            // captures.
            //
            // Note that it is not necessary to emit such a guard into side-traces stemming from
            // regular conditionals, since a conditional has only two sucessors. The parent trace
            // captures one, so by construction the side trace must capture the other.
            let prevbb = self.aot_mod.bblock(prev_bid.as_ref().unwrap());
            if let aot_ir::Inst::Switch {
                test_val,
                default_dest,
                case_values,
                case_dests,
                safepoint,
            } = &prevbb.insts.last().unwrap()
            {
                let nextbb = match &tas.first() {
                    Some(b) => self.lookup_aot_block(b),
                    _ => panic!(),
                };
                self.handle_switch(
                    prev_bid.as_ref().unwrap(), // this is safe, we've just created this above
                    prevbb.insts.len() - 1,
                    safepoint,
                    nextbb.as_ref().unwrap(),
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                )?;
            }
        }
        // The variable `prev_bid` contains the block of the guard that initiated side-tracing (for
        // normal traces this is set to `None`). When hardware tracing, we capture this block again
        // as part of the side-trace. However, since we've already processed this block in the
        // parent trace, we must not process it again in the side-trace.
        //
        // Typically, the mapper would strip this block for us, but for codegen related reasons,
        // e.g. a switch statement codegenning to many machine blocks, it's possible for multiple
        // duplicates of this same block to show up here, which all need to be skipped.
        let mut trace_iter = tas.into_iter().peekable();
        if prev_bid.is_some() {
            while self.lookup_aot_block(trace_iter.peek().unwrap()) == prev_bid {
                trace_iter.next().unwrap();
            }
        }

        match self.jit_mod.tracekind() {
            TraceKind::HeaderOnly | TraceKind::HeaderAndBody => {
                // Find the block containing the control point call. This is the (sole) predecessor of the
                // first (guaranteed mappable) block in the trace. Note that empty traces are handled in
                // the tracing phase so the `unwrap` is safe.
                let prev = match trace_iter.peek().unwrap() {
                    TraceAction::MappedAOTBBlock { func_name, bb } => {
                        debug_assert!(*bb > 0);
                        // It's `- 1` due to the way the ykllvm block splitting pass works.
                        TraceAction::MappedAOTBBlock {
                            func_name: func_name.clone(),
                            bb: bb - 1,
                        }
                    }
                    TraceAction::UnmappableBBlock => panic!(),
                    TraceAction::Promotion => todo!(),
                };

                self.cp_block = self.lookup_aot_block(&prev);
                self.frames.last_mut().unwrap().funcidx =
                    Some(self.cp_block.as_ref().unwrap().funcidx());
                // This unwrap can't fail. If it does that means the tracer has given us a mappable block
                // that doesn't exist in the AOT module.
                self.create_trace_header(self.aot_mod.bblock(self.cp_block.as_ref().unwrap()))?;
            }
            TraceKind::Sidetrace(_) => (),
        }

        // FIXME: this section of code needs to be refactored.
        #[cfg(tracer_hwt)]
        let mut last_blk_is_return = false;
        while let Some(b) = trace_iter.next() {
            match self.lookup_aot_block(&b) {
                Some(bid) => {
                    // MappedAOTBBlock block
                    if let Some(ref tgtbid) = self.outline_target_blk {
                        // We are currently outlining.
                        if bid.funcidx() == tgtbid.funcidx() {
                            // We are inside the same function that started outlining.
                            if bid.is_entry() {
                                // We are recursing into the function that started
                                // outlining.
                                self.recursion_count += 1;
                            }
                            if self.aot_mod.bblock(&bid).is_return() {
                                // We are returning from the function that started
                                // outlining. This may be one of multiple inlined calls, so
                                // we may not be done outlining just yet.
                                self.recursion_count -= 1;
                            }
                        }
                        if self.recursion_count == 0 && bid == *tgtbid {
                            // We've reached the successor block of the function/block that
                            // started outlining. We are done and can continue processing
                            // blocks normally.
                            self.outline_target_blk = None;
                        } else {
                            // We are outlining so just skip this block. However, we still need to
                            // process promoted values to make sure we've processed all promotion
                            // data and haven't messed up the mapping.
                            #[cfg(tracer_hwt)]
                            {
                                // Due to hardware tracing we see the same block twice whenever
                                // there is a call. We only need to process one of them. We can
                                // skip the block if:
                                //  a) The previous block had a return.
                                //  b) The previous block is unmappable and the current block isn't
                                //  an entry block.
                                if last_blk_is_return {
                                    last_blk_is_return = self.aot_mod.bblock(&bid).is_return();
                                    prev_bid = Some(bid);
                                    continue;
                                }
                                last_blk_is_return = self.aot_mod.bblock(&bid).is_return();
                                if prev_bid.is_none() && !bid.is_entry() {
                                    prev_bid = Some(bid);
                                    continue;
                                }
                            }
                            self.process_promotions_and_debug_strs_only(&bid)?;
                            prev_bid = Some(bid);
                            continue;
                        }
                    } else {
                        // We are not outlining. Process blocks normally.
                        #[cfg(tracer_hwt)]
                        if last_blk_is_return {
                            // If the last block had a return terminator, we are returning
                            // from a call, which means the HWT will have recorded an
                            // additional caller block that we'll have to skip.
                            // FIXME: This only applies to the HWT as the SWT doesn't
                            // record these extra blocks.
                            last_blk_is_return = false;
                            prev_bid = Some(bid);
                            continue;
                        }
                        #[cfg(tracer_hwt)]
                        if self.aot_mod.bblock(&bid).is_return() {
                            last_blk_is_return = true;
                        }
                    }

                    // In order to emit guards for conditional branches we need to peek at the next
                    // block.
                    let nextbb = trace_iter.peek().and_then(|x| self.lookup_aot_block(x));
                    self.process_block(&bid, &prev_bid, nextbb)?;
                    if self.cp_block.as_ref() == Some(&bid) {
                        // When using the hardware tracer we will see two control point
                        // blocks here. We must only process one of them. The simplest way
                        // to do this that is compatible with the software tracer is to
                        // start outlining here by setting the outlining stop target to
                        // this blocks successor.
                        let blk = self.aot_mod.bblock(&bid);
                        let br = blk.insts.iter().last();
                        // Unwrap safe: block guaranteed to have instructions.
                        match br.unwrap() {
                            aot_ir::Inst::Br { succ } => {
                                let succbid = BBlockId::new(bid.funcidx(), *succ);
                                self.outline_target_blk = Some(succbid);
                                self.recursion_count = 0;
                            }
                            _ => panic!(),
                        }
                    }
                    prev_bid = Some(bid);
                }
                None => {
                    // Unmappable block
                    prev_bid = None;
                }
            }
        }

        debug_assert_eq!(self.promote_idx, self.promotions.len());
        debug_assert_eq!(self.debug_str_idx, self.debug_strs.len());
        let blk = self.aot_mod.bblock(self.cp_block.as_ref().unwrap());
        let cpcall = blk.insts.iter().rev().nth(1).unwrap();
        debug_assert!(cpcall.is_control_point(self.aot_mod));
        match self.jit_mod.tracekind() {
            TraceKind::Sidetrace(_) => {
                // This is the end of a side-trace. Create a jump back to the root trace.
                let safepoint = cpcall.safepoint().unwrap();
                for idx in 0..safepoint.lives.len() {
                    let aot_op = &safepoint.lives[idx];
                    let jit_op = &self.local_map[&aot_op.to_inst_id()];
                    self.jit_mod.push_header_end_var(jit_op.clone());
                }
                self.jit_mod.push(jit_ir::Inst::SidetraceEnd)?;
            }
            TraceKind::HeaderAndBody | TraceKind::HeaderOnly => {
                // For normal traces insert a jump back to the loop start.
                let safepoint = cpcall.safepoint().unwrap();
                for idx in 0..safepoint.lives.len() {
                    let aot_op = &safepoint.lives[idx];
                    let jit_op = &self.local_map[&aot_op.to_inst_id()];
                    self.jit_mod.push_header_end_var(jit_op.clone());
                }
                self.jit_mod.push(jit_ir::Inst::TraceHeaderEnd)?;
            }
        }

        Ok(self.jit_mod)
    }
}

/// A local version of [jit_ir::InlinedFrame] that deals with the fact that we build up information
/// about an inlined frame bit-by-bit using `Option`s, all of which will end up as `Some`.
#[derive(Debug, Clone)]
struct InlinedFrame {
    funcidx: Option<aot_ir::FuncIdx>,
    callinst: Option<aot_ir::InstID>,
    safepoint: Option<&'static aot_ir::DeoptSafepoint>,
    args: Vec<PackedOperand>,
}

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn build<Register: Send + Sync + 'static>(
    ctr_id: CompiledTraceId,
    aot_mod: &'static Module,
    ta_iter: Box<dyn AOTTraceIterator>,
    sti: Option<Arc<YkSideTraceInfo<Register>>>,
    promotions: Box<[u8]>,
    debug_strs: Vec<String>,
) -> Result<jit_ir::Module, CompilationError> {
    let tracekind = if let Some(x) = sti {
        TraceKind::Sidetrace(x)
    } else {
        TraceKind::HeaderOnly
    };
    TraceBuilder::<Register>::new(tracekind, ctr_id, aot_mod, promotions, debug_strs)?
        .build(ta_iter)
}
