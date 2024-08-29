//! The JIT IR trace builder.
//!
//! This takes in an (AOT IR, execution trace) pair and constructs a JIT IR trace from it.

use super::aot_ir::{self, BBlockId, BinOp, Module};
use super::YkSideTraceInfo;
use super::{
    jit_ir::{self, Const, Operand, PackedOperand},
    AOT_MOD,
};
use crate::aotsmp::AOT_STACKMAPS;
use crate::compile::CompilationError;
use crate::trace::{AOTTraceIterator, AOTTraceIteratorError, TraceAction};
use std::{collections::HashMap, ffi::CString, sync::Arc};

/// The argument index of the trace inputs struct in the trace function.
const U64SIZE: usize = 8;

/// Given an execution trace and AOT IR, creates a JIT IR trace.
pub(crate) struct TraceBuilder {
    /// The AOT IR.
    aot_mod: &'static Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    /// Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstID, jit_ir::Operand>,
    /// BBlock containing the current control point (i.e. the control point that started this trace).
    cp_block: Option<aot_ir::BBlockId>,
    /// Index of the first traceinput instruction.
    first_ti_idx: usize,
    /// Inlined calls.
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
}

impl TraceBuilder {
    /// Create a trace builder.
    ///
    /// Arguments:
    ///  - `aot_mod`: The AOT IR module that the trace flows through.
    ///  - `mtrace`: The mapped trace.
    ///  - `promotions`: Values promoted to constants during runtime.
    fn new(
        ctr_id: u64,
        aot_mod: &'static Module,
        promotions: Box<[u8]>,
    ) -> Result<Self, CompilationError> {
        Ok(Self {
            aot_mod,
            jit_mod: jit_ir::Module::new(ctr_id, aot_mod.global_decls_len())?,
            local_map: HashMap::new(),
            cp_block: None,
            first_ti_idx: 0,
            // We have to set the funcidx to None here as we don't know what it is yet. We'll
            // update it as soon as we do.
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
            TraceAction::UnmappableBBlock { .. } => None,
            TraceAction::Promotion => todo!(),
        }
    }

    /// Create the prolog of the trace.
    fn create_trace_header(
        &mut self,
        blk: &'static aot_ir::BBlock,
        is_sidetrace: bool,
    ) -> Result<(), CompilationError> {
        if is_sidetrace {
            todo!();
        }
        // Find the control point call and extract the trace inputs struct from its operands.
        //
        // FIXME: Stash the location at IR lowering time, instead of searching at runtime.
        let mut inst_iter = blk.insts.iter().enumerate().rev();
        for (_, inst) in inst_iter.by_ref() {
            // Is it a call to the control point, then insert loads for the trace inputs. These
            // directly reference registers or stack slots in the parent frame and thus don't
            // necessarily result in machine code during codegen.
            if inst.is_control_point(self.aot_mod) {
                let safepoint = inst.safepoint().unwrap();
                let (rec, _) = AOT_STACKMAPS
                    .as_ref()
                    .unwrap()
                    .get(usize::try_from(safepoint.id).unwrap());

                debug_assert!(safepoint.lives.len() == rec.live_vars.len());
                for idx in 0..safepoint.lives.len() {
                    let aot_op = &safepoint.lives[idx];
                    let input_tyidx = self.handle_type(aot_op.type_(self.aot_mod))?;
                    let load_ti_inst =
                        jit_ir::LoadTraceInputInst::new(u32::try_from(idx).unwrap(), input_tyidx)
                            .into();
                    self.jit_mod.push(load_ti_inst)?;

                    // Get the location for this input variable.
                    let var = &rec.live_vars[idx];
                    if var.len() > 1 {
                        todo!();
                    }
                    self.jit_mod.push_tiloc(var.get(0).unwrap().clone());
                    self.local_map.insert(
                        aot_op.to_inst_id(),
                        jit_ir::Operand::Local(self.jit_mod.last_inst_idx()),
                    );
                    self.jit_mod
                        .push_loop_start_var(jit_ir::Operand::Local(self.jit_mod.last_inst_idx()));
                }
                break;
            }
        }
        self.jit_mod.push(jit_ir::Inst::TraceLoopStart)?;
        Ok(())
    }

    /// Walk over a traced AOT block, translating the constituent instructions into the JIT module.
    fn process_block(
        &mut self,
        bid: &aot_ir::BBlockId,
        prevbb: &Option<aot_ir::BBlockId>,
        nextbb: Option<aot_ir::BBlockId>,
    ) -> Result<(), CompilationError> {
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
                    if self.cp_block.as_ref() == Some(bid) && iidx == self.first_ti_idx {
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
                } => self.handle_condbr(safepoint, nextbb.as_ref().unwrap(), cond, true_bb),
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
                    self.local_map.insert(aot_iid, jitop.clone());
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
        self.local_map.insert(
            aot_iid,
            jit_ir::Operand::Local(self.jit_mod.last_inst_idx()),
        );
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
            aot_ir::Ty::Integer(aot_ir::IntegerTy { num_bits }) => {
                // FIXME: It would be better if the AOT IR had converted these integers in advance
                // rather than doing this dance here.
                let x = match num_bits {
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
                    _ => todo!("{}", num_bits),
                };
                let jit_tyidx = self.jit_mod.insert_ty(jit_ir::Ty::Integer(*num_bits))?;
                Ok(jit_ir::Const::Int(jit_tyidx, x))
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
            aot_ir::Ty::Integer(it) => jit_ir::Ty::Integer(it.num_bits()),
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
                            jit_ir::Operand::Local(liidx) => live_vars.push((
                                iid.clone(),
                                PackedOperand::new(&jit_ir::Operand::Local(liidx)),
                            )),
                            jit_ir::Operand::Const(_) => {
                                // Since we are forcing constants into `ProxyConst`s during inlining, this
                                // case should never happen. If you see this panic, then look for a
                                // safepoint live variable that maps to a constant and make the builder
                                // insert a `ProxyConst` for it instead.
                                panic!("constant encountered while building guardinfo!")
                            }
                        }
                    }
                    _ => panic!(), // IR malformed.
                }
            }
        }

        let gi = jit_ir::GuardInfo::new(live_vars, callframes);
        let gi_idx = self.jit_mod.push_guardinfo(gi)?;

        Ok(jit_ir::GuardInst::new(cond.clone(), expect, gi_idx))
    }

    /// Translate a conditional `Br` instruction.
    fn handle_condbr(
        &mut self,
        safepoint: &'static aot_ir::DeoptSafepoint,
        next_bb: &aot_ir::BBlockId,
        cond: &aot_ir::Operand,
        true_bb: &aot_ir::BBlockIdx,
    ) -> Result<(), CompilationError> {
        let jit_cond = self.handle_operand(cond)?;
        let guard = self.create_guard(&jit_cond, *true_bb == next_bb.bbidx(), safepoint)?;
        self.jit_mod.push(guard.into())?;
        Ok(())
    }

    fn handle_ret(
        &mut self,
        _bid: &aot_ir::BBlockId,
        _aot_inst_idx: usize,
        val: &Option<aot_ir::Operand>,
    ) -> Result<(), CompilationError> {
        // FIXME: Map return value to AOT call instruction.
        let frame = self.frames.pop().unwrap();
        if let Some(val) = val {
            let op = self.handle_operand(val)?;
            self.local_map.insert(frame.callinst.unwrap(), op);
        }
        Ok(())
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

        match nextinst {
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
        // `yk_trace_basicblock` instruction calls into the beginning of
        // every basic block. These calls can be ignored as they are
        // only used to collect runtime information for the tracer itself.
        if AOT_MOD.func(*callee).name() == "yk_trace_basicblock" {
            return Ok(());
        }
        // Convert AOT args to JIT args.
        let mut jit_args = Vec::new();
        for arg in args {
            match self.handle_operand(arg)? {
                jit_ir::Operand::Const(c) => {
                    // We don't want to do constant propagation here as it makes our life harder
                    // creating guards. Instead we simply create a proxy instruction here and
                    // reference that.
                    let inst = jit_ir::Inst::ProxyConst(c);
                    self.jit_mod.push(inst)?;
                    let op = jit_ir::Operand::Local(self.jit_mod.last_inst_idx());
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
                args: jit_args,
            });
            Ok(())
        } else {
            // This call can't be inlined. It is either unmappable (a declaration or an indirect
            // call) or the compiler annotated it with `yk_outline`.
            match nextinst {
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
            if inst.is_control_point(self.aot_mod) {
                return Ok(());
            }
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
                Operand::Local(iidx) => self
                    .jit_mod
                    .push_and_make_operand(jit_ir::Inst::ProxyInst(iidx))?,
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
            aot_ir::CastKind::ZeroExtend => jit_ir::ZeroExtendInst::new(
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

        // Find out which case we traced.
        let guard = match case_dests.iter().position(|&cd| cd == next_bb.bbidx()) {
            Some(cidx) => {
                // A non-default case was traced.
                let val = case_values[cidx];
                let bb = case_dests[cidx];

                // Build the constant value to guard.
                let jit_const = jit_ir::Const::Int(jit_tyidx, val);
                let jit_const_opnd = jit_ir::Operand::Const(self.jit_mod.insert_const(jit_const)?);

                // Perform the comparison.
                let jit_test_val = self.handle_operand(test_val)?;
                let cmp_inst =
                    jit_ir::ICmpInst::new(jit_test_val, jit_ir::Predicate::Equal, jit_const_opnd);
                let jit_cond = self.jit_mod.push_and_make_operand(cmp_inst.into())?;

                // Guard the result of the comparison.
                self.create_guard(&jit_cond, bb == next_bb.bbidx(), safepoint)?
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
                    let jit_const = jit_ir::Const::Int(jit_tyidx, *cv);
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
                self.create_guard(&jit_cond.unwrap(), false, safepoint)?
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
                // creating guards. Instead we simply create a proxy instruction here and
                // reference that.
                let inst = jit_ir::Inst::ProxyConst(c);
                self.jit_mod.push(inst)?;
                jit_ir::Operand::Local(self.jit_mod.last_inst_idx())
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

    fn handle_promote(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        val: &aot_ir::Operand,
        safepoint: &'static aot_ir::DeoptSafepoint,
        tyidx: &aot_ir::TyIdx,
        nextinst: &'static aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        match nextinst {
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
        match self.handle_operand(val)? {
            jit_ir::Operand::Local(ref_iidx) => {
                self.jit_mod.push(jit_ir::Inst::ProxyInst(ref_iidx))?;
                self.link_iid_to_last_inst(bid, aot_inst_idx);

                // Insert a guard to ensure the trace only runs if the value we encounter is the
                // same each time.
                let ty = self.handle_type(self.aot_mod.type_(*tyidx))?;
                // Create the constant from the runtime value.
                let c = match self.jit_mod.type_(ty) {
                    jit_ir::Ty::Void => unreachable!(),
                    jit_ir::Ty::Integer(width_bits) => {
                        let width_bytes = usize::try_from(*width_bits).unwrap() / 8;
                        let v = match width_bits {
                            64 => u64::from_ne_bytes(
                                self.promotions[self.promote_idx..self.promote_idx + width_bytes]
                                    .try_into()
                                    .unwrap(),
                            ),
                            32 => u64::from(u32::from_ne_bytes(
                                self.promotions[self.promote_idx..self.promote_idx + width_bytes]
                                    .try_into()
                                    .unwrap(),
                            )),
                            x => todo!("{x}"),
                        };
                        self.promote_idx += width_bytes;
                        Const::Int(ty, v)
                    }
                    jit_ir::Ty::Ptr => todo!(),
                    jit_ir::Ty::Func(_) => todo!(),
                    jit_ir::Ty::Float(_) => todo!(),
                    jit_ir::Ty::Unimplemented(_) => todo!(),
                };
                let cidx = self.jit_mod.insert_const(c)?;
                let cmp_instr = jit_ir::ICmpInst::new(
                    jit_ir::Operand::Local(self.jit_mod.last_inst_idx()),
                    aot_ir::Predicate::Equal,
                    jit_ir::Operand::Const(cidx),
                );
                let jit_cond = self.jit_mod.push_and_make_operand(cmp_instr.into())?;
                let guard = self.create_guard(&jit_cond, true, safepoint)?;
                self.copy_inst(guard.into(), bid, aot_inst_idx)
            }
            jit_ir::Operand::Const(_cidx) => todo!(),
        }
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
        sti: Option<Arc<YkSideTraceInfo>>,
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

        if let Some(sti) = sti.as_ref() {
            // Setup the trace builder for side-tracing.
            let lastblk = match &tas.last() {
                Some(b) => self.lookup_aot_block(b),
                _ => todo!(),
            };
            self.create_trace_header(self.aot_mod.bblock(lastblk.as_ref().unwrap()), true)?;
            // Create loads for the live variables that will be passed into the side-trace from the
            // parent trace.
            let mut off = 0;
            for aotid in sti.aotlives() {
                let aotinst = self.aot_mod.inst(aotid);
                // Unwrap is safe as all live variables must be definitions.
                let aotty = aotinst.def_type(self.aot_mod).unwrap();
                let tyidx = self.handle_type(aotty)?;
                let load_ti_inst =
                    jit_ir::LoadTraceInputInst::new(u32::try_from(off).unwrap(), tyidx).into();
                self.jit_mod.push(load_ti_inst)?;
                self.local_map.insert(
                    aotid.clone(),
                    jit_ir::Operand::Local(self.jit_mod.last_inst_idx()),
                );
                // FIXME: We currently pass all live variables into the side-trace as `u64`s.
                off += U64SIZE;
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
        }

        let mut trace_iter = tas.into_iter().peekable();

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

        if sti.is_none() {
            self.cp_block = self.lookup_aot_block(&prev);
            self.frames.last_mut().unwrap().funcidx =
                Some(self.cp_block.as_ref().unwrap().funcidx());
            // This unwrap can't fail. If it does that means the tracer has given us a mappable block
            // that doesn't exist in the AOT module.
            self.create_trace_header(self.aot_mod.bblock(self.cp_block.as_ref().unwrap()), false)?;
        }

        // FIXME: this section of code needs to be refactored.
        #[cfg(tracer_hwt)]
        let mut last_blk_is_return = false;
        let mut prev_bid = None;
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
                            // We are outlining so just skip this block.
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
                    // UnmappableBBlock block
                    prev_bid = None;
                }
            }
        }

        let blk = self.aot_mod.bblock(self.cp_block.as_ref().unwrap());
        let cpcall = blk.insts.iter().rev().nth(1).unwrap();
        debug_assert!(cpcall.is_control_point(self.aot_mod));
        // If this is a side trace insert a guard that always fails so we can safely return to the
        // beginning of the control point where this trace ended.
        if sti.is_some() {
            // Make a 0 constant.
            let jitconst = jit_ir::Const::Int(self.jit_mod.int1_tyidx(), 0);
            let constop = jit_ir::Operand::Const(self.jit_mod.insert_const(jitconst)?);

            // Create a guard that will always fail.
            let guard = self.create_guard(&constop, true, cpcall.safepoint().unwrap())?;
            self.jit_mod.push(guard.into())?;
        } else {
            // For normal traces insert a jump back to the loop start.
            let safepoint = cpcall.safepoint().unwrap();
            for idx in 0..safepoint.lives.len() {
                let aot_op = &safepoint.lives[idx];
                let jit_op = &self.local_map[&aot_op.to_inst_id()];
                self.jit_mod.push_loop_jump_var(jit_op.clone());
            }
            self.jit_mod.push(jit_ir::Inst::TraceLoopJump)?;
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
    args: Vec<Operand>,
}

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn build(
    ctr_id: u64,
    aot_mod: &'static Module,
    ta_iter: Box<dyn AOTTraceIterator>,
    sti: Option<Arc<YkSideTraceInfo>>,
    promotions: Box<[u8]>,
) -> Result<jit_ir::Module, CompilationError> {
    TraceBuilder::new(ctr_id, aot_mod, promotions)?.build(ta_iter, sti)
}
