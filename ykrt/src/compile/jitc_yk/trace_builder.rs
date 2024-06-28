//! The JIT IR trace builder.
//!
//! This takes in an (AOT IR, execution trace) pair and constructs a JIT IR trace from it.

use super::aot_ir::{self, BBlockId, BinOp, FuncIdx, Module};
use super::jit_ir;
use crate::compile::CompilationError;
use crate::trace::{AOTTraceIterator, AOTTraceIteratorError, TraceAction};
use std::{collections::HashMap, ffi::CString};

/// The argument index of the trace inputs struct in the trace function.
const TRACE_FUNC_CTRLP_ARGIDX: u16 = 0;

/// A TraceBuilder frame. Keeps track of inlined calls and stores information about the last
/// processed safepoint, call instruction and its arguments.
struct Frame<'a> {
    // The call instruction of this frame.
    callinst: Option<aot_ir::InstID>,
    // Index of the function of this frame.
    funcidx: Option<FuncIdx>,
    /// Safepoint for this frame.
    safepoint: Option<&'a aot_ir::DeoptSafepoint>,
    /// JIT arguments of this frame's caller.
    args: Vec<jit_ir::Operand>,
}

impl<'a> Frame<'a> {
    fn new(
        callinst: Option<aot_ir::InstID>,
        funcidx: Option<FuncIdx>,
        safepoint: Option<&'a aot_ir::DeoptSafepoint>,
        args: Vec<jit_ir::Operand>,
    ) -> Frame<'a> {
        Frame {
            callinst,
            funcidx,
            safepoint,
            args,
        }
    }
}

/// Given an execution trace and AOT IR, creates a JIT IR trace.
pub(crate) struct TraceBuilder<'a> {
    /// The AOT IR.
    aot_mod: &'a Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    /// Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstID, jit_ir::Operand>,
    // BBlock containing the current control point (i.e. the control point that started this trace).
    cp_block: Option<aot_ir::BBlockId>,
    // Index of the first traceinput instruction.
    first_ti_idx: usize,
    // Inlined calls.
    frames: Vec<Frame<'a>>,
    // The block at which to stop outlining.
    outline_target_blk: Option<BBlockId>,
    // Current count of recursive calls to the function in which outlining was started. Will be 0
    // if `outline_target_blk` is None.
    recursion_count: usize,
}

impl<'a> TraceBuilder<'a> {
    /// Create a trace builder.
    ///
    /// Arguments:
    ///  - `aot_mod`: The AOT IR module that the trace flows through.
    ///  - `mtrace`: The mapped trace.
    fn new(ctr_id: u64, aot_mod: &'a Module) -> Result<Self, CompilationError> {
        Ok(Self {
            aot_mod,
            jit_mod: jit_ir::Module::new(ctr_id, aot_mod.global_decls_len())?,
            local_map: HashMap::new(),
            cp_block: None,
            first_ti_idx: 0,
            // We have to set the funcidx to None here as we don't know what it is yet. We'll
            // update it as soon as we do.
            frames: vec![Frame::new(None, None, None, vec![])],
            outline_target_blk: None,
            recursion_count: 0,
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
    fn create_trace_header(&mut self, blk: &aot_ir::BBlock) -> Result<(), CompilationError> {
        // Find trace input variables and emit `LoadTraceInput` instructions for them.
        let mut trace_inputs = None;

        // PHASE 1:
        // Find the control point call and extract the trace inputs struct from its operands.
        //
        // FIXME: Stash the location at IR lowering time, instead of searching at runtime.
        let mut inst_iter = blk.insts.iter().enumerate().rev();
        for (_, inst) in inst_iter.by_ref() {
            // Is it a call to the control point? If so, extract the live vars struct.
            if let Some(tis) = inst.control_point_call_trace_inputs(self.aot_mod) {
                trace_inputs = Some(tis.to_inst(self.aot_mod));
                let arg = jit_ir::Inst::Arg(TRACE_FUNC_CTRLP_ARGIDX);
                self.jit_mod.push(arg)?;
                // Add the trace input argument to the local map so it can be tracked and
                // deoptimised.
                self.local_map.insert(
                    tis.to_inst_id(),
                    jit_ir::Operand::Local(self.jit_mod.last_inst_idx()),
                );
                break;
            }
        }

        // If this unwrap fails, we didn't find the call to the control point and something is
        // profoundly wrong with the AOT IR.
        let trace_inputs = trace_inputs.unwrap();
        let trace_input_struct_ty = match trace_inputs {
            aot_ir::Inst::Alloca { tyidx, .. } => {
                let aot_ir::Ty::Struct(x) = self.aot_mod.type_(*tyidx) else {
                    panic!()
                };
                x
            }
            _ => panic!(), // IR malformed.
        };

        // We visit the trace inputs in reverse order, so we start with high indices first. This
        // value then decrements in the below loop.
        let mut trace_input_idx = trace_input_struct_ty.num_fields();

        // PHASE 2:
        // Keep walking backwards over the ptradd/store pairs emitting LoadTraceInput instructions.
        //
        // FIXME: Can we do something at IR lowering time to make this easier?
        let mut last_store_ptr = None;
        for (iidx, inst) in inst_iter {
            match inst {
                aot_ir::Inst::Store { val, .. } => last_store_ptr = Some(val),
                aot_ir::Inst::PtrAdd { ptr, .. } => {
                    // Is the pointer operand of this PtrAdd targeting the trace inputs?
                    if trace_inputs.ptr_eq(ptr.to_inst(self.aot_mod)) {
                        // We found a trace input. Now we emit a `LoadTraceInput` instruction into the
                        // trace. This assigns the input to a local variable that other instructions
                        // can then use.
                        //
                        // Note: This code assumes that the `PtrAdd` instructions in the AOT IR were
                        // emitted sequentially.
                        trace_input_idx -= 1;
                        // FIXME: assumes the field is byte-aligned. If it isn't, field_byte_off() will
                        // crash.
                        let aot_field_off = trace_input_struct_ty.field_byte_off(trace_input_idx);
                        let aot_field_ty = trace_input_struct_ty.field_tyidx(trace_input_idx);
                        // FIXME: we should check at compile-time that this will fit.
                        match u32::try_from(aot_field_off) {
                            Ok(u32_off) => {
                                let input_tyidx =
                                    self.handle_type(self.aot_mod.type_(aot_field_ty))?;
                                let load_ti_inst =
                                    jit_ir::LoadTraceInputInst::new(u32_off, input_tyidx).into();
                                self.jit_mod.push(load_ti_inst)?;
                                // If this take fails, we didn't see a corresponding store and the
                                // IR is malformed.
                                self.local_map.insert(
                                    last_store_ptr.take().unwrap().to_inst_id(),
                                    jit_ir::Operand::Local(self.jit_mod.last_inst_idx()),
                                );
                                self.first_ti_idx = iidx;
                            }
                            _ => {
                                return Err(CompilationError::InternalError(
                                    "Offset {aot_field_off} doesn't fit".into(),
                                ));
                            }
                        }
                    }
                }
                _ => (),
            }
        }

        // Mark this location as the start of the trace loop.
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
                _ => todo!("{:?}", inst),
            }?;
        }
        Ok(())
    }

    /// Link the AOT IR to the last instruction pushed into the JIT IR.
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
        safepoint: &'a aot_ir::DeoptSafepoint,
    ) -> Result<jit_ir::GuardInst, CompilationError> {
        // Assign this branch's stackmap to the current frame.
        self.frames.last_mut().unwrap().safepoint = Some(safepoint);

        // Collect the safepoint IDs and live variables from this conditional branch and the
        // previous frames to store inside the guard.
        // Unwrap-safe as each frame at this point must have a safepoint associated with it.
        let mut smids = Vec::new(); // List of stackmap ids of the current call stack.
        let mut live_args = Vec::new(); // List of live JIT variables.
        for safepoint in self.frames.iter().map(|f| (f.safepoint.unwrap())) {
            let aot_ir::Operand::Const(cidx) = safepoint.id else {
                panic!();
            };
            let c = self.aot_mod.const_(cidx);
            let aot_ir::Ty::Integer(ity) = self.aot_mod.const_type(c) else {
                panic!();
            };
            assert!(ity.num_bits() == u64::BITS);
            let id = u64::from_ne_bytes(c.unwrap_val().bytes()[0..8].try_into().unwrap());
            smids.push(id);

            // Collect live variables.
            for op in safepoint.lives.iter() {
                let op = match op {
                    aot_ir::Operand::LocalVariable(iid) => &self.local_map[iid],
                    _ => panic!(), // IR malformed.
                };
                match op {
                    jit_ir::Operand::Local(lidx) => {
                        live_args.push(*lidx);
                    }
                    jit_ir::Operand::Const(_) => {
                        // Since we are forcing constants into `ProxyConst`s during inlining, this
                        // case should never happen.
                        panic!()
                    }
                }
            }
        }

        let gi = jit_ir::GuardInfo::new(smids, live_args);
        let gi_idx = self.jit_mod.push_guardinfo(gi)?;

        Ok(jit_ir::GuardInst::new(cond.clone(), expect, gi_idx))
    }

    /// Translate a conditional `Br` instruction.
    fn handle_condbr(
        &mut self,
        safepoint: &'a aot_ir::DeoptSafepoint,
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

    /// Translate a `Icmp` instruction.
    fn handle_icmp(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        lhs: &aot_ir::Operand,
        pred: &aot_ir::Predicate,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let inst =
            jit_ir::IcmpInst::new(self.handle_operand(lhs)?, *pred, self.handle_operand(rhs)?)
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
        inst: &'a aot_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ftyidx: &aot_ir::TyIdx,
        callop: &aot_ir::Operand,
        args: &[aot_ir::Operand],
        nextinst: &'a aot_ir::Inst,
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
        inst: &'a aot_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        callee: &aot_ir::FuncIdx,
        args: &[aot_ir::Operand],
        nextinst: &'a aot_ir::Inst,
    ) -> Result<(), CompilationError> {
        // Ignore special functions that we neither want to inline nor copy.
        if inst.is_debug_call(self.aot_mod) {
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
            self.frames
                .push(Frame::new(Some(aot_iid), Some(*callee), None, jit_args));
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
        };
        self.copy_inst(inst, bid, aot_inst_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_switch(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        safepoint: &'a aot_ir::DeoptSafepoint,
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
                    jit_ir::IcmpInst::new(jit_test_val, jit_ir::Predicate::Equal, jit_const_opnd);
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
                    let cmp = jit_ir::IcmpInst::new(
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
        let op = self.handle_operand(chosen_val)?;
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
        let mut trace_iter = ta_iter.peekable();
        let first_blk = match trace_iter.peek() {
            Some(Ok(b)) => b,
            Some(Err(_)) => todo!(),
            None => {
                // Empty traces are handled in the tracing phase.
                panic!();
            }
        };

        // Find the block containing the control point call. This is the (sole) predecessor of the
        // first (guaranteed mappable) block in the trace.
        let prev = match first_blk {
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
        self.frames.last_mut().unwrap().funcidx = Some(self.cp_block.as_ref().unwrap().funcidx());
        // This unwrap can't fail. If it does that means the tracer has given us a mappable block
        // that doesn't exist in the AOT module.
        self.create_trace_header(self.aot_mod.bblock(self.cp_block.as_ref().unwrap()))?;

        // FIXME: this section of code needs to be refactored.
        let mut last_blk_is_return = false;
        let mut prev_bid = None;
        while let Some(tblk) = trace_iter.next() {
            match tblk {
                Ok(b) => {
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
                                if self.aot_mod.bblock(&bid).is_return() {
                                    last_blk_is_return = true;
                                }
                            }

                            // In order to emit guards for conditional branches we need to peek at the next
                            // block.
                            let nextbb = if let Some(tpeek) = trace_iter.peek() {
                                match tpeek {
                                    Ok(tp) => self.lookup_aot_block(tp),
                                    Err(e) => match e {
                                        AOTTraceIteratorError::TraceTooLong => {
                                            return Err(CompilationError::LimitExceeded(
                                                "Trace too long.".into(),
                                            ));
                                        }
                                        AOTTraceIteratorError::LongJmpEncountered => {
                                            return Err(CompilationError::General(
                                                "Long jump encountered.".into(),
                                            ));
                                        }
                                    },
                                }
                            } else {
                                None
                            };
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
                Err(e) => match e {
                    AOTTraceIteratorError::TraceTooLong => {
                        return Err(CompilationError::LimitExceeded("Trace too long.".into()));
                    }
                    AOTTraceIteratorError::LongJmpEncountered => {
                        return Err(CompilationError::General("Long jump encountered.".into()));
                    }
                },
            }
        }
        Ok(self.jit_mod)
    }
}

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn build(
    ctr_id: u64,
    aot_mod: &Module,
    ta_iter: Box<dyn AOTTraceIterator>,
) -> Result<jit_ir::Module, CompilationError> {
    TraceBuilder::new(ctr_id, aot_mod)?.build(ta_iter)
}
