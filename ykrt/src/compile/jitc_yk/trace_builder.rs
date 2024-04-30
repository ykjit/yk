//! The trace builder.

use super::aot_ir::{self, AotIRDisplay, BBlockId, FuncIdx, InstrIdx, Module};
use super::jit_ir;
use crate::compile::CompilationError;
use crate::trace::{AOTTraceIterator, TraceAction};
use std::{collections::HashMap, ffi::CString};

/// The argument index of the trace inputs struct in the trace function.
const TRACE_FUNC_CTRLP_ARGIDX: u16 = 0;

/// A TraceBuilder frame. Keeps track of inlined calls and stores information about the last
/// processed stackmap, call instruction and its arguments.
struct Frame<'a> {
    // Index of the function of this frame.
    func_idx: Option<FuncIdx>,
    /// Stackmap for this frame.
    sm: Option<&'a aot_ir::Instruction>,
    /// JIT arguments of this frame's caller.
    args: Vec<jit_ir::Operand>,
}

impl<'a> Frame<'a> {
    fn new(
        func_idx: Option<FuncIdx>,
        sm: Option<&'a aot_ir::Instruction>,
        args: Vec<jit_ir::Operand>,
    ) -> Frame<'a> {
        Frame { func_idx, sm, args }
    }
}

/// Given a mapped trace and an AOT module, assembles an in-memory Yk IR trace by copying basic
/// blocks from the AOT IR. The output of this process will be the input to the code generator.
pub(crate) struct TraceBuilder<'a> {
    /// The AOR IR.
    aot_mod: &'a Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    /// Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstructionID, jit_ir::InstIdx>,
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
            // We have to set the func_idx to None here as we don't know what it is yet. We'll
            // update it as soon as we do.
            frames: vec![Frame::new(None, None, vec![])],
            outline_target_blk: None,
            recursion_count: 0,
        })
    }

    // Given a mapped block, find the AOT block ID, or return `None` if it is unmapped.
    fn lookup_aot_block(&self, tb: &TraceAction) -> Option<aot_ir::BBlockId> {
        match tb {
            TraceAction::MappedAOTBBlock { func_name, bb } => {
                let func_name = func_name.to_str().unwrap(); // safe: func names are valid UTF-8.
                let func = self.aot_mod.func_idx(func_name);
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
        let mut inst_iter = blk.instrs.iter().enumerate().rev();
        while let Some((_, inst)) = inst_iter.next() {
            // Is it a call to the control point? If so, extract the live vars struct.
            if let Some(tis) = inst.control_point_call_trace_inputs(self.aot_mod) {
                trace_inputs = Some(tis.to_instr(self.aot_mod));
                // Add the trace input argument to the local map so it can be tracked and
                // deoptimised.
                self.local_map
                    .insert(tis.to_instr_id(), self.next_instr_id()?);
                let arg = jit_ir::Inst::Arg(TRACE_FUNC_CTRLP_ARGIDX);
                self.jit_mod.push(arg)?;
                break;
            }
        }

        // If this unwrap fails, we didn't find the call to the control point and something is
        // profoundly wrong with the AOT IR.
        let trace_inputs = trace_inputs.unwrap();
        let trace_input_struct_ty = match trace_inputs {
            aot_ir::Instruction::Alloca { type_idx, .. } => {
                match self.aot_mod.type_(*type_idx) {
                    aot_ir::Type::Struct(x) => x,
                    _ => panic!(), // IR malformed.
                }
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
        while let Some((inst_idx, inst)) = inst_iter.next() {
            match inst {
                aot_ir::Instruction::Store { val, .. } => last_store_ptr = Some(val),
                aot_ir::Instruction::PtrAdd { ptr, .. } => {
                    // Is the pointer operand of this PtrAdd targeting the trace inputs?
                    if trace_inputs.ptr_eq(ptr.to_instr(self.aot_mod)) {
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
                        let aot_field_ty = trace_input_struct_ty.field_type_idx(trace_input_idx);
                        // FIXME: we should check at compile-time that this will fit.
                        match u32::try_from(aot_field_off) {
                            Ok(u32_off) => {
                                let input_ty_idx =
                                    self.handle_type(self.aot_mod.type_(aot_field_ty))?;
                                let load_ti_instr =
                                    jit_ir::LoadTraceInputInst::new(u32_off, input_ty_idx).into();
                                // If this take fails, we didn't see a corresponding store and the
                                // IR is malformed.
                                self.local_map.insert(
                                    last_store_ptr.take().unwrap().to_instr_id(),
                                    self.next_instr_id()?,
                                );
                                self.jit_mod.push(load_ti_instr)?;
                                self.first_ti_idx = inst_idx;
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
        bid: aot_ir::BBlockId,
        nextbb: Option<aot_ir::BBlockId>,
    ) -> Result<(), CompilationError> {
        // unwrap safe: can't trace a block not in the AOT module.
        let blk = self.aot_mod.bblock(&bid);

        // Decide how to translate each AOT instruction.
        for (inst_idx, inst) in blk.instrs.iter().enumerate() {
            match inst {
                aot_ir::Instruction::Br { .. } => Ok(()),
                aot_ir::Instruction::Load { ptr, type_idx } => {
                    self.handle_load(&bid, inst_idx, ptr, type_idx)
                }
                // FIXME: ignore remaining instructions after a call.
                aot_ir::Instruction::Call { callee, args } => {
                    // Get the branch instruction of this block.
                    let nextinst = blk.instrs.last().unwrap();
                    self.handle_call(inst, &bid, inst_idx, callee, args, nextinst)
                }
                aot_ir::Instruction::Store { val, ptr } => {
                    self.handle_store(&bid, inst_idx, val, ptr)
                }
                aot_ir::Instruction::PtrAdd { ptr, off, .. } => {
                    if self.cp_block.as_ref() == Some(&bid) && inst_idx == self.first_ti_idx {
                        // We've reached the trace inputs part of the control point block. There's
                        // no point in copying these instructions over and we can just skip to the
                        // next block.
                        return Ok(());
                    }
                    self.handle_ptradd(&bid, inst_idx, ptr, off)
                }
                aot_ir::Instruction::BinaryOp {
                    lhs,
                    binop: aot_ir::BinOp::Add,
                    rhs,
                } => self.handle_add(&bid, inst_idx, lhs, rhs),
                aot_ir::Instruction::BinaryOp { binop, .. } => {
                    todo!("{binop:?}");
                }
                aot_ir::Instruction::ICmp { lhs, pred, rhs, .. } => {
                    self.handle_icmp(&bid, inst_idx, lhs, pred, rhs)
                }
                aot_ir::Instruction::CondBr { cond, true_bb, .. } => {
                    let sm = &blk.instrs[InstrIdx::new(inst_idx - 1)];
                    debug_assert!(sm.is_safepoint());
                    self.handle_condbr(sm, nextbb.as_ref().unwrap(), cond, true_bb)
                }
                aot_ir::Instruction::Cast {
                    cast_kind,
                    val,
                    dest_type_idx,
                } => self.handle_cast(&bid, inst_idx, cast_kind, val, dest_type_idx),
                aot_ir::Instruction::Ret { val } => self.handle_ret(&bid, inst_idx, val),
                aot_ir::Instruction::DeoptSafepoint { .. } => Ok(()),
                aot_ir::Instruction::Switch {
                    test_val,
                    default_dest,
                    case_values,
                    case_dests,
                } => {
                    let sm = &blk.instrs[InstrIdx::new(inst_idx - 1)];
                    debug_assert!(sm.is_safepoint());
                    self.handle_switch(
                        &bid,
                        inst_idx,
                        sm,
                        nextbb.as_ref().unwrap(),
                        test_val,
                        default_dest,
                        case_values,
                        case_dests,
                    )
                }
                _ => todo!("{:?}", inst),
            }?;
        }
        Ok(())
    }

    fn copy_instruction(
        &mut self,
        jit_inst: jit_ir::Inst,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        // If the AOT instruction defines a new value, then add it to the local map.
        if jit_inst.def_type(&self.jit_mod).is_some() {
            let aot_iid = aot_ir::InstructionID::new(
                bid.func_idx(),
                bid.bb_idx(),
                aot_ir::InstrIdx::new(aot_inst_idx),
            );
            self.local_map.insert(aot_iid, self.next_instr_id()?);
        }

        // Insert the newly-translated instruction into the JIT module.
        self.jit_mod.push(jit_inst)?;
        Ok(())
    }

    fn next_instr_id(&self) -> Result<jit_ir::InstIdx, CompilationError> {
        jit_ir::InstIdx::new(self.jit_mod.len())
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
        self.jit_mod.global_decl_idx(&jit_global, idx)
    }

    /// Translate a constant value.
    fn handle_const(
        &mut self,
        aot_const: &aot_ir::Constant,
    ) -> Result<jit_ir::Constant, CompilationError> {
        let jit_type_idx = self.handle_type(self.aot_mod.type_(aot_const.type_idx()))?;
        Ok(jit_ir::Constant::new(
            jit_type_idx,
            Vec::from(aot_const.bytes()),
        ))
    }

    /// Translate an operand.
    fn handle_operand(
        &mut self,
        op: &aot_ir::Operand,
    ) -> Result<jit_ir::Operand, CompilationError> {
        let ret = match op {
            aot_ir::Operand::LocalVariable(iid) => {
                let instridx = self.local_map[iid];
                jit_ir::Operand::Local(instridx)
            }
            aot_ir::Operand::Constant(cidx) => {
                let jit_const = self.handle_const(self.aot_mod.constant(cidx))?;
                jit_ir::Operand::Const(self.jit_mod.const_idx(&jit_const)?)
            }
            aot_ir::Operand::Global(gd_idx) => {
                let load = jit_ir::LookupGlobalInst::new(self.handle_global(*gd_idx)?)?;
                self.jit_mod.push_and_make_operand(load.into())?
            }
            aot_ir::Operand::Arg { arg_idx, .. } => {
                // Lookup the JIT instruction that was passed into this function as an argument.
                // Unwrap is safe since an `Arg` means we are currently inlining a function.
                // FIXME: Is the above correct? What about args in the control point frame?
                self.frames.last().unwrap().args[usize::from(*arg_idx)].clone()
            }
            _ => todo!("{}", op.to_string(self.aot_mod)),
        };
        Ok(ret)
    }

    /// Translate a type.
    fn handle_type(&mut self, aot_type: &aot_ir::Type) -> Result<jit_ir::TyIdx, CompilationError> {
        let jit_ty = match aot_type {
            aot_ir::Type::Void => jit_ir::Ty::Void,
            aot_ir::Type::Integer(it) => jit_ir::Ty::Integer(jit_ir::IntegerTy::new(it.num_bits())),
            aot_ir::Type::Ptr => jit_ir::Ty::Ptr,
            aot_ir::Type::Func(ft) => {
                let mut jit_args = Vec::new();
                for aot_arg_ty_idx in ft.arg_ty_idxs() {
                    let jit_ty = self.handle_type(self.aot_mod.type_(*aot_arg_ty_idx))?;
                    jit_args.push(jit_ty);
                }
                let jit_retty = self.handle_type(self.aot_mod.type_(ft.ret_ty()))?;
                jit_ir::Ty::Func(jit_ir::FuncTy::new(jit_args, jit_retty, ft.is_vararg()))
            }
            aot_ir::Type::Struct(_st) => todo!(),
            aot_ir::Type::Unimplemented(s) => jit_ir::Ty::Unimplemented(s.to_owned()),
        };
        self.jit_mod.ty_idx(&jit_ty)
    }

    /// Translate a function.
    fn handle_func(
        &mut self,
        aot_idx: aot_ir::FuncIdx,
    ) -> Result<jit_ir::FuncDeclIdx, CompilationError> {
        let aot_func = self.aot_mod.func(aot_idx);
        let jit_func = jit_ir::FuncDecl::new(
            aot_func.name().to_owned(),
            self.handle_type(self.aot_mod.type_(aot_func.type_idx()))?,
        );
        self.jit_mod.func_decl_idx(&jit_func)
    }

    /// Translate binary operations such as add, sub, mul, etc.
    fn handle_add(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        lhs: &aot_ir::Operand,
        rhs: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let instr = jit_ir::AddInst::new(self.handle_operand(lhs)?, self.handle_operand(rhs)?);
        self.copy_instruction(instr.into(), bid, aot_inst_idx)
    }

    /// Create a guard.
    ///
    /// The guard fails if `cond` is not `expect`.
    fn create_guard(
        &mut self,
        cond: &jit_ir::Operand,
        expect: bool,
        sm: &'a aot_ir::Instruction,
    ) -> Result<jit_ir::GuardInst, CompilationError> {
        // Assign this branch's stackmap to the current frame.
        self.frames.last_mut().unwrap().sm = Some(sm);

        // Iterate over the stackmaps of the previous frames as well as the stackmap from this
        // conditional branch and collect stackmap IDs and live variables into vectors to store
        // inside the guard.
        // Unwrap safe as each frame at this point must have a stackmap associated with it.
        let mut smids = Vec::new(); // List of stackmap ids of the current call stack.
        let mut live_args = Vec::new(); // List of live JIT variables.
        for (sm, sm_args) in self.frames.iter().map(|f| (f.sm.unwrap(), &f.args)) {
            let aot_ir::Instruction::DeoptSafepoint { id, lives, .. } = sm else {
                panic!();
            };
            let aot_ir::Operand::Constant(cidx) = id else {
                panic!();
            };
            let c = self.aot_mod.constant(&cidx);
            let aot_ir::Type::Integer(ity) = self.aot_mod.const_type(c) else {
                panic!();
            };
            assert!(ity.num_bits() == u64::BITS);
            let id = u64::from_ne_bytes(c.bytes()[0..8].try_into().unwrap());
            smids.push(id);

            // Collect live variables.
            for op in lives.iter() {
                match op {
                    aot_ir::Operand::LocalVariable(iid) => {
                        live_args.push(self.local_map[iid]);
                    }
                    aot_ir::Operand::Arg { arg_idx, .. } => {
                        // Lookup the JIT value of the argument from the caller (stored in
                        // the previous frame's `args` field).
                        let jit_ir::Operand::Local(idx) = sm_args[usize::from(*arg_idx)] else {
                            panic!(); // IR malformed.
                        };
                        live_args.push(idx);
                    }
                    _ => panic!(), // IR malformed.
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
        sm: &'a aot_ir::Instruction,
        next_bb: &aot_ir::BBlockId,
        cond: &aot_ir::Operand,
        true_bb: &aot_ir::BBlockIdx,
    ) -> Result<(), CompilationError> {
        let jit_cond = self.handle_operand(cond)?;
        let guard = self.create_guard(&jit_cond, *true_bb == next_bb.bb_idx(), sm)?;
        self.jit_mod.push(guard.into())?;
        Ok(())
    }

    fn handle_ret(
        &mut self,
        _bid: &aot_ir::BBlockId,
        _aot_inst_idx: usize,
        _val: &Option<aot_ir::Operand>,
    ) -> Result<(), CompilationError> {
        // FIXME: Map return value to AOT call instruction.
        self.frames.pop();
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
        let instr =
            jit_ir::IcmpInst::new(self.handle_operand(lhs)?, *pred, self.handle_operand(rhs)?)
                .into();
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    /// Translate a `Load` instruction.
    fn handle_load(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ptr: &aot_ir::Operand,
        type_idx: &aot_ir::TypeIdx,
    ) -> Result<(), CompilationError> {
        let inst = jit_ir::LoadInst::new(
            self.handle_operand(ptr)?,
            self.handle_type(self.aot_mod.type_(*type_idx))?,
        )
        .into();
        self.copy_instruction(inst, bid, aot_inst_idx)
    }

    fn handle_call(
        &mut self,
        inst: &'a aot_ir::Instruction,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        callee: &aot_ir::FuncIdx,
        args: &[aot_ir::Operand],
        nextinst: &'a aot_ir::Instruction,
    ) -> Result<(), CompilationError> {
        // Ignore special functions that we neither want to inline nor copy.
        if inst.is_debug_call(self.aot_mod) {
            return Ok(());
        }

        // Safepoint calls should always be specialised to `DeoptSafepoint` instructions.
        debug_assert!(!inst.is_safepoint());

        // Convert AOT args to JIT args.
        let mut jit_args = Vec::new();
        for arg in args {
            jit_args.push(self.handle_operand(arg)?);
        }

        // Check if this is a recursive call by scanning the call stack for the callee.
        let is_recursive = self
            .frames
            .iter()
            .position(|f| f.func_idx == Some(*callee))
            .is_some();

        if inst.is_mappable_call(self.aot_mod)
            && !self.aot_mod.func(*callee).is_outline()
            && !is_recursive
        {
            // This is a mappable call that we want to inline.
            // Retrieve the stackmap that follows every mappable call.
            let blk = self.aot_mod.bblock(bid);
            let sm = &blk.instrs[InstrIdx::new(aot_inst_idx + 1)];
            debug_assert!(sm.is_safepoint());
            // Assign stackmap to the current frame.
            // Unwrap is safe as there's always at least one frame.
            self.frames.last_mut().unwrap().sm = Some(sm);
            // Create a new frame for the inlined call and pass in the arguments of the caller.
            self.frames.push(Frame::new(Some(*callee), None, jit_args));
            Ok(())
        } else {
            // This call can't be inlined. It is either unmappable (a declaration or an indirect
            // call) or the compiler annotated it with `yk_outline`.
            match nextinst {
                aot_ir::Instruction::Br { succ } => {
                    // We can only stop outlining when we see the succesor block and we are not in
                    // the middle of recursion.
                    let succbid = BBlockId::new(bid.func_idx(), *succ);
                    self.outline_target_blk = Some(succbid);
                    self.recursion_count = 0;
                }
                aot_ir::Instruction::CondBr { .. } => {
                    // Currently, the successor of a call is always an unconditional branch due to
                    // the block spitting pass. However, there's a FIXME in that pass which could
                    // lead to conditional branches showing up here. Leave a todo here so we know
                    // when this happens.
                    todo!()
                }
                _ => panic!(),
            }

            let jit_func_decl_idx = self.handle_func(*callee)?;
            let instr = if !self
                .aot_mod
                .func(*callee)
                .func_type(self.aot_mod)
                .is_vararg()
            {
                jit_ir::CallInst::new(&mut self.jit_mod, jit_func_decl_idx, &jit_args)?.into()
            } else {
                jit_ir::VACallInst::new(&mut self.jit_mod, jit_func_decl_idx, &jit_args)?.into()
            };
            self.copy_instruction(instr, bid, aot_inst_idx)
        }
    }

    fn handle_store(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        val: &aot_ir::Operand,
        ptr: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let instr =
            jit_ir::StoreInst::new(self.handle_operand(val)?, self.handle_operand(ptr)?).into();
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    fn handle_ptradd(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        ptr: &aot_ir::Operand,
        off: &aot_ir::Operand,
    ) -> Result<(), CompilationError> {
        let jit_ptr = self.handle_operand(ptr)?;
        match off {
            aot_ir::Operand::Constant(co) => {
                let c = self.aot_mod.constant(co);
                if let aot_ir::Type::Integer(it) = self.aot_mod.const_type(c) {
                    // Convert the offset into a 32 bit value, as that is the maximum we can fit into
                    // the jit_ir::PtrAddInst.
                    let offset: u32 = match it.num_bits() {
                        // This unwrap can't fail unless we did something wrong during lowering.
                        64 => u64::from_ne_bytes(c.bytes()[0..8].try_into().unwrap())
                            .try_into()
                            .map_err(|_| {
                                CompilationError::LimitExceeded("ptradd offset too big".into())
                            })?,
                        _ => panic!(),
                    };
                    let instr = jit_ir::PtrAddInst::new(jit_ptr, offset).into();
                    self.copy_instruction(instr, bid, aot_inst_idx)
                } else {
                    panic!(); // Non-integer offset. Malformed IR.
                }
            }
            _ => todo!(),
        }
    }

    fn handle_cast(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        cast_kind: &aot_ir::CastKind,
        val: &aot_ir::Operand,
        dest_type_idx: &aot_ir::TypeIdx,
    ) -> Result<(), CompilationError> {
        let instr = match cast_kind {
            aot_ir::CastKind::SignExtend => jit_ir::SignExtendInst::new(
                &self.handle_operand(val)?,
                self.handle_type(self.aot_mod.type_(*dest_type_idx))?,
            ),
        };
        self.copy_instruction(instr.into(), bid, aot_inst_idx)
    }

    fn handle_switch(
        &mut self,
        bid: &aot_ir::BBlockId,
        aot_inst_idx: usize,
        sm: &'a aot_ir::Instruction,
        next_bb: &aot_ir::BBlockId,
        test_val: &aot_ir::Operand,
        _default_dest: &aot_ir::BBlockIdx,
        case_values: &Vec<u64>,
        case_dests: &Vec<aot_ir::BBlockIdx>,
    ) -> Result<(), CompilationError> {
        if case_values.len() == 0 {
            // Degenerate switch. Not sure it can even happen.
            panic!();
        }

        // Find the JIT type of the value being tested.
        let jit_type_idx = self.handle_type(test_val.type_(self.aot_mod))?;
        let jit_int_type = match self.jit_mod.type_(jit_type_idx) {
            jit_ir::Ty::Integer(it) => it,
            _ => unreachable!(),
        }
        .to_owned();

        // Find out which case we traced.
        let guard = match case_dests.iter().position(|&cd| cd == next_bb.bb_idx()) {
            Some(cidx) => {
                // A non-default case was traced.
                let val = case_values[cidx];
                let bb = case_dests[cidx];

                // Build the constant value to guard.
                let jit_const = jit_int_type
                    .to_owned()
                    .make_constant(&mut self.jit_mod, val)?;
                let jit_const_opnd = jit_ir::Operand::Const(self.jit_mod.const_idx(&jit_const)?);

                // Perform the comparison.
                let jit_test_val = self.handle_operand(test_val)?;
                let cmp_inst =
                    jit_ir::IcmpInst::new(jit_test_val, jit_ir::Predicate::Equal, jit_const_opnd);
                let jit_cond = self.jit_mod.push_and_make_operand(cmp_inst.into())?;

                // Guard the result of the comparison.
                self.create_guard(&jit_cond, bb == next_bb.bb_idx(), sm)?
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
                    let jit_const = jit_int_type
                        .to_owned()
                        .make_constant(&mut self.jit_mod, *cv)?;
                    let jit_const_opnd =
                        jit_ir::Operand::Const(self.jit_mod.const_idx(&jit_const)?);

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
                        let and = jit_ir::OrInst::new(lhs, cmp);
                        jit_cond = Some(self.jit_mod.push_and_make_operand(and.into())?);
                    }
                }

                // Guard the result of ORing all the comparisons together.
                // unwrap can't fail: we already disregarded degenerate switches with no
                // non-default cases.
                self.create_guard(&jit_cond.unwrap(), false, sm)?
            }
        };
        self.copy_instruction(guard.into(), bid, aot_inst_idx)
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
        self.frames.last_mut().unwrap().func_idx = Some(self.cp_block.as_ref().unwrap().func_idx());
        // This unwrap can't fail. If it does that means the tracer has given us a mappable block
        // that doesn't exist in the AOT module.
        self.create_trace_header(self.aot_mod.bblock(self.cp_block.as_ref().unwrap()))?;

        let mut last_blk_is_return = false;
        while let Some(tblk) = trace_iter.next() {
            match tblk {
                Ok(b) => {
                    match self.lookup_aot_block(&b) {
                        Some(bid) => {
                            // MappedAOTBBlock block
                            if let Some(ref tgtbid) = self.outline_target_blk {
                                // We are currently outlining.
                                if bid.func_idx() == tgtbid.func_idx() {
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
                                    Err(_) => todo!(),
                                }
                            } else {
                                None
                            };
                            self.process_block(bid, nextbb)?;
                        }
                        None => {
                            // UnmappableBBlock block
                        }
                    }
                }
                Err(_) => todo!(),
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
