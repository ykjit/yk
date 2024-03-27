//! The trace builder.

use super::aot_ir::{self, AotIRDisplay, InstrIdx, Module};
use super::jit_ir;
use crate::compile::CompilationError;
use crate::trace::{AOTTraceIterator, TraceAction};
use std::{collections::HashMap, ffi::CString};

/// The argument index of the trace inputs struct in the control point call.
const CTRL_POINT_ARGIDX_INPUTS: usize = 3;
/// The argument index of the trace inputs struct in the trace function.
const TRACE_FUNC_CTRLP_ARGIDX: u16 = 0;

/// Given a mapped trace and an AOT module, assembles an in-memory Yk IR trace by copying blocks
/// from the AOT IR. The output of this process will be the input to the code generator.
struct TraceBuilder<'a> {
    /// The AOR IR.
    aot_mod: &'a Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    /// Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstructionID, jit_ir::InstrIdx>,
    // Block containing the current control point (i.e. the control point that started this trace).
    cp_block: Option<aot_ir::BlockID>,
    // Index of the first traceinput instruction.
    first_ti_idx: usize,
}

impl<'a> TraceBuilder<'a> {
    /// Create a trace builder.
    ///
    /// Arguments:
    ///  - `trace_name`: The eventual symbol name for the JITted code.
    ///  - `aot_mod`: The AOT IR module that the trace flows through.
    ///  - `mtrace`: The mapped trace.
    fn new(trace_name: String, aot_mod: &'a Module) -> Self {
        Self {
            aot_mod,
            jit_mod: jit_ir::Module::new(trace_name),
            local_map: HashMap::new(),
            cp_block: None,
            first_ti_idx: 0,
        }
    }

    // Given a mapped block, find the AOT block ID, or return `None` if it is unmapped.
    fn lookup_aot_block(&self, tb: &TraceAction) -> Option<aot_ir::BlockID> {
        match tb {
            TraceAction::MappedAOTBlock { func_name, bb } => {
                let func_name = func_name.to_str().unwrap(); // safe: func names are valid UTF-8.
                let func = self.aot_mod.func_idx(func_name);
                Some(aot_ir::BlockID::new(func, aot_ir::BlockIdx::new(*bb)))
            }
            TraceAction::UnmappableBlock { .. } => None,
            TraceAction::Promotion => todo!(),
        }
    }

    /// Create the prolog of the trace.
    fn create_trace_header(&mut self, blk: &aot_ir::Block) -> Result<(), CompilationError> {
        // Find trace input variables and emit `LoadTraceInput` instructions for them.
        let mut trace_inputs = None;

        // PHASE 1:
        // Find the control point call and extract the trace inputs struct from its operands.
        let mut inst_iter = blk.instrs.iter().enumerate().rev();
        for (_, inst) in &mut inst_iter {
            if inst.is_control_point(self.aot_mod) {
                let op = inst.operand(CTRL_POINT_ARGIDX_INPUTS);
                trace_inputs = Some(op.to_instr(self.aot_mod));
                // Add the trace input argument to the local map so it can be tracked and
                // deoptimised.
                self.local_map
                    .insert(op.to_instr_id(), self.next_instr_id()?);
                let arg = jit_ir::Instruction::Arg(TRACE_FUNC_CTRLP_ARGIDX);
                self.jit_mod.push(arg);
                break;
            }
        }

        // If this unwrap fails, we didn't find the call to the control point and something is
        // profoundly wrong with the AOT IR.
        let trace_inputs = trace_inputs.unwrap();
        let trace_input_struct_ty = trace_inputs.operand(0).type_(self.aot_mod).as_struct();
        // We visit the trace inputs in reverse order, so we start with high indices first. This
        // value then decrements in the below loop.
        let mut trace_input_idx = trace_input_struct_ty.num_fields();

        // PHASE 2:
        // Keep walking backwards over the ptradd/store pairs emitting LoadTraceInput instructions.
        let mut last_store = None;
        for (inst_idx, inst) in inst_iter {
            if inst.is_store() {
                last_store = Some(inst);
            }
            if inst.is_ptr_add() {
                // Is the pointer operand of this PtrAdd targeting the trace inputs?
                if trace_inputs.ptr_eq(inst.operand(0).to_instr(self.aot_mod)) {
                    // We found a trace input. Now we emit a `LoadTraceInput` instruction into the
                    // trace. This assigns the input to a local variable that other instructions
                    // can then use.
                    //
                    // Note: This code assumes that the `PtrAdd` instructions in the AOT IR were
                    // emitted sequentially.
                    //
                    // unwrap safe: we know the AOT code was produced by ykllvm.
                    let trace_input_val = last_store.unwrap().operand(0);
                    trace_input_idx -= 1;
                    // FIXME: assumes the field is byte-aligned. If it isn't, field_byte_off() will
                    // crash.
                    let aot_field_off = trace_input_struct_ty.field_byte_off(trace_input_idx);
                    let aot_field_ty = trace_input_struct_ty.field_type_idx(trace_input_idx);
                    match u32::try_from(aot_field_off) {
                        Ok(u32_off) => {
                            let input_ty_idx = self.handle_type(aot_field_ty)?;
                            let load_arg =
                                jit_ir::LoadTraceInputInstruction::new(u32_off, input_ty_idx)
                                    .into();
                            self.local_map
                                .insert(trace_input_val.to_instr_id(), self.next_instr_id()?);
                            self.jit_mod.push(load_arg);
                            self.first_ti_idx = inst_idx;
                        }
                        _ => {
                            return Err(CompilationError("offset doesn't fit".into()));
                        }
                    }
                }
            }
        }

        // Mark this location as the start of the trace loop.
        self.jit_mod.push(jit_ir::Instruction::TraceLoopStart);

        Ok(())
    }

    /// Walk over a traced AOT block, translating the constituent instructions into the JIT module.
    fn process_block(
        &mut self,
        bid: aot_ir::BlockID,
        nextbb: Option<aot_ir::BlockID>,
    ) -> Result<(), CompilationError> {
        // unwrap safe: can't trace a block not in the AOT module.
        let blk = self.aot_mod.block(&bid);

        // Decide how to translate each AOT instruction based upon its opcode.
        for (inst_idx, inst) in blk.instrs.iter().enumerate() {
            match inst.opcode() {
                aot_ir::Opcode::Br => Ok(()),
                aot_ir::Opcode::Load => self.handle_load(inst, &bid, inst_idx),
                aot_ir::Opcode::Call => self.handle_call(inst, &bid, inst_idx),
                aot_ir::Opcode::Store => self.handle_store(inst, &bid, inst_idx),
                aot_ir::Opcode::PtrAdd => {
                    if self.cp_block.as_ref() == Some(&bid) && inst_idx == self.first_ti_idx {
                        // We've reached the trace inputs part of the control point block. There's
                        // no point in copying these instructions over and we can just skip to the
                        // next block.
                        return Ok(());
                    }
                    self.handle_ptradd(inst, &bid, inst_idx)
                }
                aot_ir::Opcode::Add => self.handle_binop(inst, &bid, inst_idx),
                aot_ir::Opcode::Icmp => self.handle_icmp(inst, &bid, inst_idx),
                aot_ir::Opcode::CondBr => {
                    let sm = &blk.instrs[InstrIdx::new(inst_idx - 1)];
                    debug_assert!(sm.is_stackmap_call(self.aot_mod));
                    self.handle_condbr(inst, sm, nextbb.as_ref().unwrap())
                }
                _ => todo!("{:?}", inst),
            }?;
        }
        Ok(())
    }

    fn copy_instruction(
        &mut self,
        jit_inst: jit_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        // If the AOT instruction defines a new value, then add it to the local map.
        if jit_inst.is_def() {
            let aot_iid = aot_ir::InstructionID::new(
                bid.func_idx(),
                bid.block_idx(),
                aot_ir::InstrIdx::new(aot_inst_idx),
            );
            self.local_map.insert(aot_iid, self.next_instr_id()?);
        }

        // Insert the newly-translated instruction into the JIT module.
        self.jit_mod.push(jit_inst);
        Ok(())
    }

    fn next_instr_id(&self) -> Result<jit_ir::InstrIdx, CompilationError> {
        jit_ir::InstrIdx::new(self.jit_mod.len())
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
        Ok(self.jit_mod.global_decl_idx(&jit_global, idx)?)
    }

    /// Translate a constant value.
    fn handle_const(
        &mut self,
        aot_const: &aot_ir::Constant,
    ) -> Result<jit_ir::Constant, CompilationError> {
        let jit_type_idx = self.handle_type(aot_const.type_idx())?;
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
            aot_ir::Operand::LocalVariable(lvo) => {
                let instridx = self.local_map[lvo.instr_id()];
                jit_ir::Operand::Local(instridx)
            }
            aot_ir::Operand::Constant(cidx) => {
                let jit_const = self.handle_const(self.aot_mod.constant(cidx))?;
                jit_ir::Operand::Const(self.jit_mod.const_idx(&jit_const)?)
            }
            aot_ir::Operand::Global(go) => {
                let load = jit_ir::LookupGlobalInstruction::new(self.handle_global(go.index())?)?;
                self.jit_mod.push_and_make_operand(load.into())?
            }
            aot_ir::Operand::Unimplemented(_) => {
                // FIXME: for now we push an arbitrary constant.
                use std::mem;
                let jit_const =
                    jit_ir::IntegerType::new(u32::try_from(mem::size_of::<isize>()).unwrap())
                        .make_constant(&mut self.jit_mod, 0xdeadbeef as isize)?;
                let const_idx = self.jit_mod.const_idx(&jit_const)?;
                jit_ir::Operand::Const(const_idx)
            }
            _ => todo!("{}", op.to_string(self.aot_mod)),
        };
        Ok(ret)
    }

    /// Translate a type.
    fn handle_type(
        &mut self,
        aot_idx: aot_ir::TypeIdx,
    ) -> Result<jit_ir::TypeIdx, CompilationError> {
        let jit_ty = match self.aot_mod.type_(aot_idx) {
            aot_ir::Type::Void => jit_ir::Type::Void,
            aot_ir::Type::Integer(it) => {
                jit_ir::Type::Integer(jit_ir::IntegerType::new(it.num_bits()))
            }
            aot_ir::Type::Ptr => jit_ir::Type::Ptr,
            aot_ir::Type::Func(ft) => {
                let mut jit_args = Vec::new();
                for aot_arg_ty_idx in ft.arg_ty_idxs() {
                    let jit_ty = self.handle_type(*aot_arg_ty_idx)?;
                    jit_args.push(jit_ty);
                }
                let jit_retty = self.handle_type(ft.ret_ty())?;
                jit_ir::Type::Func(jit_ir::FuncType::new(jit_args, jit_retty, ft.is_vararg()))
            }
            aot_ir::Type::Struct(_st) => todo!(),
            aot_ir::Type::Unimplemented(s) => jit_ir::Type::Unimplemented(s.to_owned()),
        };
        self.jit_mod.type_idx(&jit_ty)
    }

    /// Translate a function.
    fn handle_func(
        &mut self,
        aot_idx: aot_ir::FuncIdx,
    ) -> Result<jit_ir::FuncDeclIdx, CompilationError> {
        let aot_func = self.aot_mod.func(aot_idx);
        let jit_func = jit_ir::FuncDecl::new(
            aot_func.name().to_owned(),
            self.handle_type(aot_func.type_idx())?,
        );
        self.jit_mod.func_decl_idx(&jit_func)
    }

    /// Translate binary operations such as add, sub, mul, etc.
    fn handle_binop(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        let op1 = self.handle_operand(inst.operand(0))?;
        let op2 = self.handle_operand(inst.operand(1))?;
        let instr = match inst.opcode() {
            aot_ir::Opcode::Add => jit_ir::AddInstruction::new(op1, op2),
            _ => todo!(),
        };
        self.copy_instruction(instr.into(), bid, aot_inst_idx)
    }

    /// Translate a conditional `Br` instruction.
    fn handle_condbr(
        &mut self,
        inst: &aot_ir::Instruction,
        sm: &aot_ir::Instruction,
        nextbb: &aot_ir::BlockID,
    ) -> Result<(), CompilationError> {
        let cond = match &inst.operand(0) {
            aot_ir::Operand::LocalVariable(aot_ir::LocalVariableOperand(iid)) => {
                self.local_map[iid]
            }
            _ => panic!(),
        };
        let succ_bb = match inst.operand(1) {
            aot_ir::Operand::Block(aot_ir::BlockOperand { bb_idx }) => bb_idx,
            _ => panic!(),
        };

        // Find live variables in the current stack frame and add them into the guard.
        // FIXME: Once we handle mappable calls we need to push the live variables of other stack
        // frames too.
        let mut smids = Vec::new(); // List of stackmap ids of the current call stack.
        let mut lives = Vec::new(); // List of live JIT variables.

        match sm.operand(1) {
            aot_ir::Operand::Constant(co) => {
                let c = self.aot_mod.constant(co);
                match self.aot_mod.const_type(c) {
                    aot_ir::Type::Integer(it) => {
                        let id: u64 = match it.num_bits() {
                            // This unwrap can't fail unless we did something wrong during lowering.
                            64 => u64::from_ne_bytes(c.bytes()[0..8].try_into().unwrap()),
                            _ => panic!(),
                        };
                        smids.push(id);
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
        for op in sm.remaining_operands(3) {
            match op {
                aot_ir::Operand::LocalVariable(lvo) => {
                    lives.push(self.local_map[&lvo.0]);
                }
                _ => panic!(),
            }
        }

        let gi = jit_ir::GuardInfo::new(smids, lives);
        let gi_idx = self.jit_mod.push_guardinfo(gi)?;
        let expect = *succ_bb == nextbb.block_idx();
        let guard = jit_ir::GuardInstruction::new(jit_ir::Operand::Local(cond), expect, gi_idx);
        Ok(self.jit_mod.push(guard.into()))
    }

    /// Translate a `Icmp` instruction.
    fn handle_icmp(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        let op1 = self.handle_operand(inst.operand(0))?;
        let pred = match inst.operand(1) {
            aot_ir::Operand::Predicate(p) => p,
            _ => panic!(), // second operand must be a predicate.
        };
        let op2 = self.handle_operand(inst.operand(2))?;
        let instr = jit_ir::IcmpInstruction::new(op1, *pred, op2).into();
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    /// Translate a `Load` instruction.
    fn handle_load(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        let aot_op = inst.operand(0);
        let aot_ty = inst.type_idx();
        let jit_op = self.handle_operand(aot_op)?;
        let jit_ty = self.handle_type(aot_ty)?;
        let instr = jit_ir::LoadInstruction::new(jit_op, jit_ty).into();
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    fn handle_call(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        if inst.is_stackmap_call(self.aot_mod) || inst.is_debug_call(self.aot_mod) {
            // FIXME: If this is a mappable call that we are going to inline, push its stackmap
            // onto the callstack.
            return Ok(());
        }
        let mut args = Vec::new();
        for arg in inst.remaining_operands(1) {
            args.push(self.handle_operand(arg)?);
        }
        let jit_func_decl_idx = self.handle_func(inst.callee())?;
        let instr =
            jit_ir::CallInstruction::new(&mut self.jit_mod, jit_func_decl_idx, &args)?.into();
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    fn handle_store(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        let val = self.handle_operand(inst.operand(0))?;
        let ptr = self.handle_operand(inst.operand(1))?;
        let instr = jit_ir::StoreInstruction::new(val, ptr).into();
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    fn handle_ptradd(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        let target = self.handle_operand(inst.operand(0))?;
        if let aot_ir::Operand::Constant(co) = inst.operand(1) {
            let c = self.aot_mod.constant(co);
            if let aot_ir::Type::Integer(it) = self.aot_mod.const_type(c) {
                // Convert the offset into a 32 bit value, as that is the maximum we can fit into
                // the jit_ir::PtrAddInstruction.
                let offset: u32 = match it.num_bits() {
                    // This unwrap can't fail unless we did something wrong during lowering.
                    64 => u64::from_ne_bytes(c.bytes()[0..8].try_into().unwrap())
                        .try_into()
                        .map_err(|_| CompilationError("ptradd offset too big".into())),
                    _ => panic!(),
                }?;
                let instr = jit_ir::PtrAddInstruction::new(target, offset).into();
                return self.copy_instruction(instr, bid, aot_inst_idx);
            };
        }
        panic!()
    }

    /// Entry point for building an IR trace.
    ///
    /// Consumes the trace builder, returning a JIT module.
    fn build(
        mut self,
        mut ta_iter: Box<dyn AOTTraceIterator>,
    ) -> Result<jit_ir::Module, CompilationError> {
        let first_blk = match ta_iter.next() {
            Some(Ok(b)) => b,
            Some(Err(_)) => todo!(),
            None => return Err(CompilationError("empty trace".into())),
        };

        // Find the block containing the control point call. This is the (sole) predecessor of the
        // first (guaranteed mappable) block in the trace.
        let prev = match first_blk {
            TraceAction::MappedAOTBlock { func_name, bb } => {
                debug_assert!(bb > 0);
                // It's `- 1` due to the way the ykllvm block splitting pass works.
                TraceAction::MappedAOTBlock {
                    func_name: func_name.clone(),
                    bb: bb - 1,
                }
            }
            TraceAction::UnmappableBlock => panic!(),
            TraceAction::Promotion => todo!(),
        };

        self.cp_block = self.lookup_aot_block(&prev);
        // This unwrap can't fail. If it does that means the tracer has given us a mappable block
        // that doesn't exist in the AOT module.
        self.create_trace_header(self.aot_mod.block(self.cp_block.as_ref().unwrap()))?;
        let mut trace_iter = ta_iter.peekable();
        while let Some(tblk) = trace_iter.next() {
            match tblk {
                Ok(b) => {
                    match self.lookup_aot_block(&b) {
                        Some(bid) => {
                            // MappedAOTBlock block
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
                            // UnmappableBlock block
                            // Ignore for now. May be later used to make sense of the control flow. Though
                            // ideally we remove unmappable blocks from the trace so we can handle software
                            // and hardware traces the same.
                        }
                    }
                }
                Err(_) => todo!(),
            }
        }
        Ok(self.jit_mod)
    }
}

/// Given a mapped trace (through `aot_mod`), assemble and return a Yk IR trace.
pub(super) fn build(
    aot_mod: &Module,
    ta_iter: Box<dyn AOTTraceIterator>,
) -> Result<jit_ir::Module, CompilationError> {
    // FIXME: the XXX below should be a thread-safe monotonically incrementing integer.
    TraceBuilder::new("__yk_compiled_trace_XXX".into(), aot_mod).build(ta_iter)
}
