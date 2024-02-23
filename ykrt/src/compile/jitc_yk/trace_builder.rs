//! The trace builder.

use super::aot_ir::{self, IRDisplay, Module};
use super::jit_ir;
use crate::compile::CompilationError;
use crate::trace::{AOTTraceIterator, TraceAction};
use std::collections::HashMap;

/// The argument index of the trace inputs struct in the control point call.
const CTRL_POINT_ARGIDX_INPUTS: usize = 3;

/// Given a mapped trace and an AOT module, assembles an in-memory Yk IR trace by copying blocks
/// from the AOT IR. The output of this process will be the input to the code generator.
struct TraceBuilder<'a> {
    /// The AOR IR.
    aot_mod: &'a Module,
    /// The JIT IR this struct builds.
    jit_mod: jit_ir::Module,
    // Maps an AOT instruction to a jit instruction via their index-based IDs.
    local_map: HashMap<aot_ir::InstructionID, jit_ir::InstrIdx>,
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
        // Find trace input variables and emit `LoadArg` instructions for them.
        let mut last_store = None;
        let mut trace_input = None;
        let mut input = Vec::new();
        for inst in blk.instrs.iter().rev() {
            if inst.is_control_point(self.aot_mod) {
                trace_input = Some(inst.operand(CTRL_POINT_ARGIDX_INPUTS));
            }
            if inst.is_store() {
                last_store = Some(inst);
            }
            if inst.is_ptr_add() {
                let op = inst.operand(0);
                // unwrap safe: we know the AOT code was produced by ykllvm.
                if trace_input
                    .unwrap()
                    .to_instr(self.aot_mod)
                    .ptr_eq(op.to_instr(self.aot_mod))
                {
                    // Found a trace input.
                    // unwrap safe: we know the AOT code was produced by ykllvm.
                    let inp = last_store.unwrap().operand(0);
                    input.insert(0, inp.to_instr(self.aot_mod));
                    let load_arg = jit_ir::LoadArgInstruction::new().into();
                    self.local_map
                        .insert(inp.to_instr_id(), self.next_instr_id()?);
                    self.jit_mod.push(load_arg);
                }
            }
        }
        Ok(())
    }

    /// Walk over a traced AOT block, translating the constituent instructions into the JIT module.
    fn process_block(&mut self, bid: aot_ir::BlockID) -> Result<(), CompilationError> {
        // unwrap safe: can't trace a block not in the AOT module.
        let blk = self.aot_mod.block(&bid);

        // Decide how to translate each AOT instruction based upon its opcode.
        for (inst_idx, inst) in blk.instrs.iter().enumerate() {
            match inst.opcode() {
                aot_ir::Opcode::Br => self.handle_br(inst),
                aot_ir::Opcode::Load => self.handle_load(inst, &bid, inst_idx),
                aot_ir::Opcode::Call => self.handle_call(inst, &bid, inst_idx),
                aot_ir::Opcode::Store => self.handle_store(inst, &bid, inst_idx),
                aot_ir::Opcode::PtrAdd => self.handle_ptradd(inst, &bid, inst_idx),
                aot_ir::Opcode::Add => self.handle_binop(inst, &bid, inst_idx),
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
        Ok(self.jit_mod.global_decl_idx(aot_global)?)
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
            aot_ir::Operand::Constant(co) => {
                let c = self.aot_mod.constant(co);
                let cidx = match self.aot_mod.const_type(c) {
                    aot_ir::Type::Integer(it) => match it.num_bits() {
                        32 => {
                            // These unwraps can't fail unless we did something wrong during
                            // lowering.
                            let val = u32::from_ne_bytes(c.bytes()[0..4].try_into().unwrap())
                                .try_into()
                                .unwrap();
                            self.jit_mod.const_idx(&jit_ir::Constant::U32(val))?
                        }
                        _ => todo!(),
                    },
                    _ => todo!(),
                };
                jit_ir::Operand::Const(cidx)
            }
            aot_ir::Operand::Global(go) => {
                let load = jit_ir::LoadGlobalInstruction::new(self.handle_global(go.index())?)?;
                let idx = self.next_instr_id()?;
                self.jit_mod.push(load.into());
                jit_ir::Operand::Local(idx)
            }
            aot_ir::Operand::Unimplemented(_) => {
                // FIXME: for now we push an arbitrary constant.
                let constidx = self
                    .jit_mod
                    .const_idx(&jit_ir::Constant::Usize(0xdeadbeef))?;
                jit_ir::Operand::Const(constidx)
            }
            _ => todo!("{}", op.to_str(self.aot_mod)),
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
            aot_ir::Type::Integer(it) => jit_ir::Type::Integer(it.clone()),
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

    /// Translate a `Br` instruction.
    fn handle_br(&mut self, inst: &aot_ir::Instruction) -> Result<(), CompilationError> {
        if inst.operands_len() == 0 {
            // Unconditional branch.
            Ok(())
        } else {
            // Conditional branches require guards.
            todo!()
        }
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
        let instr = if let aot_ir::Operand::Global(go) = aot_op {
            // Generate a special load instruction for globals.
            jit_ir::LoadGlobalInstruction::new(self.handle_global(go.index())?)?.into()
        } else {
            let jit_op = self.handle_operand(aot_op)?;
            let jit_ty = self.handle_type(aot_ty)?;
            jit_ir::LoadInstruction::new(jit_op, jit_ty).into()
        };
        self.copy_instruction(instr, bid, aot_inst_idx)
    }

    fn handle_call(
        &mut self,
        inst: &aot_ir::Instruction,
        bid: &aot_ir::BlockID,
        aot_inst_idx: usize,
    ) -> Result<(), CompilationError> {
        if inst.is_stackmap_call(self.aot_mod) || inst.is_debug_call(self.aot_mod) {
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
        let instr = if let aot_ir::Operand::Global(go) = inst.operand(1) {
            // Generate a special store instruction for globals.
            jit_ir::StoreGlobalInstruction::new(val, self.handle_global(go.index())?)?.into()
        } else {
            let ptr = self.handle_operand(inst.operand(1))?;
            jit_ir::StoreInstruction::new(val, ptr).into()
        };
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
                        .map_err(|_| {
                            CompilationError::Unrecoverable("ptradd offset too big".into())
                        }),
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
            None => return Err(CompilationError::Unrecoverable("empty trace".into())),
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

        let first_blk = self.lookup_aot_block(&prev);
        self.create_trace_header(self.aot_mod.block(&first_blk.unwrap()))?;

        for tblk in ta_iter {
            match tblk {
                Ok(b) => {
                    match self.lookup_aot_block(&b) {
                        Some(bid) => {
                            // MappedAOTBlock block
                            self.process_block(bid)?;
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
