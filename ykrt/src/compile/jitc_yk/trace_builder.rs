//! The trace builder.

use super::aot_ir::{self, IRDisplay, Module};
use super::jit_ir;
use crate::compile::CompilationError;
use crate::trace::TracedAOTBlock;
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
    /// The mapped trace.
    mtrace: &'a Vec<TracedAOTBlock>,
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
    fn new(trace_name: String, aot_mod: &'a Module, mtrace: &'a Vec<TracedAOTBlock>) -> Self {
        Self {
            aot_mod,
            mtrace,
            jit_mod: jit_ir::Module::new(trace_name),
            local_map: HashMap::new(),
        }
    }

    // Given a mapped block, find the AOT block ID, or return `None` if it is unmapped.
    fn lookup_aot_block(&self, tb: &TracedAOTBlock) -> Option<aot_ir::BlockID> {
        match tb {
            TracedAOTBlock::Mapped { func_name, bb } => {
                let func_name = func_name.to_str().unwrap(); // safe: func names are valid UTF-8.
                let func = self.aot_mod.func_idx(func_name);
                Some(aot_ir::BlockID::new(func, aot_ir::BlockIdx::new(*bb)))
            }
            TracedAOTBlock::Unmappable { .. } => None,
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
            if inst.is_gep() {
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
            let jit_inst = match inst.opcode() {
                aot_ir::Opcode::Load => self.handle_load(inst),
                aot_ir::Opcode::Call => self.handle_call(inst),
                _ => todo!("{:?}", inst),
            }?;

            // If the AOT instruction defines a new value, then add it to the local map.
            if jit_inst.is_def() {
                let aot_iid = aot_ir::InstructionID::new(
                    bid.func_idx(),
                    bid.block_idx(),
                    aot_ir::InstrIdx::new(inst_idx),
                );
                self.local_map.insert(aot_iid, self.next_instr_id()?);
            }

            // Insert the newly-translated instruction into the JIT module.
            self.jit_mod.push(jit_inst);
        }
        Ok(())
    }

    fn next_instr_id(&self) -> Result<jit_ir::InstrIdx, CompilationError> {
        jit_ir::InstrIdx::new(self.jit_mod.len())
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

    /// Translate a `Load` instruction.
    fn handle_load(
        &mut self,
        inst: &aot_ir::Instruction,
    ) -> Result<jit_ir::Instruction, CompilationError> {
        let jit_op = self.handle_operand(inst.operand(0))?;
        Ok(
            jit_ir::LoadInstruction::new(jit_op, jit_ir::TypeIdx::from_aot(inst.type_idx())?)
                .into(),
        )
    }

    fn handle_call(
        &mut self,
        inst: &aot_ir::Instruction,
    ) -> Result<jit_ir::Instruction, CompilationError> {
        let mut args = Vec::new();
        for arg in inst.remaining_operands(1) {
            args.push(self.handle_operand(arg)?);
        }
        Ok(jit_ir::CallInstruction::new(&mut self.jit_mod, inst.callee(), &args)?.into())
    }

    /// Entry point for building an IR trace.
    ///
    /// Consumes the trace builder, returning a JIT module.
    fn build(mut self) -> Result<jit_ir::Module, CompilationError> {
        let first_blk = match self.mtrace.get(0) {
            Some(b) => Ok(b),
            None => Err(CompilationError::Unrecoverable("empty trace".into())),
        }?;
        let firstblk = self.lookup_aot_block(first_blk);
        // FIXME: This unwrap assumes the first block is mappable, but Laurie just merged a change
        // that strips the initial block (the block we return to from the control point), so I
        // don't think this assumption necessarily holds any more. Investigate.
        self.create_trace_header(self.aot_mod.block(&firstblk.unwrap()))?;

        for tblk in self.mtrace {
            match self.lookup_aot_block(tblk) {
                Some(bid) => {
                    // Mapped block
                    self.process_block(bid)?;
                }
                None => {
                    // Unmappable block
                    todo!();
                }
            }
        }
        Ok(self.jit_mod)
    }
}

/// Given a mapped trace (through `aot_mod`), assemble and return a Yk IR trace.
pub(super) fn build(
    aot_mod: &Module,
    mtrace: &Vec<TracedAOTBlock>,
) -> Result<jit_ir::Module, CompilationError> {
    // FIXME: the XXX below should be a thread-safe monotonically incrementing integer.
    TraceBuilder::new("__yk_compiled_trace_XXX".into(), aot_mod, mtrace).build()
}
