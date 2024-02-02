use super::jit_ir::Instruction;

/// A wrapper around a vector of instructions used to store a trace. Since we disallow removal and
/// insertion (with the exception of appending at the end) into the vector, the wrapper only
/// exposes the pushing and replacing of instructions.
#[derive(Debug)]
pub struct TraceVec(Vec<Instruction>);

impl TraceVec {
    pub fn new() -> Self {
        TraceVec(Vec::new())
    }

    pub fn push(&mut self, instr: Instruction) {
        self.0.push(instr);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Instruction> {
        self.0.iter()
    }
}
