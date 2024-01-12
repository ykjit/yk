//! The Yk JIT IR
//!
//! This is the in-memory trace IR constructed by the trace builder and mutated by optimisations.
//!
//! Design notes:
//!
//!  - This module uses `u64` extensively for bit-fields. This is not a consequence of any
//!    particular hardware platform, we just chose a 64-bit field.
//!
//!  - We avoid heap allocations at all costs.

use std::fmt;
use strum_macros::FromRepr;

/// Bit fiddling.
///
/// In the constants below:
///  - `*_SIZE`: the size of a field in bits.
///  - `*_MASK`: a mask with one bits occupying the field in question.
///  - `*_SHIFT`: the number of bits required to left shift a field's value into position (from the
///  LSB).
///
/// Bit fiddling for instructions.
const INSTR_ISSHORT_SIZE: u64 = 1;
const INSTR_ISSHORT_MASK: u64 = 1;
/// Bit fiddling for short instructions.
const SHORT_INSTR_OPCODE_MASK: u64 = 0xe;
const SHORT_INSTR_OPCODE_SHIFT: u64 = INSTR_ISSHORT_SIZE;
const SHORT_LOAD_OPERAND_SHIFT: u64 = 9;
const SHORT_LOAD_OPERAND_SIZE: u64 = 53;

/// An instruction is identified by its index in the instruction vector.
#[derive(Copy, Clone)]
pub(crate) struct InstructionID(usize);

impl InstructionID {
    pub(crate) fn new(v: usize) -> Self {
        Self(v)
    }

    pub(crate) fn get(&self) -> usize {
        self.0
    }
}

#[derive(Debug, FromRepr, PartialEq)]
#[repr(u64)]
pub enum OpCode {
    Load,
    LoadArg,
}

impl From<u64> for OpCode {
    fn from(v: u64) -> Self {
        // unwrap safe assuming only valid discriminant numbers are used.
        Self::from_repr(v).unwrap()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Local(u64),
    ConstantPointer(u64),
}

impl Operand {
    fn fits_in_n_bits(&self, n: u64) -> bool {
        match self {
            Self::Local(idx) => idx >> n == 0,
            Self::ConstantPointer(_) => n >= 64,
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Local(idx) => write!(f, "%{}", idx),
            _ => todo!(),
        }
    }
}

/// A generic instruction.
///
/// This struct is the generic "at-rest" encoding of a specific JIT IR instruction. In order to
/// access the operands of an instruction, it must first be "interpreted" as the correct specific
/// instruction.
///
/// For example:
///
/// ```ignore
/// let instr: Instruction = LoadInstruction::new(Operand::Local(0));
/// let loadinstr = LoadInstruction(&instr);
/// let op = loadinstr.operand();
/// ```
///
/// Instructions come in short and long variants. A short instruction packs all of its data inline
/// into the 64-bit [Instruction], whereas a long instruction is a box pointer to a larger data
/// structure describing the operands.
///
/// The IR is designed such that (where possible) the most common kinds of operations can be
/// encoded into a short instruction.
///
/// Instructions are immutable and must not be copied. If you wish to change an instruction,
/// you must make a new one and replace the old one. If you wish to copy an instruction, you
/// should make a new one with the same operands. There are several reasons for this:
///
///   1) modifying a (boxed) long instruction could lead to it become short, which may result in
///      accidental memory leaks if we are not careful to reconsititute the [Box] and let it drop.
///   2) copying a long instruction would make a hidden alias to a boxed pointer. This can lead to
///      double frees or use after.
///   3) when instructions are stored inside a vector (e.g. a trace), we may only mutably borrow a
///      single instruction at a time, which would be impractical when working on the IR, e.g.
///      during optimisation.
///
/// ## Encodings (LSB first)
///
/// ### Short
///
/// ```ignore
/// 0:      is_short=x1
/// 1-9:    opcode discriminant
/// 10-63:  opcode-specific bitvector
/// ```
///
/// ### Long
///
/// ```ignore
/// 0-63:   box pointer
/// ```
///
/// The pointer is assumed to be at least 2-byte aligned, thus guaranteeing the LSB to be 0.
#[derive(Debug, Clone)]
pub(crate) struct Instruction(u64);

impl Instruction {
    fn new_short(opcode: OpCode) -> Self {
        Self(((opcode as u64) << SHORT_INSTR_OPCODE_SHIFT) | INSTR_ISSHORT_MASK)
    }

    /// Returns true if the instruction is short.
    fn is_short(&self) -> bool {
        self.0 & INSTR_ISSHORT_MASK != 0
    }

    /// Returns the opcode.
    fn opcode(&self) -> OpCode {
        if self.is_short() {
            OpCode::from((self.0 & SHORT_INSTR_OPCODE_MASK) >> SHORT_INSTR_OPCODE_SHIFT)
        } else {
            todo!();
        }
    }

    /// Returns `true` if the instruction defines a local variable.
    pub(crate) fn is_def(&self) -> bool {
        match self.opcode() {
            OpCode::Load => true,
            OpCode::LoadArg => true,
        }
    }
}

impl Drop for Instruction {
    fn drop(&mut self) {
        if !self.is_short() {
            unsafe { *Box::from_raw(self.0 as *mut _) }
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.opcode() {
            OpCode::Load => write!(f, "{}", LoadInstruction(&self)),
            OpCode::LoadArg => write!(f, "{}", LoadArgInstruction(&self)),
        }
    }
}

// Generate generic accessors for specific instruction types.
macro_rules! instr {
    ($struct:ident, $opcode:ident) => {
        impl<'a> $struct<'a> {
            #[allow(dead_code)]
            fn is_short(&self) -> bool {
                self.0.is_short()
            }

            #[allow(dead_code)]
            fn is_def(&self) -> bool {
                self.0.is_def()
            }
        }
    };
}

/// The `Load` instruction.
///
/// ## Semantics
///
/// Return the value obtained by dereferencing the operand (which must be pointer-typed).
///
/// ## Operands
///
/// - ptr: The pointer to load from.
///
/// ## Encodings (LSB first)
///
/// ### Short
///
/// A short load always loads from a pointer stored in a local variable.
///
/// ```ignore
///   0-9: short operand header
///   10-63: local index (of pointer operand).
/// ```
///
/// ### Long
///
/// ```ignore
/// 0-63: boxed pointer to a [LongLoadInstruction].
/// ```
pub(crate) struct LoadInstruction<'a>(&'a Instruction);
instr!(LoadInstruction, Load);

impl LoadInstruction<'_> {
    pub(crate) fn new(op: Operand) -> Instruction {
        if op.fits_in_n_bits(SHORT_LOAD_OPERAND_SIZE) {
            let mut instr = Instruction::new_short(OpCode::Load);
            if let Operand::Local(lidx) = op {
                instr.0 |= lidx << SHORT_LOAD_OPERAND_SHIFT;
                instr
            } else {
                panic!();
            }
        } else {
            // Create long instruction.
            let b = Box::new(LongLoadInstruction::new(op));
            Instruction(Box::into_raw(b) as u64)
        }
    }

    pub(crate) fn operand(&self) -> Operand {
        if self.is_short() {
            // Shift operand down the the LSB.
            Operand::Local(self.0 .0 >> SHORT_LOAD_OPERAND_SHIFT)
        } else {
            let b: Box<LongLoadInstruction> =
                unsafe { Box::from_raw(self.0 .0 as *mut LongLoadInstruction) };
            let ret = b.operand().clone(); // FIXME? unforntunate clone...
            Box::leak(b);
            ret
        }
    }
}

impl fmt::Display for LoadInstruction<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Load {}", self.operand())
    }
}

struct LongLoadInstruction {
    op: Operand,
}

impl LongLoadInstruction {
    fn new(op: Operand) -> Self {
        Self { op }
    }

    pub(crate) fn operand(&self) -> &Operand {
        &self.op
    }
}

/// The `LoadArg` instruction.
///
/// ## Semantics
///
/// FIXME
///
/// ## Operands
///
/// FIXME
///
/// ## Encodings (LSB first)
///
/// FIXME
pub(crate) struct LoadArgInstruction<'a>(&'a Instruction);
instr!(LoadArgInstruction, LoadArg);

impl fmt::Display for LoadArgInstruction<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LoadArg")
    }
}

impl<'a> LoadArgInstruction<'a> {
    pub(crate) fn new() -> Instruction {
        Instruction::new_short(OpCode::LoadArg)
    }
}

/// The `Module` is the top-level container for JIT IR.
#[derive(Debug)]
pub(crate) struct Module {
    /// The name of the module and the eventual symbol name for the JITted code.
    name: String,
    /// The IR trace as a linear sequence of instructions.
    instrs: Vec<Instruction>,
}

impl Module {
    /// Create a new [Module] with the specified name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            instrs: Vec::new(),
        }
    }

    /// Push an instruction to the end of the [Module].
    pub(crate) fn push(&mut self, instr: Instruction) {
        self.instrs.push(instr);
    }

    /// Returns the number of [Instruction]s in the [Module].
    pub(crate) fn len(&self) -> usize {
        self.instrs.len()
    }

    /// Print the [Module] to `stderr`.
    pub(crate) fn dump(&self) {
        eprintln!("{}", self);
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "; {}", self.name)?;
        for (i, instr) in self.instrs.iter().enumerate() {
            if instr.is_def() {
                write!(f, "%{} = ", i)?;
            }
            writeln!(f, "{}", instr)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_instruction() {
        let op = Operand::Local(10);
        let instr = LoadInstruction::new(op);
        assert_eq!(instr.0, 0x1401);
        assert!(instr.is_short());
    }

    #[test]
    fn long_instruction() {
        let op = Operand::Local(u64::max_value());
        let instr = LoadInstruction::new(op);
        assert!(!instr.is_short());
    }

    #[test]
    fn is_def() {
        let op = Operand::Local(10);
        let instr = LoadInstruction::new(op);
        assert!(instr.is_def());
    }

    /// The IR encoding uses a LSB tag to determine if an instruction is short or not, and if it
    /// isn't short then it's interpreted as a box pointer. So a box pointer had better be at least
    /// 2-byte aligned!
    ///
    /// This test (somewhat) proves that we are safe by allocating a bunch of `Box<u8>` (which in
    /// theory could be stored contiguously) and then checks their addresses don't have the LSB set
    /// (as this would indicate 1-byte alignment!).
    #[test]
    fn tagging_valid() {
        let mut boxes = Vec::new();
        for i in 0..8192 {
            boxes.push(Box::new(i as u8));
        }

        for b in boxes {
            assert_eq!((&*b as *const u8 as usize) & 1, 0);
        }
    }

    #[test]
    fn update_instr() {
        let mut prog = vec![
            LoadArgInstruction::new(),
            LoadArgInstruction::new(),
            LoadInstruction::new(Operand::Local(0)),
        ];
        prog[2] = LoadInstruction::new(Operand::Local(1));
    }

    #[test]
    fn read_loadinstr() {
        let prog = vec![LoadInstruction::new(Operand::Local(0))];
        let load = LoadInstruction(&prog[0]);
        assert_eq!(load.operand(), Operand::Local(0));
    }

    #[test]
    fn read_long_loadinstr() {
        let prog = vec![LoadInstruction::new(Operand::Local(u64::max_value()))];
        let load = LoadInstruction(&prog[0]);
        assert_eq!(load.operand(), Operand::Local(u64::max_value()));
    }
}
