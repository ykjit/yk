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

/// Number of bits used to encode an opcode.
const OPCODE_SIZE: u64 = 8;

/// Max number of operands in a short instruction.
const SHORT_INSTR_MAX_OPERANDS: u64 = 3;

/// Bit fiddling.
///
/// In the constants below:
///  - `*_SIZE`: the size of a field in bits.
///  - `*_MASK`: a mask with one bits occupying the field in question.
///  - `*_SHIFT`: the number of bits required to left shift a field's value into position (from the
///  LSB).
///
/// Bit fiddling for a short operands:
const SHORT_OPERAND_SIZE: u64 = 18;
const SHORT_OPERAND_KIND_SIZE: u64 = 3;
const SHORT_OPERAND_KIND_MASK: u64 = 7;
const SHORT_OPERAND_VALUE_SIZE: u64 = 15;
const SHORT_OPERAND_VALUE_SHIFT: u64 = SHORT_OPERAND_KIND_SIZE;
const SHORT_OPERAND_MAX_VALUE: u64 = !(u64::MAX << SHORT_OPERAND_VALUE_SIZE);
const SHORT_OPERAND_MASK: u64 = 0x3ffff;
/// Bit fiddling for instructions.
const INSTR_ISSHORT_SIZE: u64 = 1;
const INSTR_ISSHORT_MASK: u64 = 1;
const INSTR_OPCODE_MASK: u64 = 0xe;
/// Bit fiddling for short instructions.
const SHORT_INSTR_OPCODE_SHIFT: u64 = INSTR_ISSHORT_SIZE;
const SHORT_INSTR_FIRST_OPERAND_SHIFT: u64 = INSTR_ISSHORT_SIZE + OPCODE_SIZE;

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

/// An operand kind.
#[repr(u64)]
#[derive(Debug, FromRepr, PartialEq)]
pub enum OpKind {
    /// The operand is not present.
    ///
    /// This is used in short instructions where 3 operands are inlined. If the instruction
    /// requires fewer then 3 operands, then it can use this variant to express that.
    ///
    /// By using the zero discriminant, this means that a freshly created short instruction has
    /// with zero operands until they are explicitly filled in.
    NotPresent = 0,
    /// The operand references a previously defined local variable.
    Local,
}

impl From<u64> for OpKind {
    fn from(v: u64) -> Self {
        // unwrap safe assuming only valid discriminant numbers are used.
        Self::from_repr(v).unwrap()
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

#[derive(Debug, PartialEq)]
pub enum Operand {
    Long(LongOperand),
    Short(ShortOperand),
}

impl Operand {
    pub(crate) fn new(kind: OpKind, val: u64) -> Self {
        // check if the operand's value can fit in a short operand.
        if val <= SHORT_OPERAND_MAX_VALUE {
            Self::Short(ShortOperand::new(kind, val))
        } else {
            todo!()
        }
    }

    fn raw(&self) -> u64 {
        match self {
            Self::Long(_) => todo!(),
            Self::Short(op) => op.0,
        }
    }

    fn kind(&self) -> OpKind {
        match self {
            Self::Long(_) => todo!(),
            Self::Short(op) => op.kind(),
        }
    }

    fn val(&self) -> u64 {
        match self {
            Self::Long(_) => todo!(),
            Self::Short(op) => op.val(),
        }
    }

    fn is_short(&self) -> bool {
        matches!(self, Self::Short(_))
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind() {
            OpKind::Local => write!(f, " %{}", self.val())?,
            OpKind::NotPresent => (),
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct LongOperand(u64);

#[derive(Debug, PartialEq)]
pub struct ShortOperand(u64);

impl ShortOperand {
    fn new(kind: OpKind, val: u64) -> ShortOperand {
        ShortOperand((kind as u64) | (val << SHORT_OPERAND_VALUE_SHIFT))
    }

    fn kind(&self) -> OpKind {
        OpKind::from(self.0 & SHORT_OPERAND_KIND_MASK)
    }

    fn val(&self) -> u64 {
        self.0 >> SHORT_OPERAND_VALUE_SHIFT
    }
}

/// An instruction.
///
/// An instruction is either a short instruction or a long instruction.
///
/// ## Short instruction
///
/// - A 64-bit bit-field that encodes the entire instruction inline.
/// - Can encode up to three short operands.
/// - Is designed to encode the most commonly encountered instructions.
///
/// Encoding (LSB first):
/// ```ignore
/// field           bit-size
/// ------------------------
/// is_short=1      1
/// opcode          8
/// short_operand0  18
/// short_operand1  18
/// short_operand2  18
/// reserved        1
/// ```
///
/// Where a short operand is encoded like this (LSB first):
/// ```ignore
/// field       bit-size
/// --------------------
/// kind        3
/// payload    15
/// ```
///
/// ## Long instruction
///
/// - A pointer to an instruction description.
/// - Can encode an arbitrary number of long operands.
///
/// The pointer is assumed to be at least 2-byte aligned, thus guaranteeing the LSB to be 0.
#[derive(Debug)]
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
        debug_assert!(self.is_short());
        OpCode::from((self.0 & INSTR_OPCODE_MASK) >> SHORT_INSTR_OPCODE_SHIFT)
    }

    /// Returns the specified operand.
    fn operand(&self, index: u64) -> Operand {
        if self.is_short() {
            // Shift operand down the the LSB.
            let op = self.0 >> (SHORT_INSTR_FIRST_OPERAND_SHIFT + SHORT_OPERAND_SIZE * index);
            // Then mask it out.
            Operand::Short(ShortOperand(op & SHORT_OPERAND_MASK))
        } else {
            todo!()
        }
    }

    /// Create a new `Load` instruction.
    ///
    /// ## Operands
    ///
    /// - `<ptr>`:  The pointer to load from.
    ///
    /// ## Semantics
    ///
    /// Return the value obtained by dereferencing the operand (which must be pointer-typed).
    pub(crate) fn create_load(op: Operand) -> Self {
        if op.is_short() {
            let mut instr = Instruction::new_short(OpCode::Load);
            instr.set_short_operand(op, 0);
            instr
        } else {
            todo!();
        }
    }

    /// Create a new `LoadArg` instruction.
    ///
    /// ## Operands
    ///
    /// FIXME
    ///
    /// ## Semantics
    ///
    /// FIXME
    pub(crate) fn create_loadarg() -> Self {
        Instruction::new_short(OpCode::LoadArg)
    }

    /// Set the short operand at the specified index.
    fn set_short_operand(&mut self, op: Operand, idx: u64) {
        debug_assert!(self.is_short());
        debug_assert!(idx < SHORT_INSTR_MAX_OPERANDS);
        self.0 |= op.raw() << (SHORT_INSTR_FIRST_OPERAND_SHIFT + SHORT_OPERAND_SIZE * idx);
    }

    /// Returns `true` if the instruction defines a local variable.
    pub(crate) fn is_def(&self) -> bool {
        match self.opcode() {
            OpCode::Load => true,
            OpCode::LoadArg => true,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let opc = self.opcode();
        write!(f, "{:?}", opc)?;
        if self.is_short() {
            for i in 0..=2 {
                let op = self.operand(i);
                write!(f, "{}", op)?;
            }
        }
        Ok(())
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
        let op = Operand::new(OpKind::Local, 10);
        let instr = Instruction::create_load(op);
        assert_eq!(instr.opcode(), OpCode::Load);
        assert_eq!(instr.operand(0).kind(), OpKind::Local);
        assert_eq!(instr.operand(0).val(), 10);
        assert!(instr.is_def());
        assert_eq!(instr.0, 0xa201);
        assert!(instr.is_short());
    }

    #[test]
    fn long_instruction() {
        // FIXME: expand when long instructions are implemented.
        let instr = Instruction(0);
        assert!(!instr.is_short());
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
    fn short_operand_getters() {
        let mut word = 1; // short instruction.

        // operand0:
        word |= 0x0aaa8 << SHORT_INSTR_FIRST_OPERAND_SHIFT;
        // operand1:
        word |= 0x1bbb1 << SHORT_INSTR_FIRST_OPERAND_SHIFT + SHORT_OPERAND_SIZE;
        // operand2:
        word |= 0x2ccc8 << SHORT_INSTR_FIRST_OPERAND_SHIFT + SHORT_OPERAND_SIZE * 2;

        let inst = Instruction(word);

        assert_eq!(inst.operand(0), Operand::Short(ShortOperand(0x0aaa8)));
        assert_eq!(inst.operand(0).kind() as u64, 0);
        assert_eq!(inst.operand(0).val() as u64, 0x1555);

        assert_eq!(inst.operand(1), Operand::Short(ShortOperand(0x1bbb1)));
        assert_eq!(inst.operand(1).kind() as u64, 1);
        assert_eq!(inst.operand(1).val() as u64, 0x3776);

        assert_eq!(inst.operand(2), Operand::Short(ShortOperand(0x2ccc8)));
        assert_eq!(inst.operand(2).kind() as u64, 0);
        assert_eq!(inst.operand(2).val() as u64, 0x5999);
    }

    #[test]
    fn short_operand_setters() {
        let mut inst = Instruction::new_short(OpCode::Load);
        inst.set_short_operand(Operand::Short(ShortOperand(0x3ffff)), 0);
        debug_assert_eq!(inst.operand(0), Operand::Short(ShortOperand(0x3ffff)));
        debug_assert_eq!(inst.operand(1), Operand::Short(ShortOperand(0)));
        debug_assert_eq!(inst.operand(2), Operand::Short(ShortOperand(0)));

        let mut inst = Instruction::new_short(OpCode::Load);
        inst.set_short_operand(Operand::Short(ShortOperand(0x3ffff)), 1);
        debug_assert_eq!(inst.operand(0), Operand::Short(ShortOperand(0)));
        debug_assert_eq!(inst.operand(1), Operand::Short(ShortOperand(0x3ffff)));
        debug_assert_eq!(inst.operand(2), Operand::Short(ShortOperand(0)));

        let mut inst = Instruction::new_short(OpCode::Load);
        inst.set_short_operand(Operand::Short(ShortOperand(0x3ffff)), 2);
        debug_assert_eq!(inst.operand(0), Operand::Short(ShortOperand(0)));
        debug_assert_eq!(inst.operand(1), Operand::Short(ShortOperand(0)));
        debug_assert_eq!(inst.operand(2), Operand::Short(ShortOperand(0x3ffff)));
    }

    #[test]
    fn does_fit_short_operand() {
        for i in 0..SHORT_OPERAND_VALUE_SIZE {
            matches!(Operand::new(OpKind::Local, 1 << i), Operand::Short(_));
        }
    }

    #[test]
    #[should_panic] // Once long operands are implemented, remove.
    fn doesnt_fit_short_operand() {
        matches!(
            Operand::new(OpKind::Local, 1 << SHORT_OPERAND_VALUE_SIZE),
            Operand::Long(_)
        );
    }
}
