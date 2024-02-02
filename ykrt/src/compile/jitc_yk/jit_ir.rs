//! The Yk JIT IR
//!
//! This is the in-memory trace IR constructed by the trace builder and mutated by optimisations.

// For now, don't swap others working in other areas of the system.
// FIXME: eventually delete.
#![allow(dead_code)]

use crate::compile::CompilationError;

use super::aot_ir;
use std::{fmt, mem, ptr};

/// Bit fiddling.
///
/// In the constants below:
///  - `*_SIZE`: the size of a field in bits.
///  - `*_MASK`: a mask with one bits occupying the field in question.
///  - `*_SHIFT`: the number of bits required to left shift a field's value into position (from the
///  LSB).
///
const OPERAND_IDX_MASK: u16 = 0x7fff;

// The largest operand index we can express in 15 bits.
const MAX_OPERAND_IDX: u16 = (1 << 15) - 1;

/// A packed 24-bit unsigned integer.
#[repr(packed)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct U24([u8; 3]);

impl U24 {
    /// Create a [U24] from a `usize`. Returns `None` if it won't fit.
    ///
    /// Returns none if the value won't fit.
    fn from_usize(val: usize) -> Option<Self> {
        if val >= 1 << 24 {
            None
        } else {
            let b0 = val & 0xff;
            let b1 = (val & 0xff00) >> 8;
            let b2 = (val & 0xff0000) >> 16;
            Some(Self([b2 as u8, b1 as u8, b0 as u8]))
        }
    }

    /// Converts 3-bytes conceptually representing a `u24` to a usize.
    fn to_usize(&self) -> usize {
        static_assertions::const_assert!(mem::size_of::<usize>() >= 3);
        let b0 = self.0[0] as usize; // most-significant byte.
        let b1 = self.0[1] as usize;
        let b2 = self.0[2] as usize;
        (b0 << 16) | (b1 << 8) | b2
    }
}

/// Helper to create index overflow errors.
fn index_overflow(typ: &str) -> CompilationError {
    CompilationError::Temporary(format!("index overflow: {}", typ))
}

// Generate common methods for 24-bit index types.
macro_rules! index_24bit {
    ($struct:ident) => {
        impl $struct {
            /// Convert an AOT index to a reduced-size JIT index (if possible).
            pub(crate) fn from_aot(aot_idx: aot_ir::$struct) -> Result<$struct, CompilationError> {
                U24::from_usize(usize::from(aot_idx))
                    .ok_or(index_overflow(stringify!($struct)))
                    .map(|u| Self(u))
            }

            /// Convert a JIT index to an AOT index.
            pub(crate) fn into_aot(&self) -> aot_ir::$struct {
                aot_ir::$struct::new(self.0.to_usize())
            }
        }
    };
}

// Generate common methods for 16-bit index types.
macro_rules! index_16bit {
    ($struct:ident) => {
        impl $struct {
            pub(crate) fn new(v: usize) -> Result<Self, CompilationError> {
                u16::try_from(v)
                    .map_err(|_| index_overflow(stringify!($struct)))
                    .map(|u| Self(u))
            }

            pub(crate) fn to_u16(&self) -> u16 {
                self.0.into()
            }
        }
    };
}

/// A function index that refers to a function in the AOT module's function table.
///
/// The JIT module shares its functions with the AOT module, but note that we use only a 24-bit
/// index in order to pack instructions down into 64-bits. The ramifications of are that the JIT IR
/// can only address a subset of the functions that the AOT module could possibly store. Arguably,
/// if a trace refers to that many functions, then it is likely to be a long trace that we probably
/// don't want to compile anyway.
#[derive(Copy, Clone, Debug)]
pub(crate) struct FuncIdx(U24);
index_24bit!(FuncIdx);

/// A type index that refers to a type in the AOT module's type table.
///
/// This works similarly to [FuncIdx], i.e. a reduced-size index type is used for compactness at
/// the cost of not being able to index every possible AOT type index.
///
/// See the [FuncIdx] docs for a full justification of this design.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TypeIdx(U24);
index_24bit!(TypeIdx);

/// An extra argument index.
///
/// One of these is an index into the [Module::extra_args].
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ExtraArgsIdx(u16);
index_16bit!(ExtraArgsIdx);

/// A constant index.
///
/// One of these is an index into the [Module::consts].
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(crate) struct ConstIdx(u16);
index_16bit!(ConstIdx);

/// An instruction index.
///
/// One of these is an index into the [Module::instrs].
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct InstrIdx(u16);
index_16bit!(InstrIdx);

/// The packed representation of an instruction operand.
///
/// # Encoding
///
/// ```ignore
///  1             15
/// +---+--------------------------+
/// | k |         index            |
/// +---+--------------------------+
/// ```
///
///  - `k=0`: `index` is a local variable index
///  - `k=1`: `index` is a constant index
///
///  The IR can represent 2^{15} = 32768 locals, and as many constants.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PackedOperand(u16);

impl PackedOperand {
    pub fn new(op: &Operand) -> Self {
        match op {
            Operand::Local(lidx) => {
                debug_assert!(lidx.to_u16() <= MAX_OPERAND_IDX);
                PackedOperand(lidx.to_u16())
            }
            Operand::Const(constidx) => {
                debug_assert!(constidx.to_u16() <= MAX_OPERAND_IDX);
                PackedOperand(constidx.to_u16() | !OPERAND_IDX_MASK)
            }
        }
    }

    /// Unpacks a [PackedOperand] into a [Operand].
    pub fn get(&self) -> Operand {
        if (self.0 & !OPERAND_IDX_MASK) == 0 {
            Operand::Local(InstrIdx(self.0))
        } else {
            Operand::Const(ConstIdx(self.0 & OPERAND_IDX_MASK))
        }
    }
}

/// An unpacked representation of a operand.
///
/// This exists both as a convenience (as working with packed operands is laborious) and as a means
/// to add type safety when using operands.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Operand {
    Local(InstrIdx),
    Const(ConstIdx),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Local(idx) => write!(f, "%{}", idx.to_u16()),
            Self::Const(idx) => write!(f, "Const({})", idx.to_u16()), // FIXME print constant properly.
        }
    }
}

// FIXME: this isn't the correct representation of a constant.
// It should be a bag of bytes and a type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Constant {
    Usize(usize),
}

/// An IR instruction.
#[repr(u8)]
#[derive(Debug)]
pub enum Instruction {
    Load(LoadInstruction),
    LoadArg(LoadArgInstruction),
    Call(CallInstruction),
}

impl Instruction {
    /// Returns `true` if the instruction defines a local variable.
    pub(crate) fn is_def(&self) -> bool {
        match self {
            Self::Load(..) => true,
            Self::LoadArg(..) => true,
            Self::Call(..) => true, // FIXME: May or may not define. Ask func sig.
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Load(i) => write!(f, "{}", i),
            Self::LoadArg(i) => write!(f, "{}", i),
            Self::Call(i) => write!(f, "{}", i),
        }
    }
}

macro_rules! instr {
    ($discrim:ident, $instr_type:ident) => {
        impl From<$instr_type> for Instruction {
            fn from(instr: $instr_type) -> Instruction {
                Instruction::$discrim(instr)
            }
        }
    };
}

instr!(Load, LoadInstruction);
instr!(LoadArg, LoadArgInstruction);
instr!(Call, CallInstruction);

/// The operands for a [Instruction::Load]
///
/// # Semantics
///
/// Loads a value from a given pointer operand.
///
#[derive(Debug)]
pub struct LoadInstruction {
    /// The pointer to load from.
    op: PackedOperand,
    /// The type of the pointee.
    ty_idx: TypeIdx,
}

impl LoadInstruction {
    pub(crate) fn new(op: Operand, ty_idx: TypeIdx) -> LoadInstruction {
        LoadInstruction {
            op: PackedOperand::new(&op),
            ty_idx,
        }
    }

    /// Return the pointer operand.
    pub(crate) fn operand(&self) -> Operand {
        self.op.get()
    }
}

impl fmt::Display for LoadInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Load {}", self.operand())
    }
}

/// The `LoadArg` instruction.
///
/// ## Semantics
///
/// Loads a live variable from the trace input struct.
///
/// ## Operands
///
/// FIXME: unimplemented as yet.
#[derive(Debug)]
pub struct LoadArgInstruction {
    // FIXME: todo
}

impl fmt::Display for LoadArgInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LoadArg")
    }
}

impl LoadArgInstruction {
    pub(crate) fn new() -> LoadArgInstruction {
        Self {}
    }
}

/// The operands for a [Instruction::Call]
///
/// # Semantics
///
/// Perform a call to an external or AOT function.
#[derive(Debug)]
#[repr(packed)]
pub struct CallInstruction {
    /// The callee.
    target: FuncIdx,
    /// The first argument to the call, if present. Undefined if not present.
    arg1: PackedOperand,
    /// Extra arguments, if the call requires more than a single argument.
    extra: ExtraArgsIdx,
}

impl fmt::Display for CallInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LoadArg")
    }
}

impl CallInstruction {
    pub(crate) fn new(
        m: &mut Module,
        target: aot_ir::FuncIdx,
        args: &[Operand],
    ) -> Result<CallInstruction, CompilationError> {
        let mut arg1 = PackedOperand::default();
        let mut extra = ExtraArgsIdx::default();

        if args.len() >= 1 {
            arg1 = PackedOperand::new(&args[0]);
        }
        if args.len() >= 2 {
            extra = m.push_extra_args(&args[1..])?;
        }
        Ok(Self {
            target: FuncIdx::from_aot(target)?,
            arg1,
            extra,
        })
    }

    fn arg1(&self) -> PackedOperand {
        let unaligned = ptr::addr_of!(self.arg1);
        unsafe { ptr::read_unaligned(unaligned) }
    }

    /// Fetch the operand at the specified index.
    ///
    /// It is undefined behaviour to provide an out-of-bounds index.
    pub(crate) fn operand(
        &self,
        _aot_mod: &aot_ir::Module,
        jit_mod: &Module,
        idx: usize,
    ) -> Option<Operand> {
        #[cfg(debug_assertions)]
        {
            let ft = _aot_mod.func_ty(self.target.into_aot());
            debug_assert!(ft.num_args() > idx);
        }
        if idx == 0 {
            Some(self.arg1().get())
        } else {
            Some(jit_mod.extra_args[usize::from(self.extra.0) + idx - 1].clone())
        }
    }
}

/// The `Module` is the top-level container for JIT IR.
///
/// The IR is conceptually a list of word-sized instructions containing indices into auxiliary
/// vectors.
///
/// The instruction stream of a [Module] is partially mutable:
/// - you may append new instructions to the end.
/// - you may replace and instruction with another.
/// - you may NOT remove an instruction.
#[derive(Debug)]
pub(crate) struct Module {
    /// The name of the module and the eventual symbol name for the JITted code.
    name: String,
    /// The IR trace as a linear sequence of instructions.
    instrs: Vec<Instruction>,
    /// The extra argument table.
    ///
    /// Used when a [CallInstruction]'s arguments don't fit inline.
    ///
    /// An [ExtraArgsIdx] describes an index into this.
    extra_args: Vec<Operand>,
    /// The constant table.
    ///
    /// A [ConstIdx] describes an index into this.
    consts: Vec<Constant>,
}

impl Module {
    /// Create a new [Module] with the specified name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            instrs: Vec::new(),
            extra_args: Vec::new(),
            consts: Vec::new(),
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

    /// Push a slice of extra arguments into the extra arg table.
    fn push_extra_args(&mut self, ops: &[Operand]) -> Result<ExtraArgsIdx, CompilationError> {
        let idx = self.extra_args.len();
        self.extra_args.extend_from_slice(ops); // FIXME: this clones.
        ExtraArgsIdx::new(idx)
    }

    /// Push a new constant into the constant table and return its index.
    pub(crate) fn push_const(&mut self, constant: Constant) -> Result<ConstIdx, CompilationError> {
        let idx = self.consts.len();
        self.consts.push(constant);
        ConstIdx::new(idx)
    }

    /// Get the index of a type, inserting it in the type table if necessary.
    pub fn const_idx(&mut self, c: &Constant) -> Result<ConstIdx, CompilationError> {
        // FIXME: can we optimise this?
        for (idx, tc) in self.consts.iter().enumerate() {
            if tc == c {
                // const table hit.
                return ConstIdx::new(idx);
            }
        }
        // type table miss, we need to insert it.
        self.push_const(c.clone())
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
    use std::mem;

    #[test]
    fn operand() {
        let op = PackedOperand::new(&Operand::Local(InstrIdx(192)));
        assert_eq!(op.get(), Operand::Local(InstrIdx(192)));

        let op = PackedOperand::new(&Operand::Local(InstrIdx(0x7fff)));
        assert_eq!(op.get(), Operand::Local(InstrIdx(0x7fff)));

        let op = PackedOperand::new(&Operand::Local(InstrIdx(0)));
        assert_eq!(op.get(), Operand::Local(InstrIdx(0)));

        let op = PackedOperand::new(&Operand::Const(ConstIdx(192)));
        assert_eq!(op.get(), Operand::Const(ConstIdx(192)));

        let op = PackedOperand::new(&Operand::Const(ConstIdx(0x7fff)));
        assert_eq!(op.get(), Operand::Const(ConstIdx(0x7fff)));

        let op = PackedOperand::new(&Operand::Const(ConstIdx(0)));
        assert_eq!(op.get(), Operand::Const(ConstIdx(0)));
    }

    #[test]
    fn use_case_update_instr() {
        let mut prog: Vec<Instruction> = vec![
            LoadArgInstruction::new().into(),
            LoadArgInstruction::new().into(),
            LoadInstruction::new(
                Operand::Local(InstrIdx(0)),
                TypeIdx(U24::from_usize(0).unwrap()),
            )
            .into(),
        ];
        prog[2] = LoadInstruction::new(
            Operand::Local(InstrIdx(1)),
            TypeIdx(U24::from_usize(0).unwrap()),
        )
        .into();
    }

    /// Ensure that any given instruction fits in 64-bits.
    #[test]
    fn instr_size() {
        assert_eq!(mem::size_of::<CallInstruction>(), 7);
        assert!(mem::size_of::<Instruction>() <= mem::size_of::<u64>());
    }

    #[test]
    fn extra_call_args() {
        let mut aot_mod = aot_ir::Module::default();
        let arg_ty_idxs = vec![aot_ir::TypeIdx::new(0); 3];
        let func_ty = aot_ir::Type::Func(aot_ir::FuncType::new(
            arg_ty_idxs,
            aot_ir::TypeIdx::new(0),
            false,
        ));
        let func_ty_idx = aot_mod.push_type(func_ty);
        let aot_func_idx = aot_mod.push_func(aot_ir::Function::new("", func_ty_idx));

        let mut jit_mod = Module::new("test".into());
        let args = vec![
            Operand::Local(InstrIdx(0)), // inline arg
            Operand::Local(InstrIdx(1)), // first extra arg
            Operand::Local(InstrIdx(2)),
        ];
        let ci = CallInstruction::new(&mut jit_mod, aot_func_idx, &args).unwrap();

        assert_eq!(
            ci.operand(&aot_mod, &jit_mod, 0),
            Some(Operand::Local(InstrIdx(0)))
        );
        assert_eq!(
            ci.operand(&aot_mod, &jit_mod, 1),
            Some(Operand::Local(InstrIdx(1)))
        );
        assert_eq!(
            ci.operand(&aot_mod, &jit_mod, 2),
            Some(Operand::Local(InstrIdx(2)))
        );
        assert_eq!(
            jit_mod.extra_args,
            vec![Operand::Local(InstrIdx(1)), Operand::Local(InstrIdx(2))]
        );
    }

    #[test]
    #[should_panic]
    fn call_args_out_of_bounds() {
        let mut aot_mod = aot_ir::Module::default();
        let arg_ty_idxs = vec![aot_ir::TypeIdx::new(0); 3];
        let func_ty = aot_ir::Type::Func(aot_ir::FuncType::new(
            arg_ty_idxs,
            aot_ir::TypeIdx::new(0),
            false,
        ));
        let func_ty_idx = aot_mod.push_type(func_ty);
        let aot_func_idx = aot_mod.push_func(aot_ir::Function::new("", func_ty_idx));

        let mut jit_mod = Module::new("test".into());
        let args = vec![
            Operand::Local(InstrIdx(0)), // inline arg
            Operand::Local(InstrIdx(1)), // first extra arg
            Operand::Local(InstrIdx(2)),
        ];
        let ci = CallInstruction::new(&mut jit_mod, aot_func_idx, &args).unwrap();

        ci.operand(&aot_mod, &jit_mod, 3).unwrap();
    }

    #[test]
    fn u24_from_usize() {
        assert_eq!(U24::from_usize(0x000000), Some(U24([0x00, 0x00, 0x00])));
        assert_eq!(U24::from_usize(0x123456), Some(U24([0x12, 0x34, 0x56])));
        assert_eq!(U24::from_usize(0xffffff), Some(U24([0xff, 0xff, 0xff])));
        assert_eq!(U24::from_usize(0x1000000), None);
        assert_eq!(U24::from_usize(0x1234567), None);
        assert_eq!(U24::from_usize(0xfffffff), None);
    }

    #[test]
    fn u24_to_usize() {
        assert_eq!(U24([0x00, 0x00, 0x00]).to_usize(), 0x000000);
        assert_eq!(U24([0x12, 0x34, 0x56]).to_usize(), 0x123456);
        assert_eq!(U24([0xff, 0xff, 0xff]).to_usize(), 0xffffff);
    }

    #[test]
    fn u24_round_trip() {
        assert_eq!(U24::from_usize(0x000000).unwrap().to_usize(), 0x000000);
        assert_eq!(U24::from_usize(0x123456).unwrap().to_usize(), 0x123456);
        assert_eq!(U24::from_usize(0xffffff).unwrap().to_usize(), 0xffffff);
    }

    #[test]
    fn index24_fits() {
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0)).is_ok());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(1)).is_ok());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0x1234)).is_ok());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0x123456)).is_ok());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0xffffff)).is_ok());
    }

    #[test]
    fn index24_doesnt_fit() {
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0x1000000)).is_err());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0x1234567)).is_err());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(0xeddedde)).is_err());
        assert!(TypeIdx::from_aot(aot_ir::TypeIdx::new(usize::MAX)).is_err());
    }

    #[test]
    fn index16_fits() {
        assert!(ExtraArgsIdx::new(0).is_ok());
        assert!(ExtraArgsIdx::new(1).is_ok());
        assert!(ExtraArgsIdx::new(0x1234).is_ok());
        assert!(ExtraArgsIdx::new(0xffff).is_ok());
    }

    #[test]
    fn index16_doesnt_fit() {
        assert!(ExtraArgsIdx::new(0x10000).is_err());
        assert!(ExtraArgsIdx::new(0x12345).is_err());
        assert!(ExtraArgsIdx::new(0xffffff).is_err());
        assert!(ExtraArgsIdx::new(usize::MAX).is_err());
    }
}
