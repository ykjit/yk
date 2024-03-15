//! The Yk JIT IR
//!
//! This is the in-memory trace IR constructed by the trace builder and mutated by optimisations.

// For now, don't swap others working in other areas of the system.
// FIXME: eventually delete.
#![allow(dead_code)]

use crate::compile::CompilationError;
use std::{error::Error, ffi::c_void, mem, ptr};
use typed_index_collections::TiVec;

// Since the AOT versions of these data structures contain no AOT/JIT-IR-specific indices we can
// share them. Note though, that their corresponding index types are not shared.
pub(crate) use super::aot_ir::IntegerType;
pub(crate) use super::aot_ir::{GlobalDecl, Predicate};

impl JitIRDisplay for IntegerType {
    fn to_string_impl<'a>(
        &self,
        _m: &Module,
        s: &mut String,
        _nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str(&format!("i{}", self.num_bits()));
        Ok(())
    }
}

impl JitIRDisplay for GlobalDecl {
    fn to_string_impl<'a>(
        &self,
        _m: &Module,
        s: &mut String,
        _nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("@");
        s.push_str(&self.name());
        Ok(())
    }
}

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

/// [Instruction] to [InstrIdx] mapping used for stringifying instructions.
///
/// See [JitIRDisplay] for why this is required.
pub(crate) struct LocalNumbers<'a> {
    m: &'a Module,
}

impl<'a> LocalNumbers<'a> {
    fn new(m: &'a Module) -> LocalNumbers<'a> {
        Self { m }
    }

    /// Returns the [InstrIdx] of the specified [Instruction] in the [Module].
    ///
    /// # Panics
    ///
    /// Panics if the instruction isn't present in the module.
    fn idx(&self, instr: &Instruction) -> InstrIdx {
        // FIXME: This is inefficient.
        for (idx, candidate) in self.m.instrs().iter().enumerate() {
            if ptr::addr_eq(instr, candidate) {
                // The `unwrap` cannot fail because we won't have expanded `m.instrs()` beyond the
                // ability of `InstrIdx` to represent an index.
                return InstrIdx::new(idx).unwrap();
            }
        }
        panic!(); // Not found!
    }
}

/// A trait for stringifying JIT IR.
///
/// We use this instead of [std::fmt::Display] because many IR constructs can't print something
/// human-readable without querying the [Module] for more information. For example to convert type
/// indices into a human-readable type description. Hence we need the ability to pass down a
/// reference to a module.
///
/// Stringification also requires a [LocalNumbers] mapping. This is because when you want to print
/// an [Instruction], we want to print something like `%4: i32 = Add %1, %2`. But an instruction on
/// its own doesn't know its instruction index in order to print `%4`. [LocalNumbers] lets an
/// instruction find its index in the module.
pub(crate) trait JitIRDisplay {
    /// Return a human-readable string.
    ///
    /// This is the outward-facing API for stringification. Don't call this from other
    /// `JitIRDisplay` implementations. Use [JitIRDisplay::to_string_impl()] instead and pass down
    /// the existing output string and [LocalNumbers] struct.
    fn to_string(&self, m: &Module) -> Result<String, Box<dyn Error>> {
        let mut s = String::new();
        self.to_string_impl(m, &mut s, &LocalNumbers::new(m))?;
        Ok(s)
    }

    /// This method does the actual stringification of an IR element.
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>>;

    /// Print `self` to stderr in human-readable form.
    ///
    /// This isn't used during normal operation of the system: it is provided as a debugging aid.
    #[allow(dead_code)]
    fn dump(&self, m: &Module) -> Result<(), Box<dyn Error>> {
        eprintln!("{}", self.to_string(m).map_err(|e| e)?);
        Ok(())
    }
}

/// A packed 24-bit unsigned integer.
#[repr(packed)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    CompilationError(format!("index overflow: {}", typ))
}

// Generate common methods for 24-bit index types.
macro_rules! index_24bit {
    ($struct:ident) => {
        impl $struct {
            pub(crate) fn new(v: usize) -> Result<Self, CompilationError> {
                U24::from_usize(v)
                    .ok_or(index_overflow(stringify!($struct)))
                    .map(|u| Self(u))
            }

            pub(crate) fn to_usize(&self) -> usize {
                self.0.to_usize()
            }
        }

        impl From<usize> for $struct {
            // Required for TiVec.
            //
            // Prefer use of [Self::new], which is fallable. Certainly never use this in the trace
            // builder, where we expect to be able to recover from index overflows.
            fn from(v: usize) -> Self {
                Self::new(v).unwrap()
            }
        }

        impl From<$struct> for usize {
            // Required for TiVec.
            fn from(v: $struct) -> Self {
                v.to_usize()
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

        impl From<usize> for $struct {
            // Required for TiVec.
            //
            // Prefer use of [Self::new], which is fallable. Certainly never use this in the trace
            // builder, where we expect to be able to recover from index overflows.
            fn from(v: usize) -> Self {
                Self::new(v).unwrap()
            }
        }

        impl From<$struct> for usize {
            fn from(s: $struct) -> usize {
                s.0.into()
            }
        }
    };
}

/// A function declaration index.
///
/// One of these is an index into the [Module::func_decls].
#[derive(Copy, Clone, Debug)]
pub(crate) struct FuncDeclIdx(U24);
index_24bit!(FuncDeclIdx);

impl FuncDeclIdx {
    /// Return the type of the function declaration.
    pub(crate) fn func_type<'a>(&self, m: &'a Module) -> &'a FuncType {
        m.func_decl(*self).func_type(m)
    }
}

/// A type index.
///
/// One of these is an index into the [Module::types].
///
/// A type index uniquely identifies a [Type] in a [Module]. You can rely on this uniquness
/// property for type checking: you can compare type indices instead of the corresponding [Type]s.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct TypeIdx(U24);
index_24bit!(TypeIdx);

impl TypeIdx {
    pub(crate) fn type_<'a>(&self, m: &'a Module) -> &'a Type {
        m.type_(*self)
    }
}

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

/// A global variable declaration index.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct GlobalDeclIdx(U24);
index_24bit!(GlobalDeclIdx);

/// An instruction index.
///
/// One of these is an index into the [Module::instrs].
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, PartialOrd)]
pub(crate) struct InstrIdx(u16);
index_16bit!(InstrIdx);

impl InstrIdx {
    /// Return a reference to the instruction indentified by `self` in `jit_mod`.
    pub(crate) fn instr<'a>(&'a self, jit_mod: &'a Module) -> &Instruction {
        jit_mod.instr(*self)
    }
}

impl JitIRDisplay for InstrIdx {
    fn to_string_impl<'a>(
        &self,
        _m: &Module,
        s: &mut String,
        _nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("%");
        s.push_str(&self.to_u16().to_string());
        Ok(())
    }
}

/// A function's type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FuncType {
    /// Type indices for the function's formal arguments.
    arg_ty_idxs: Vec<TypeIdx>,
    /// Type index of the function's return type.
    ret_ty_idx: TypeIdx,
    /// Is the function vararg?
    is_vararg: bool,
}

impl FuncType {
    pub(crate) fn new(arg_ty_idxs: Vec<TypeIdx>, ret_ty_idx: TypeIdx, is_vararg: bool) -> Self {
        Self {
            arg_ty_idxs,
            ret_ty_idx,
            is_vararg,
        }
    }

    /// Return the number of arguments the function accepts (not including varargs arguments).
    pub(crate) fn num_args(&self) -> usize {
        self.arg_ty_idxs.len()
    }

    /// Returns the type index of the argument at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn arg_type<'a>(&self, m: &'a Module, idx: usize) -> &'a Type {
        self.arg_ty_idxs[idx].type_(m)
    }

    /// Returns whether the function type has vararg arguments.
    pub(crate) fn is_vararg(&self) -> bool {
        self.is_vararg
    }

    /// Returns the type of the return value.
    pub(crate) fn ret_type<'a>(&self, m: &'a Module) -> &'a Type {
        self.ret_ty_idx.type_(m)
    }

    /// Returns the type index of the return value.
    pub(crate) fn ret_type_idx(&self) -> TypeIdx {
        self.ret_ty_idx
    }
}

/// A structure's type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StructType {
    /// The types of the fields.
    field_ty_idxs: Vec<TypeIdx>,
    /// The bit offsets of the fields (taking into account any required padding for alignment).
    field_bit_offs: Vec<usize>,
}

/// A type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Type {
    Void,
    Integer(IntegerType),
    Ptr,
    Func(FuncType),
    Struct(StructType),
    Unimplemented(String),
}

impl JitIRDisplay for Type {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Void => s.push_str("void"),
            Self::Integer(it) => it.to_string_impl(m, s, nums)?,
            Self::Ptr => s.push_str("ptr"),
            Self::Func(_) => todo!(),
            Self::Struct(_) => todo!(),
            Self::Unimplemented(_) => s.push_str("?type"),
        }
        Ok(())
    }
}

impl Type {
    /// Returns the size of the type in bits, or `None` if asking the size makes no sense.
    pub(crate) fn byte_size(&self) -> Option<usize> {
        // u16/u32 -> usize conversions could theoretically fail on some arches (which we probably
        // won't ever support).
        match self {
            Self::Void => Some(0),
            Self::Integer(it) => Some(usize::try_from(it.byte_size()).unwrap()),
            Self::Ptr => {
                // FIXME: In theory pointers to different types could be of different sizes. We
                // should really ask LLVM how big the pointer was when it codegenned the
                // interpreter, and on a per-pointer basis.
                //
                // For now we assume (and ykllvm assserts this) that all pointers are void
                // pointer-sized.
                Some(mem::size_of::<*const c_void>())
            }
            Self::Func(_) => None,
            Self::Struct(_) => todo!(),
            Self::Unimplemented(_) => None,
        }
    }
}

/// An (externally defined, in the AOT code) function declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FuncDecl {
    name: String,
    type_idx: TypeIdx,
}

impl FuncDecl {
    pub(crate) fn new(name: String, type_idx: TypeIdx) -> Self {
        Self { name, type_idx }
    }

    /// Returns the [FuncType] for `self.type_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `self.type_idx` isn't a type index for a [FuncType].
    pub(crate) fn func_type<'a>(&self, m: &'a Module) -> &'a FuncType {
        match m.type_(self.type_idx) {
            Type::Func(ft) => &ft,
            _ => panic!(),
        }
    }

    /// Return the name of this function declaration.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

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
    pub fn unpack(&self) -> Operand {
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

impl Operand {
    /// Returns the size of the operand in bytes.
    ///
    /// Assumes no padding is required for alignment.
    pub(crate) fn byte_size(&self, m: &Module) -> usize {
        match self {
            Self::Local(l) => l.instr(m).def_byte_size(m),
            Self::Const(c) => match m.consts[*c] {
                Constant::U32(_) => std::mem::size_of::<u32>(),
                Constant::Usize(_) => std::mem::size_of::<usize>(),
            },
        }
    }

    /// Returns the type of the operand.
    pub(crate) fn type_<'a>(&self, m: &'a Module) -> &'a Type {
        match self {
            Self::Local(l) => {
                match l.instr(m).def_type(m) {
                    Some(t) => t,
                    None => {
                        // When an operand is a local variable, the local can only come from an
                        // instruction that defines a local variable, and thus has a type. So this
                        // can't happen if the IR is well-formed.
                        unreachable!();
                    }
                }
            }
            _ => todo!(),
        }
    }

    /// Returns the type index of the operand.
    pub(crate) fn type_idx(&self, m: &Module) -> TypeIdx {
        match self {
            Self::Local(l) => l.instr(m).def_type_idx(m),
            Self::Const(_) => todo!(),
        }
    }
}

impl JitIRDisplay for Operand {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Local(idx) => s.push_str(&format!("%{}", idx.to_u16())),
            Self::Const(idx) => m.const_(*idx).to_string_impl(m, s, nums)?,
        }
        Ok(())
    }
}

// FIXME: this isn't the correct representation of a constant.
// It should be a bag of bytes and a type.
//
// FIXME: Why are the variants `U*` not `I*` like everywhere else? Our integer IR types has no
// notion of signedness -- it's all up to interpretation of the bits by an operation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Constant {
    U32(u32),
    Usize(usize),
}

impl JitIRDisplay for Constant {
    fn to_string_impl<'a>(
        &self,
        _m: &Module,
        s: &mut String,
        _nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::U32(v) => s.push_str(&v.to_string()),
            Self::Usize(v) => s.push_str(&v.to_string()),
        }
        Ok(())
    }
}

/// An IR instruction.
#[repr(u8)]
#[derive(Debug)]
pub enum Instruction {
    Load(LoadInstruction),
    LookupGlobal(LookupGlobalInstruction),
    LoadTraceInput(LoadTraceInputInstruction),
    Call(CallInstruction),
    PtrAdd(PtrAddInstruction),
    Store(StoreInstruction),
    Add(AddInstruction),
    Icmp(IcmpInstruction),
    Guard(GuardInstruction),
}

impl Instruction {
    /// Returns `true` if the instruction defines a local variable.
    ///
    /// FIXME: Because self.def_type_idx() isn't complete, we have to handle various possibilities
    /// here in order that anything works. Once self.get_type_idx() is complete (i.e. no todo!()s
    /// left) this function can become simply `self.def_type_idx() != jit_mod.void_type_idx()`.
    pub(crate) fn is_def(&self) -> bool {
        match self {
            Self::Load(..) => true,
            Self::LookupGlobal(..) => true,
            Self::LoadTraceInput(..) => true,
            Self::Call(..) => true, // FIXME: May or may not define. Ask func sig.
            Self::PtrAdd(..) => true,
            Self::Store(..) => false,
            Self::Add(..) => true,
            Self::Icmp(..) => true,
            Self::Guard(..) => false,
        }
    }

    /// Returns the type of the local variable that the instruction defines (if any).
    pub(crate) fn def_type<'a>(&self, m: &'a Module) -> Option<&'a Type> {
        let idx = self.def_type_idx(m);
        if idx != m.void_type_idx() {
            Some(m.type_(idx))
        } else {
            None
        }
    }

    /// Returns the type index of the local variable defined by the instruction.
    ///
    /// If the instruction doesn't define a type then the type index for [Type::Void] is returned.
    pub(crate) fn def_type_idx(&self, m: &Module) -> TypeIdx {
        match self {
            Self::Load(li) => li.type_idx(),
            Self::LookupGlobal(..) => m.ptr_type_idx(),
            Self::LoadTraceInput(li) => li.ty_idx(),
            Self::Call(ci) => ci.target().func_type(m).ret_type_idx(),
            Self::PtrAdd(..) => m.ptr_type_idx(),
            Self::Store(..) => m.void_type_idx(),
            Self::Add(ai) => ai.type_idx(m),
            Self::Icmp(_) => m.int8_type_idx(), // always returns a 0/1 valued byte.
            Self::Guard(..) => m.void_type_idx(),
        }
    }

    /// Returns the size of the local variable that this instruction defines (if any).
    ///
    /// # Panics
    ///
    /// Panics if:
    ///  - The instruction defines no local variable.
    ///  - The instruction defines an unsized local variable.
    pub(crate) fn def_byte_size(&self, m: &Module) -> usize {
        if let Some(ty) = self.def_type(m) {
            if let Some(size) = ty.byte_size() {
                size
            } else {
                panic!()
            }
        } else {
            panic!()
        }
    }
}

impl JitIRDisplay for Instruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        // If the instruction defines a value print it like an assignment.
        if let Some(dt) = self.def_type(m) {
            nums.idx(self).to_string_impl(m, s, nums)?;
            s.push_str(": ");
            dt.to_string_impl(m, s, nums)?;
            s.push_str(" = ");
        }
        // Then print the instruction itself along with its parameters.
        match self {
            Self::Load(i) => i.to_string_impl(m, s, nums)?,
            Self::LookupGlobal(i) => i.to_string_impl(m, s, nums)?,
            Self::LoadTraceInput(i) => i.to_string_impl(m, s, nums)?,
            Self::Call(i) => i.to_string_impl(m, s, nums)?,
            Self::PtrAdd(i) => i.to_string_impl(m, s, nums)?,
            Self::Store(i) => i.to_string_impl(m, s, nums)?,
            Self::Add(i) => i.to_string_impl(m, s, nums)?,
            Self::Icmp(i) => i.to_string_impl(m, s, nums)?,
            Self::Guard(i) => i.to_string_impl(m, s, nums)?,
        }
        Ok(())
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
instr!(LookupGlobal, LookupGlobalInstruction);
instr!(Store, StoreInstruction);
instr!(LoadTraceInput, LoadTraceInputInstruction);
instr!(Call, CallInstruction);
instr!(PtrAdd, PtrAddInstruction);
// FIXME: Use a macro for all binary operations?
instr!(Add, AddInstruction);
instr!(Icmp, IcmpInstruction);
instr!(Guard, GuardInstruction);

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
    // FIXME: why do we need to provide a type index? Can't we get that from the operand?
    pub(crate) fn new(op: Operand, ty_idx: TypeIdx) -> LoadInstruction {
        LoadInstruction {
            op: PackedOperand::new(&op),
            ty_idx,
        }
    }

    /// Return the pointer operand.
    pub(crate) fn operand(&self) -> Operand {
        self.op.unpack()
    }

    /// Returns the type of the value to be loaded.
    pub(crate) fn type_<'a>(&self, m: &'a Module) -> &'a Type {
        m.type_(self.ty_idx)
    }

    /// Returns the type index of the loaded value.
    pub(crate) fn type_idx(&self) -> TypeIdx {
        self.ty_idx
    }
}

impl JitIRDisplay for LoadInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("Load ");
        self.operand().to_string_impl(m, s, nums)
    }
}

/// The `LoadTraceInput` instruction.
///
/// ## Semantics
///
/// Loads a trace input out of the trace input struct. The variable is loaded from the specified
/// offset (`off`) and the resulting local variable is of the type indicated by the `ty_idx`.
///
/// FIXME (maybe): If we added a third `TraceInput` storage class to the register allocator, could
/// we kill this instruction kind entirely?
#[derive(Debug)]
#[repr(packed)]
pub struct LoadTraceInputInstruction {
    /// The byte offset to load from in the trace input struct.
    off: u32,
    /// The type of the resulting local variable.
    ty_idx: TypeIdx,
}

impl JitIRDisplay for LoadTraceInputInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str(&format!("LoadTraceInput {}, ", self.off()));
        m.type_(self.ty_idx).to_string_impl(m, s, nums)?;
        Ok(())
    }
}

impl LoadTraceInputInstruction {
    pub(crate) fn new(off: u32, ty_idx: TypeIdx) -> LoadTraceInputInstruction {
        Self { off, ty_idx }
    }

    pub(crate) fn ty_idx(&self) -> TypeIdx {
        self.ty_idx
    }

    pub(crate) fn off(&self) -> u32 {
        self.off
    }
}

/// The operands for a [Instruction::LoadGlobal]
///
/// # Semantics
///
/// Loads a value from a given global variable.
///
/// FIXME: Codegenning this instruction leads to unoptimial code, since all this does is write a
/// constant pointer into a register only to immediately use that register in the following
/// instruction. We'd rather want to inline the constant into the next instruction. So instead of:
/// ```ignore
/// mov rax, 0x123abc
/// mov [rax], 0x1
/// ```
/// we would get
/// ```ignore
/// mov [0x123abc], 0x1
/// ```
/// However, this requires us to change the JIT IR to allow globals inside operands (we don't want
/// to implement a special global version for each instruction, e.g. LoadGlobal/StoreGlobal/etc).
/// The easiest way to do this is to make globals a subclass of constants, similarly to what LLVM
/// does.
#[derive(Debug)]
pub struct LookupGlobalInstruction {
    /// The pointer to load from.
    global_decl_idx: GlobalDeclIdx,
}

impl LookupGlobalInstruction {
    pub(crate) fn new(global_decl_idx: GlobalDeclIdx) -> Result<Self, CompilationError> {
        Ok(Self { global_decl_idx })
    }

    pub(crate) fn decl<'a>(&self, m: &'a Module) -> &'a GlobalDecl {
        m.global_decl(self.global_decl_idx)
    }
}

impl JitIRDisplay for LookupGlobalInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("LookupGlobal ");
        m.global_decl(self.global_decl_idx)
            .to_string_impl(m, s, nums)
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
    target: FuncDeclIdx,
    /// The first argument to the call, if present. Undefined if not present.
    arg1: PackedOperand,
    /// Extra arguments, if the call requires more than a single argument.
    extra: ExtraArgsIdx,
}

impl JitIRDisplay for CallInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        let decl = m.func_decl(self.target);
        s.push_str("Call @");
        s.push_str(decl.name());

        s.push_str("(");
        let num_args = decl.func_type(m).num_args();
        for ai in 0..num_args {
            self.operand(m, ai).to_string_impl(m, s, nums)?;
            if ai != num_args - 1 {
                s.push_str(", ");
            }
        }
        s.push_str(")");

        Ok(())
    }
}

impl CallInstruction {
    pub(crate) fn new(
        m: &mut Module,
        target: FuncDeclIdx,
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
            target,
            arg1,
            extra,
        })
    }

    fn arg1(&self) -> PackedOperand {
        let unaligned = ptr::addr_of!(self.arg1);
        unsafe { ptr::read_unaligned(unaligned) }
    }

    /// Return the [FuncDeclIdx] of the callee.
    pub(crate) fn target(&self) -> FuncDeclIdx {
        self.target
    }

    /// Fetch the operand at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the operand index is out of bounds.
    pub(crate) fn operand(&self, jit_mod: &Module, idx: usize) -> Operand {
        #[cfg(debug_assertions)]
        {
            let ft = self.target.func_type(jit_mod);
            debug_assert!(ft.num_args() > idx);
        }
        if idx == 0 {
            if self.target().func_type(jit_mod).num_args() > 0 {
                self.arg1().unpack()
            } else {
                // Avoid returning an undefined operand. Storage always exists for one argument,
                // even if the function accepts no arguments.
                panic!();
            }
        } else {
            jit_mod.extra_args[usize::from(self.extra.0) + idx - 1].clone()
        }
    }
}

/// The operands for a [Instruction::Store]
///
/// # Semantics
///
/// Stores a value into a pointer.
///
#[derive(Debug)]
pub struct StoreInstruction {
    /// The value to store.
    val: PackedOperand,
    /// The pointer to store into.
    ptr: PackedOperand,
}

impl StoreInstruction {
    pub(crate) fn new(val: Operand, ptr: Operand) -> Self {
        // FIXME: assert type of pointer
        Self {
            val: PackedOperand::new(&val),
            ptr: PackedOperand::new(&ptr),
        }
    }

    /// Returns the value operand: i.e. the thing that is going to be stored.
    pub(crate) fn val(&self) -> Operand {
        self.val.unpack()
    }

    /// Returns the pointer operand: i.e. where to store the thing.
    pub(crate) fn ptr(&self) -> Operand {
        self.ptr.unpack()
    }
}

impl JitIRDisplay for StoreInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("Store ");
        self.val.unpack().to_string_impl(m, s, nums)?;
        s.push_str(", ");
        self.ptr.unpack().to_string_impl(m, s, nums)?;
        Ok(())
    }
}

/// A pointer offsetting instruction.
///
/// # Semantics
///
/// Returns a pointer value that is the result of adding the specified (byte) offset to the input
/// pointer operand.
#[derive(Debug)]
#[repr(packed)]
pub struct PtrAddInstruction {
    /// The pointer to offset
    ptr: PackedOperand,
    /// The offset.
    off: u32,
}

impl PtrAddInstruction {
    pub(crate) fn ptr(&self) -> Operand {
        let ptr = self.ptr;
        ptr.unpack()
    }

    pub(crate) fn offset(&self) -> u32 {
        self.off
    }

    pub(crate) fn new(ptr: Operand, off: u32) -> Self {
        Self {
            ptr: PackedOperand::new(&ptr),
            off,
        }
    }
}

impl JitIRDisplay for PtrAddInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("PtrAdd ");
        self.ptr().to_string_impl(m, s, nums)?;
        s.push_str(", ");
        s.push_str(&self.offset().to_string());
        Ok(())
    }
}

/// The operands for a [Instruction::Add]
///
/// # Semantics
///
/// Adds two operands together.
///
#[derive(Debug)]
pub struct AddInstruction {
    op1: PackedOperand,
    op2: PackedOperand,
}

impl AddInstruction {
    pub(crate) fn new(op1: Operand, op2: Operand) -> Self {
        Self {
            op1: PackedOperand::new(&op1),
            op2: PackedOperand::new(&op2),
        }
    }

    pub(crate) fn op1(&self) -> Operand {
        self.op1.unpack()
    }

    pub(crate) fn op2(&self) -> Operand {
        self.op2.unpack()
    }

    pub(crate) fn type_<'a>(&self, m: &'a Module) -> &'a Type {
        self.op1.unpack().type_(m)
    }

    /// Returns the type index of the operands being added.
    pub(crate) fn type_idx(&self, m: &Module) -> TypeIdx {
        self.op1.unpack().type_idx(m)
    }
}

impl JitIRDisplay for AddInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("Add ");
        self.op1().to_string_impl(m, s, nums)?;
        s.push_str(", ");
        self.op2().to_string_impl(m, s, nums)?;
        Ok(())
    }
}

/// The operand for a [Instruction::Icmp]
///
/// # Semantics
///
/// Compares two integer operands according to a predicate (e.g. greater-than). Defines a local
/// variable that dictates the truth of the comparison.
///
#[derive(Debug)]
pub struct IcmpInstruction {
    left: PackedOperand,
    pred: Predicate,
    right: PackedOperand,
}

impl IcmpInstruction {
    pub(crate) fn new(op1: Operand, pred: Predicate, op2: Operand) -> Self {
        Self {
            left: PackedOperand::new(&op1),
            pred,
            right: PackedOperand::new(&op2),
        }
    }

    /// Returns the left-hand-side of the comparison.
    ///
    /// E.g. in `x <= y`, it's `x`.
    pub(crate) fn left(&self) -> Operand {
        self.left.unpack()
    }

    /// Returns the right-hand-side of the comparison.
    ///
    /// E.g. in `x <= y`, it's `y`.
    pub(crate) fn right(&self) -> Operand {
        self.right.unpack()
    }

    /// Returns the predicate of the comparison.
    ///
    /// E.g. in `x <= y`, it's `<=`.
    pub(crate) fn predicate(&self) -> Predicate {
        self.pred
    }
}

impl JitIRDisplay for IcmpInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("Icmp ");
        self.left().to_string_impl(m, s, nums)?;
        s.push_str(", ");
        self.right().to_string_impl(m, s, nums)?;
        Ok(())
    }
}

/// The operand for a [Instruction::Guard]
///
/// # Semantics
///
/// Guards a trace against diverging execution. The remainder of the trace will be compiled under
/// the assumption that (at runtime) the guard condition is true. If the guard condition is false,
/// then execution may not continue, and deoptimisation must occur.
///
#[derive(Debug)]
pub struct GuardInstruction {
    /// The condition to guard against.
    cond: PackedOperand,
    /// The expected outcome of the condition.
    expect: bool,
}

impl GuardInstruction {
    pub(crate) fn new(cond: Operand, expect: bool) -> Self {
        GuardInstruction {
            cond: PackedOperand::new(&cond),
            expect,
        }
    }

    pub(crate) fn cond(&self) -> Operand {
        self.cond.unpack()
    }

    pub(crate) fn expect(&self) -> bool {
        self.expect
    }
}

impl JitIRDisplay for GuardInstruction {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("Guard ");
        self.cond().to_string_impl(m, s, nums)?;
        s.push_str(", ");
        if self.expect {
            s.push_str("true");
        } else {
            s.push_str("false");
        }
        Ok(())
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
    instrs: Vec<Instruction>, // FIXME: this should be a TiVec.
    /// The extra argument table.
    ///
    /// Used when a [CallInstruction]'s arguments don't fit inline.
    ///
    /// An [ExtraArgsIdx] describes an index into this.
    extra_args: Vec<Operand>,
    /// The constant table.
    ///
    /// A [ConstIdx] describes an index into this.
    consts: TiVec<ConstIdx, Constant>,
    /// The type table.
    ///
    /// A [TypeIdx] describes an index into this.
    types: TiVec<TypeIdx, Type>,
    /// The type index of the void type. Cached for convinience.
    void_type_idx: TypeIdx,
    /// The type index of a pointer type. Cached for convinience.
    ptr_type_idx: TypeIdx,
    /// The type index of an 8-bit integer. Cached for convinience.
    int8_type_idx: TypeIdx,
    /// The function declaration table.
    ///
    /// These are declarations of externally compiled functions that the JITted trace might need to
    /// call.
    ///
    /// A [FuncDeclIdx] is an index into this.
    func_decls: TiVec<FuncDeclIdx, FuncDecl>,
    /// The global variable declaration table.
    ///
    /// This is a collection of externally defined global variables that the trace may need to
    /// reference. Because they are externally initialised, these are *declarations*.
    global_decls: TiVec<GlobalDeclIdx, GlobalDecl>,
}

impl Module {
    /// Create a new [Module] with the specified name.
    pub fn new(name: String) -> Self {
        // Create some commonly used types ahead of time. Aside from being convenient, this allows
        // us to find their (now statically known) indices in scenarios where Rust forbids us from
        // holding a mutable reference to the Module (and thus we cannot use [Module::type_idx]).
        let mut types = TiVec::new();
        let void_type_idx = types.len().into();
        types.push(Type::Void);
        let ptr_type_idx = types.len().into();
        types.push(Type::Ptr);
        let int8_type_idx = types.len().into();
        types.push(Type::Integer(IntegerType::new(8)));

        Self {
            name,
            instrs: Vec::new(),
            extra_args: Vec::new(),
            consts: TiVec::new(),
            types,
            void_type_idx,
            ptr_type_idx,
            int8_type_idx,
            func_decls: TiVec::new(),
            global_decls: TiVec::new(),
        }
    }

    /// Returns the type index of [Type::Void].
    pub(crate) fn void_type_idx(&self) -> TypeIdx {
        self.void_type_idx
    }

    /// Returns the type index of [Type::Ptr].
    pub(crate) fn ptr_type_idx(&self) -> TypeIdx {
        self.ptr_type_idx
    }

    /// Returns the type index of an 8-bit integer.
    pub(crate) fn int8_type_idx(&self) -> TypeIdx {
        self.int8_type_idx
    }

    /// Return the instruction at the specified index.
    pub(crate) fn instr(&self, idx: InstrIdx) -> &Instruction {
        &self.instrs[usize::try_from(idx).unwrap()]
    }

    /// Return the [Type] for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn type_(&self, idx: TypeIdx) -> &Type {
        &self.types[idx]
    }

    /// Return the global declaration for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn global_decl<'a>(&'a self, idx: GlobalDeclIdx) -> &'a GlobalDecl {
        &self.global_decls[idx]
    }

    /// Push an instruction to the end of the [Module].
    ///
    /// # Panics
    ///
    /// If `instr` would overflow the index type.
    pub(crate) fn push(&mut self, instr: Instruction) {
        assert!(InstrIdx::new(self.instrs.len()).is_ok());
        self.instrs.push(instr);
    }

    /// Push an instruction to the end of the [Module] and create a local variable [Operand] out of
    /// the value that the instruction defines.
    ///
    /// This is useful for forwarding the local variable a instruction defines as operand of a
    /// subsequent instruction: an idiom used a lot (but not exclusively) in testing.
    ///
    /// This must only be used for instructions that define a local variable. If you want to push
    /// an instruction that doesn't define a value, or it does, but you don't want it as an
    /// [Operand], use [Module::push()] instead.
    ///
    /// # Panics
    ///
    /// Panics if the instruction doesn't define a local variable that we could use to build an
    /// [Operand] or if `instr` would overflow the index type.
    pub(crate) fn push_and_make_operand(
        &mut self,
        instr: Instruction,
    ) -> Result<Operand, CompilationError> {
        assert!(InstrIdx::new(self.instrs.len()).is_ok());
        if !instr.is_def() {
            panic!();
        }
        let ret = Operand::Local(InstrIdx::new(self.len())?);
        self.instrs.push(instr);
        Ok(ret)
    }

    /// Returns the number of [Instruction]s in the [Module].
    pub(crate) fn len(&self) -> usize {
        self.instrs.len()
    }

    /// Stringify the module.
    pub(crate) fn to_string(&self) -> Result<String, Box<dyn Error>> {
        (self as &dyn JitIRDisplay).to_string(self)
    }

    /// Print the [Module] to `stderr`.
    pub(crate) fn dump(&self) -> Result<(), Box<dyn Error>> {
        eprintln!("{}", self.to_string()?);
        Ok(())
    }

    /// Returns a reference to the instruction stream.
    pub(crate) fn instrs(&self) -> &Vec<Instruction> {
        &self.instrs
    }

    /// Push a slice of extra arguments into the extra arg table.
    ///
    /// # Panics
    ///
    /// If `ops` would overflow the index type.
    fn push_extra_args(&mut self, ops: &[Operand]) -> Result<ExtraArgsIdx, CompilationError> {
        assert!(ExtraArgsIdx::new(self.types.len()).is_ok());
        let idx = self.extra_args.len();
        self.extra_args.extend_from_slice(ops); // FIXME: this clones.
        ExtraArgsIdx::new(idx)
    }

    /// Push a new type into the type table and return its index.
    ///
    /// The type must not already exist in the module's type table.
    ///
    /// # Panics
    ///
    /// If `ty` would overflow the index type.
    fn push_type(&mut self, ty: Type) -> Result<TypeIdx, CompilationError> {
        #[cfg(debug_assertions)]
        {
            for et in &self.types {
                debug_assert_ne!(et, &ty, "type already exists");
            }
        }
        assert!(TypeIdx::new(self.types.len()).is_ok());
        let idx = self.types.len();
        self.types.push(ty);
        Ok(TypeIdx::new(idx)?)
    }

    /// Push a new function declaration into the function declaration table and return its index.
    ///
    /// # Panics
    ///
    /// If `func_decl` would overflow the index type.
    fn push_func_decl(&mut self, func_decl: FuncDecl) -> Result<FuncDeclIdx, CompilationError> {
        assert!(FuncDeclIdx::new(self.func_decls.len()).is_ok());
        let idx = self.func_decls.len();
        self.func_decls.push(func_decl);
        Ok(FuncDeclIdx::new(idx)?)
    }

    /// Push a new constant into the constant table and return its index.
    ///
    /// # Panics
    ///
    /// If `constant` would overflow the index type.
    pub fn push_const(&mut self, constant: Constant) -> Result<ConstIdx, CompilationError> {
        assert!(ConstIdx::new(self.consts.len()).is_ok());
        let idx = self.consts.len();
        self.consts.push(constant);
        ConstIdx::new(idx)
    }

    /// Push a new declaration into the global variable declaration table and return its index.
    ///
    /// # Panics
    ///
    /// If `decl` would overflow the index type.
    pub fn push_global_decl(
        &mut self,
        decl: GlobalDecl,
    ) -> Result<GlobalDeclIdx, CompilationError> {
        assert!(GlobalDeclIdx::new(self.global_decls.len()).is_ok());
        let idx = self.global_decls.len();
        self.global_decls.push(decl);
        GlobalDeclIdx::new(idx)
    }

    /// Get the index of a constant, inserting it in the constant table if necessary.
    pub fn const_idx(&mut self, c: &Constant) -> Result<ConstIdx, CompilationError> {
        // FIXME: can we optimise this?
        if let Some(idx) = self.consts.iter().position(|tc| tc == c) {
            Ok(ConstIdx::new(idx)?)
        } else {
            // const table miss, we need to insert it.
            self.push_const(c.clone())
        }
    }

    /// Return the const for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn const_<'a>(&'a self, idx: ConstIdx) -> &'a Constant {
        &self.consts[idx]
    }

    /// Get the index of a type, inserting it into the type table if necessary.
    pub(crate) fn type_idx(&mut self, t: &Type) -> Result<TypeIdx, CompilationError> {
        // FIXME: can we optimise this?
        if let Some(idx) = self.types.position(|tt| tt == t) {
            Ok(idx)
        } else {
            // type table miss, we need to insert it.
            self.push_type(t.clone())
        }
    }

    /// Get the index of a function declaration, inserting it into the func decl table if necessary.
    pub(crate) fn func_decl_idx(&mut self, d: &FuncDecl) -> Result<FuncDeclIdx, CompilationError> {
        // FIXME: can we optimise this?
        if let Some(idx) = self.func_decls.position(|td| td == d) {
            Ok(idx)
        } else {
            // type table miss, we need to insert it.
            self.push_func_decl(d.clone())
        }
    }

    /// Get the index of a global, inserting it into the global declaration table if necessary.
    pub(crate) fn global_decl_idx(
        &mut self,
        g: &GlobalDecl,
    ) -> Result<GlobalDeclIdx, CompilationError> {
        // FIXME: can we optimise this?
        if let Some(idx) = self.global_decls.position(|tg| tg == g) {
            Ok(idx)
        } else {
            // global decl table miss, we need to insert it.
            self.push_global_decl(g.clone())
        }
    }

    /// Return the [FuncDecl] for the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds
    pub(crate) fn func_decl(&self, idx: FuncDeclIdx) -> &FuncDecl {
        &self.func_decls[idx]
    }
}

impl JitIRDisplay for Module {
    fn to_string_impl<'a>(
        &self,
        m: &Module,
        s: &mut String,
        nums: &LocalNumbers<'a>,
    ) -> Result<(), Box<dyn Error>> {
        s.push_str("; ");
        s.push_str(&self.name);

        s.push_str("\n\n; globals\n");
        for g in &self.global_decls {
            g.to_string_impl(m, s, nums)?;
            if g.is_threadlocal() {
                s.push_str("    ; thread local");
            }
            s.push('\n');
        }

        s.push_str("\nentry:\n");
        for (idx, instr) in self.instrs().iter().enumerate() {
            s.push_str("    ");
            instr.to_string_impl(m, s, nums)?;
            if idx != m.len() - 1 {
                s.push('\n');
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand() {
        let op = PackedOperand::new(&Operand::Local(InstrIdx(192)));
        assert_eq!(op.unpack(), Operand::Local(InstrIdx(192)));

        let op = PackedOperand::new(&Operand::Local(InstrIdx(0x7fff)));
        assert_eq!(op.unpack(), Operand::Local(InstrIdx(0x7fff)));

        let op = PackedOperand::new(&Operand::Local(InstrIdx(0)));
        assert_eq!(op.unpack(), Operand::Local(InstrIdx(0)));

        let op = PackedOperand::new(&Operand::Const(ConstIdx(192)));
        assert_eq!(op.unpack(), Operand::Const(ConstIdx(192)));

        let op = PackedOperand::new(&Operand::Const(ConstIdx(0x7fff)));
        assert_eq!(op.unpack(), Operand::Const(ConstIdx(0x7fff)));

        let op = PackedOperand::new(&Operand::Const(ConstIdx(0)));
        assert_eq!(op.unpack(), Operand::Const(ConstIdx(0)));
    }

    #[test]
    fn use_case_update_instr() {
        let mut prog: Vec<Instruction> = vec![
            LoadTraceInputInstruction::new(0, TypeIdx::new(0).unwrap()).into(),
            LoadTraceInputInstruction::new(8, TypeIdx::new(0).unwrap()).into(),
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
        assert_eq!(mem::size_of::<StoreInstruction>(), 4);
        assert_eq!(mem::size_of::<LoadInstruction>(), 6);
        assert_eq!(mem::size_of::<LookupGlobalInstruction>(), 3);
        assert_eq!(mem::size_of::<PtrAddInstruction>(), 6);
        assert!(mem::size_of::<Instruction>() <= mem::size_of::<u64>());
    }

    #[test]
    fn extra_call_args() {
        // Set up a function to call.
        let mut jit_mod = Module::new("test".into());
        let i32_tyidx = jit_mod
            .push_type(Type::Integer(IntegerType::new(32)))
            .unwrap();
        let func_ty = Type::Func(FuncType::new(vec![i32_tyidx; 3], i32_tyidx, false));
        let func_ty_idx = jit_mod.push_type(func_ty).unwrap();
        let func_decl = FuncDecl::new("foo".to_owned(), func_ty_idx);
        let func_decl_idx = jit_mod.push_func_decl(func_decl).unwrap();

        // Build a call to the function.
        let args = vec![
            Operand::Local(InstrIdx(0)), // inline arg
            Operand::Local(InstrIdx(1)), // first extra arg
            Operand::Local(InstrIdx(2)),
        ];
        let ci = CallInstruction::new(&mut jit_mod, func_decl_idx, &args).unwrap();

        // Now request the operands and check they all look as they should.
        assert_eq!(ci.operand(&jit_mod, 0), Operand::Local(InstrIdx(0)));
        assert_eq!(ci.operand(&jit_mod, 1), Operand::Local(InstrIdx(1)));
        assert_eq!(ci.operand(&jit_mod, 2), Operand::Local(InstrIdx(2)));
        assert_eq!(
            jit_mod.extra_args,
            vec![Operand::Local(InstrIdx(1)), Operand::Local(InstrIdx(2))]
        );
    }

    #[test]
    #[should_panic]
    fn call_args_out_of_bounds() {
        // Set up a function to call.
        let mut jit_mod = Module::new("test".into());
        let arg_ty_idxs = vec![jit_mod.ptr_type_idx(); 3];
        let ret_ty_idx = jit_mod.type_idx(&Type::Void).unwrap();
        let func_ty = FuncType::new(arg_ty_idxs, ret_ty_idx, false);
        let func_ty_idx = jit_mod.type_idx(&Type::Func(func_ty)).unwrap();
        let func_decl_idx = jit_mod
            .func_decl_idx(&FuncDecl::new("blah".into(), func_ty_idx))
            .unwrap();

        // Now build a call to the function.
        let args = vec![
            Operand::Local(InstrIdx(0)), // inline arg
            Operand::Local(InstrIdx(1)), // first extra arg
            Operand::Local(InstrIdx(2)),
        ];
        let ci = CallInstruction::new(&mut jit_mod, func_decl_idx, &args).unwrap();

        // Request an operand with an out-of-bounds index.
        ci.operand(&jit_mod, 3);
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
        assert!(TypeIdx::new(0).is_ok());
        assert!(TypeIdx::new(1).is_ok());
        assert!(TypeIdx::new(0x1234).is_ok());
        assert!(TypeIdx::new(0x123456).is_ok());
        assert!(TypeIdx::new(0xffffff).is_ok());
    }

    #[test]
    fn index24_doesnt_fit() {
        assert!(TypeIdx::new(0x1000000).is_err());
        assert!(TypeIdx::new(0x1234567).is_err());
        assert!(TypeIdx::new(0xeddedde).is_err());
        assert!(TypeIdx::new(usize::MAX).is_err());
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

    #[test]
    fn void_type_size() {
        assert_eq!(Type::Void.byte_size(), Some(0));
    }

    #[test]
    fn int_type_size() {
        assert_eq!(Type::Integer(IntegerType::new(0)).byte_size(), Some(0));
        for i in 1..8 {
            assert_eq!(Type::Integer(IntegerType::new(i)).byte_size(), Some(1));
        }
        for i in 9..16 {
            assert_eq!(Type::Integer(IntegerType::new(i)).byte_size(), Some(2));
        }
        assert_eq!(Type::Integer(IntegerType::new(127)).byte_size(), Some(16));
        assert_eq!(Type::Integer(IntegerType::new(128)).byte_size(), Some(16));
        assert_eq!(Type::Integer(IntegerType::new(129)).byte_size(), Some(17));
    }

    #[cfg(debug_assertions)]
    #[should_panic(expected = "type already exists")]
    #[test]
    fn push_duplicate_type() {
        let mut jit_mod = Module::new("test".into());
        let _ = jit_mod.push_type(Type::Void);
        let _ = jit_mod.push_type(Type::Void);
    }

    #[test]
    fn stringify_consts() {
        let m = Module::new("test".into());
        assert_eq!(Constant::U32(123).to_string(&m).unwrap(), "123");
        assert_eq!(Constant::Usize(123).to_string(&m).unwrap(), "123");
    }

    #[test]
    fn print_module() {
        let mut m = Module::new("test".into());
        m.push(LoadTraceInputInstruction::new(0, m.int8_type_idx()).into());
        m.push(LoadTraceInputInstruction::new(8, m.int8_type_idx()).into());
        m.push(LoadTraceInputInstruction::new(16, m.int8_type_idx()).into());
        m.push_global_decl(GlobalDecl::new("some_global".into(), false))
            .unwrap();
        m.push_global_decl(GlobalDecl::new("some_thread_local".into(), true))
            .unwrap();
        let got = m.to_string().unwrap();
        let expect = [
            "; test",
            "",
            "; globals",
            "@some_global",
            "@some_thread_local    ; thread local",
            "",
            "entry:",
            "    %0: i8 = LoadTraceInput 0, i8",
            "    %1: i8 = LoadTraceInput 8, i8",
            "    %2: i8 = LoadTraceInput 16, i8",
        ]
        .join("\n");
        assert_eq!(got, expect);
    }
}
