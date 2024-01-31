//! Yk's AOT IR deserialiser.
//!
//! This is a parser for the on-disk (in the ELF binary) IR format used to express the
//! (immutable) ahead-of-time compiled interpreter.

use byteorder::{NativeEndian, ReadBytesExt};
use deku::prelude::*;
use std::{cell::RefCell, error::Error, ffi::CStr, fs, io::Cursor, path::PathBuf};

/// A magic number that all bytecode payloads begin with.
const MAGIC: u32 = 0xedd5f00d;
/// The version of the bytecode format.
const FORMAT_VERSION: u32 = 0;

/// The symbol name of the control point function (after ykllvm has transformed it).
const CONTROL_POINT_NAME: &str = "__ykrt_control_point";

// Generate common methods for index types.
macro_rules! index {
    ($struct:ident) => {
        impl $struct {
            #[allow(dead_code)] // FIXME: remove when constants and func args are implemented.
            pub(crate) fn new(v: usize) -> Self {
                Self(v)
            }

            pub(crate) fn to_usize(&self) -> usize {
                self.0
            }
        }
    };
}

#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct FuncIdx(usize);
index!(FuncIdx);

#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypeIdx(usize);
index!(TypeIdx);

/// A basic block index.
///
/// One of these is an index into [Function::blocks].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BlockIdx(usize);
index!(BlockIdx);

/// An instruction index.
///
/// One of these is an index into [Block::instrs].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct InstrIdx(usize);
index!(InstrIdx);

/// A constant index.
///
/// One of these is an index into [Module::consts].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ConstIdx(usize);
index!(ConstIdx);

/// A function argument index.
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ArgIdx(usize);
index!(ArgIdx);

fn deserialise_string(v: Vec<u8>) -> Result<String, DekuError> {
    let err = Err(DekuError::Parse("failed to parse string".to_owned()));
    match CStr::from_bytes_until_nul(v.as_slice()) {
        Ok(c) => match c.to_str() {
            Ok(s) => Ok(s.to_owned()),
            Err(_) => err,
        },
        _ => err,
    }
}

/// A trait for converting in-memory data-structures into a human-readable textual format.
///
/// This is modelled on [`std::fmt::Display`], but a reference to the module is always passed down
/// so that constructs that require lookups into the module's tables from stringification have
/// access to them.
///
/// The way we implement this (returning a `String`) is inefficient, but it doesn't hugely matter,
/// as the human-readable format is only provided as a debugging aid.
pub(crate) trait IRDisplay {
    /// Return a human-readable string.
    fn to_str(&self, m: &Module) -> String;

    /// Print myself to stderr in human-readable form.
    ///
    /// This is provided as a debugging convenience.
    fn dump(&self, m: &Module) {
        eprintln!("{}", self.to_str(m));
    }
}

/// An instruction opcode.
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[deku(type = "u8")]
pub(crate) enum Opcode {
    Nop = 0,
    Load,
    Store,
    Alloca,
    Call,
    GetElementPtr,
    Br,
    CondBr,
    Icmp,
    BinaryOperator,
    Ret,
    InsertValue,
    Unimplemented = 255,
}

impl IRDisplay for Opcode {
    fn to_str(&self, _m: &Module) -> String {
        format!("{:?}", self).to_lowercase()
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct ConstantOperand {
    const_idx: ConstIdx,
}

impl IRDisplay for ConstantOperand {
    fn to_str(&self, m: &Module) -> String {
        m.consts[self.const_idx.to_usize()].to_str(m)
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug, Hash, Eq, PartialEq)]
pub(crate) struct InstructionID {
    #[deku(skip)] // computed after deserialisation.
    func_idx: FuncIdx,
    bb_idx: BlockIdx,
    inst_idx: InstrIdx,
}

impl InstructionID {
    pub(crate) fn new(func_idx: FuncIdx, bb_idx: BlockIdx, inst_idx: InstrIdx) -> Self {
        Self {
            func_idx,
            bb_idx,
            inst_idx,
        }
    }
}

#[derive(Debug)]
pub(crate) struct BlockID {
    func_idx: FuncIdx,
    block_idx: BlockIdx,
}

impl BlockID {
    pub(crate) fn new(func_idx: FuncIdx, block_idx: BlockIdx) -> Self {
        Self {
            func_idx,
            block_idx,
        }
    }

    pub(crate) fn func_idx(&self) -> FuncIdx {
        self.func_idx
    }

    pub(crate) fn block_idx(&self) -> BlockIdx {
        self.block_idx
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug, Hash, Eq, PartialEq)]
pub(crate) struct LocalVariableOperand(InstructionID);

impl LocalVariableOperand {
    pub(crate) fn instr_id(&self) -> &InstructionID {
        &self.0
    }

    pub(crate) fn instr_id_mut(&mut self) -> &mut InstructionID {
        &mut self.0
    }
}

impl IRDisplay for LocalVariableOperand {
    fn to_str(&self, _m: &Module) -> String {
        format!(
            "${}_{}",
            self.0.bb_idx.to_usize(),
            self.0.inst_idx.to_usize()
        )
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct TypeOperand {
    type_idx: TypeIdx,
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct BlockOperand {
    bb_idx: BlockIdx,
}

impl IRDisplay for BlockOperand {
    fn to_str(&self, _m: &Module) -> String {
        format!("bb{}", self.bb_idx.to_usize())
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct FunctionOperand {
    func_idx: FuncIdx,
}

/// An operand that is an argument to the parent function.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct ArgOperand {
    arg_idx: ArgIdx,
}

impl IRDisplay for ArgOperand {
    fn to_str(&self, _m: &Module) -> String {
        format!("$arg{}", self.arg_idx.to_usize())
    }
}

const OPKIND_CONST: u8 = 0;
const OPKIND_LOCAL_VARIABLE: u8 = 1;
const OPKIND_TYPE: u8 = 2;
const OPKIND_FUNCTION: u8 = 3;
const OPKIND_BLOCK: u8 = 4;
const OPKIND_ARG: u8 = 5;
const OPKIND_UNIMPLEMENTED: u8 = 255;

#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(type = "u8")]
pub(crate) enum Operand {
    #[deku(id = "OPKIND_CONST")]
    Constant(ConstantOperand),
    #[deku(id = "OPKIND_LOCAL_VARIABLE")]
    LocalVariable(LocalVariableOperand),
    #[deku(id = "OPKIND_TYPE")]
    Type(TypeOperand),
    #[deku(id = "OPKIND_FUNCTION")]
    Function(FunctionOperand),
    #[deku(id = "OPKIND_BLOCK")]
    Block(BlockOperand),
    #[deku(id = "OPKIND_ARG")]
    Arg(ArgOperand),
    #[deku(id = "OPKIND_UNIMPLEMENTED")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "deserialise_string")] String),
}

impl Operand {
    /// For a [Self::LocalVariable] operand return the instruction that defines the variable.
    ///
    /// Panics for other kinds of operand.
    ///
    /// OPT: This is expensive.
    pub(crate) fn to_instr<'a>(&self, aotmod: &'a Module) -> &'a Instruction {
        match self {
            Self::LocalVariable(lvo) => {
                let iid = lvo.instr_id();
                &aotmod.funcs[iid.func_idx.to_usize()].blocks[iid.bb_idx.to_usize()].instrs
                    [lvo.instr_id().inst_idx.to_usize()]
            }
            _ => panic!(),
        }
    }

    /// Return the `InstructionID` of a local variable operand. Panics if called on other kinds of
    /// operands.
    pub(crate) fn to_instr_id(&self) -> InstructionID {
        match self {
            Self::LocalVariable(lvo) => {
                let iid = lvo.instr_id();
                InstructionID::new(iid.func_idx, iid.bb_idx, iid.inst_idx)
            }
            _ => panic!(),
        }
    }
}

impl IRDisplay for Operand {
    fn to_str(&self, m: &Module) -> String {
        match self {
            Self::Constant(c) => c.to_str(m),
            Self::LocalVariable(l) => l.to_str(m),
            Self::Type(t) => m.types[t.type_idx.to_usize()].to_str(m),
            Self::Function(f) => m.funcs[f.func_idx.0].name.to_owned(),
            Self::Block(bb) => bb.to_str(m),
            Self::Arg(a) => a.to_str(m),
            Self::Unimplemented(s) => format!("?op<{}>", s),
        }
    }
}

/// A bytecode instruction.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Instruction {
    type_idx: TypeIdx,
    opcode: Opcode,
    #[deku(temp)]
    num_operands: u32,
    #[deku(count = "num_operands")]
    operands: Vec<Operand>,
    /// A variable name, only computed if the instruction is ever printed.
    #[deku(skip)]
    name: RefCell<Option<String>>,
}

impl Instruction {
    /// Returns the operand at the specified index. Panics if the index is out of bounds.
    pub(crate) fn operand(&self, idx: usize) -> &Operand {
        &self.operands[idx]
    }

    /// Return a slice of the remaining operands, starting from the index `from` (inclusive).
    pub(crate) fn remaining_operands(&self, from: usize) -> &[Operand] {
        &self.operands[from..]
    }

    pub(crate) fn opcode(&self) -> Opcode {
        self.opcode
    }

    /// For a call instruction, return the callee.
    ///
    /// # Panics
    ///
    /// Panics if the instruction isn't a call instruction.
    pub(crate) fn callee<'a>(&self) -> FuncIdx {
        debug_assert!(matches!(self.opcode, Opcode::Call));
        let op = self.operand(0);
        match op {
            Operand::Function(fo) => fo.func_idx,
            _ => panic!(),
        }
    }

    pub(crate) fn type_idx(&self) -> TypeIdx {
        self.type_idx
    }

    pub(crate) fn is_store(&self) -> bool {
        self.opcode == Opcode::Store
    }

    pub(crate) fn is_gep(&self) -> bool {
        self.opcode == Opcode::GetElementPtr
    }

    pub(crate) fn is_control_point(&self, aot_mod: &Module) -> bool {
        if self.opcode == Opcode::Call {
            // Call instructions always have at least one operand (the callee), so this is safe.
            let op = &self.operands[0];
            match op {
                Operand::Function(fop) => {
                    return aot_mod.funcs[fop.func_idx.0].name == CONTROL_POINT_NAME;
                }
                _ => todo!(),
            }
        }
        false
    }

    /// Determine if two instructions in the (immutable) AOT IR are the same based on pointer
    /// identity.
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl IRDisplay for Instruction {
    fn to_str(&self, m: &Module) -> String {
        if self.opcode == Opcode::Unimplemented {
            debug_assert!(self.operands.len() == 1);
            if let Operand::Unimplemented(s) = &self.operands[0] {
                return format!("?inst<{}>", s);
            } else {
                // This would be an invalid serialisation.
                panic!();
            }
        }

        if !*m.var_names_computed.borrow() {
            m.compute_variable_names();
        }

        let mut ret = String::new();
        if m.instr_generates_value(self) {
            let name = self.name.borrow();
            // The unwrap cannot fail, as we forced computation of variable names above.
            ret.push_str(&format!(
                "${}: {} = ",
                name.as_ref().unwrap(),
                m.instr_type(self).to_str(m)
            ));
        }
        ret.push_str(&self.opcode.to_str(m));
        if !self.operands.is_empty() {
            ret.push(' ');
        }
        let op_strs = self
            .operands
            .iter()
            .map(|o| o.to_str(m))
            .collect::<Vec<_>>();

        if self.opcode != Opcode::Call {
            ret.push_str(&op_strs.join(", "));
        } else {
            // Put parentheses around the call arguments.
            let mut itr = op_strs.into_iter();
            // unwrap safe: calls must have at least a callee operand.
            ret.push_str(&itr.next().unwrap());
            let rest = itr.collect::<Vec<_>>();
            ret.push('(');
            ret.push_str(&rest.join(", "));
            ret.push(')');
        }

        ret
    }
}

/// A basic block containing bytecode instructions.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Block {
    #[deku(temp)]
    num_instrs: usize,
    #[deku(count = "num_instrs")]
    pub(crate) instrs: Vec<Instruction>,
}

impl IRDisplay for Block {
    fn to_str(&self, m: &Module) -> String {
        let mut ret = String::new();
        for i in &self.instrs {
            ret.push_str(&format!("    {}\n", i.to_str(m)));
        }
        ret
    }
}

/// A function.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Function {
    #[deku(until = "|v: &u8| *v == 0", map = "deserialise_string")]
    name: String,
    type_idx: TypeIdx,
    #[deku(temp)]
    num_blocks: usize,
    #[deku(count = "num_blocks")]
    blocks: Vec<Block>,
}

impl<'a> Function {
    fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Return the block at the specified index, or `None` if the index is out of range.
    pub(crate) fn block(&self, bb_idx: BlockIdx) -> Option<&Block> {
        self.blocks.get(bb_idx.to_usize())
    }

    #[cfg(test)]
    pub(crate) fn new(name: &str, type_idx: TypeIdx) -> Self {
        Self {
            name: name.to_string(),
            type_idx,
            blocks: Vec::new(),
        }
    }
}

impl IRDisplay for Function {
    fn to_str(&self, m: &Module) -> String {
        let ty = &m.types[self.type_idx.to_usize()];
        if let Type::Func(fty) = ty {
            let mut ret = format!(
                "func {}({}",
                self.name,
                fty.arg_ty_idxs
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("$arg{}: {}", i, m.types[t.to_usize()].to_str(m)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if fty.is_vararg {
                ret.push_str(", ...");
            }
            ret.push(')');
            let ret_ty = &m.types[fty.ret_ty.to_usize()];
            if ret_ty != &Type::Void {
                ret.push_str(&format!(" -> {}", ret_ty.to_str(m)));
            }
            if self.is_declaration() {
                // declarations have no body, so print it as such.
                ret.push_str(";\n");
            } else {
                ret.push_str(" {\n");
                for (i, b) in self.blocks.iter().enumerate() {
                    ret.push_str(&format!("  bb{}:\n{}", i, b.to_str(m)));
                }
                ret.push_str("}\n");
            }
            ret
        } else {
            unreachable!("{}", ty.to_str(m)); // Impossible for a function to not be of type `Func`.
        }
    }
}

// A fixed-width two's compliment integer.
//
// Signedness is not specified.
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IntegerType {
    num_bits: u32,
}

impl IntegerType {
    fn const_to_str(&self, c: &Constant) -> String {
        // FIXME: For now we just handle common integer types, but eventually we will need to
        // implement printing of aribitrarily-sized (in bits) integers. Consider using a bigint
        // library so we don't have to do it ourself?
        //
        // This discussion may help:
        // https://rust-lang.zulipchat.com/#narrow/stream/122651-general/topic/.E2.9C.94.20Big.20Integer.20library.20with.20bit.20granularity/near/393733327

        // All of the unwraps below are safe due to:
        debug_assert!(c.bytes.len() * 8 >= usize::try_from(self.num_bits).unwrap());

        let mut c = Cursor::new(&c.bytes);
        match self.num_bits {
            1 => format!("{}i1", c.read_i8().unwrap() & 1),
            8 => format!("{}i8", c.read_i8().unwrap()),
            16 => format!("{}i16", c.read_i16::<NativeEndian>().unwrap()),
            32 => format!("{}i32", c.read_i32::<NativeEndian>().unwrap()),
            64 => format!("{}i64", c.read_i64::<NativeEndian>().unwrap()),
            _ => todo!("{}", self.num_bits),
        }
    }
}

impl IRDisplay for IntegerType {
    fn to_str(&self, _m: &Module) -> String {
        format!("i{}", self.num_bits)
    }
}

#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FuncType {
    /// The number of formal arguments the function takes.
    #[deku(temp)]
    num_args: usize,
    /// Type indices for the function's formal arguments.
    #[deku(count = "num_args")]
    arg_ty_idxs: Vec<TypeIdx>,
    /// Type index of the function's return type.
    ret_ty: TypeIdx,
    /// Is the function vararg?
    is_vararg: bool,
}

impl FuncType {
    pub(crate) fn num_args(&self) -> usize {
        self.arg_ty_idxs.len()
    }

    #[cfg(test)]
    pub(crate) fn new(arg_ty_idxs: Vec<TypeIdx>, ret_ty: TypeIdx, is_vararg: bool) -> Self {
        Self {
            arg_ty_idxs,
            ret_ty,
            is_vararg,
        }
    }
}

#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StructType {
    /// The number of fields the struct has.
    #[deku(temp)]
    num_fields: usize,
    /// The types of the fields.
    #[deku(count = "num_fields")]
    field_ty_idxs: Vec<TypeIdx>,
    /// The bit offsets of the fields (taking into account any required padding for alignment).
    #[deku(count = "num_fields")]
    field_bit_offs: Vec<usize>,
}

impl IRDisplay for StructType {
    fn to_str(&self, m: &Module) -> String {
        let mut s = String::from("{");
        s.push_str(
            &self
                .field_ty_idxs
                .iter()
                .enumerate()
                .map(|(i, ti)| {
                    format!(
                        "{}: {}",
                        self.field_bit_offs[i],
                        m.types[ti.to_usize()].to_str(m)
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
        );
        s.push('}');
        s
    }
}

impl IRDisplay for FuncType {
    fn to_str(&self, m: &Module) -> String {
        format!(
            "func({})",
            self.arg_ty_idxs
                .iter()
                .map(|t| m.types[t.to_usize()].to_str(m))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

const TYKIND_VOID: u8 = 0;
const TYKIND_INTEGER: u8 = 1;
const TYKIND_PTR: u8 = 2;
const TYKIND_FUNC: u8 = 3;
const TYKIND_STRUCT: u8 = 4;
const TYKIND_UNIMPLEMENTED: u8 = 255;

/// A type.
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
#[deku(type = "u8")]
pub(crate) enum Type {
    #[deku(id = "TYKIND_VOID")]
    Void,
    #[deku(id = "TYKIND_INTEGER")]
    Integer(IntegerType),
    #[deku(id = "TYKIND_PTR")]
    Ptr,
    #[deku(id = "TYKIND_FUNC")]
    Func(FuncType),
    #[deku(id = "TYKIND_STRUCT")]
    Struct(StructType),
    #[deku(id = "TYKIND_UNIMPLEMENTED")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "deserialise_string")] String),
}

impl Type {
    fn const_to_str(&self, c: &Constant) -> String {
        match self {
            Self::Void => "void".to_owned(),
            Self::Integer(it) => it.const_to_str(c),
            Self::Ptr => {
                // FIXME: write a stringifier for constant pointers.
                "const_ptr".to_owned()
            }
            Self::Func(_) => unreachable!(), // No such thing as a constant function in our IR.
            Self::Struct(_) => {
                // FIXME: write a stringifier for constant structs.
                "const_struct".to_owned()
            }
            Self::Unimplemented(s) => format!("?cst<{}>", s),
        }
    }
}

impl IRDisplay for Type {
    fn to_str(&self, m: &Module) -> String {
        match self {
            Self::Void => "void".to_owned(),
            Self::Integer(i) => i.to_str(m),
            Self::Ptr => "ptr".to_owned(),
            Self::Func(ft) => ft.to_str(m),
            Self::Struct(st) => st.to_str(m),
            Self::Unimplemented(s) => format!("?ty<{}>", s),
        }
    }
}

/// A constant.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Constant {
    type_idx: TypeIdx,
    #[deku(temp)]
    num_bytes: usize,
    #[deku(count = "num_bytes")]
    bytes: Vec<u8>,
}

impl IRDisplay for Constant {
    fn to_str(&self, m: &Module) -> String {
        m.types[self.type_idx.to_usize()].const_to_str(self)
    }
}

/// A bytecode module.
///
/// This is the top-level container for the bytecode.
#[deku_derive(DekuRead)]
#[derive(Debug, Default)]
pub(crate) struct Module {
    #[deku(assert = "*magic == MAGIC", temp)]
    magic: u32,
    #[deku(assert = "*version == FORMAT_VERSION")]
    version: u32,
    #[deku(temp)]
    num_funcs: usize,
    #[deku(count = "num_funcs")]
    funcs: Vec<Function>,
    #[deku(temp)]
    num_consts: usize,
    #[deku(count = "num_consts")]
    consts: Vec<Constant>,
    #[deku(temp)]
    num_types: usize,
    #[deku(count = "num_types")]
    types: Vec<Type>,
    /// Have local variable names been computed?
    ///
    /// Names are computed on-demand when an instruction is printed for the first time.
    #[deku(skip)]
    var_names_computed: RefCell<bool>,
}

impl Module {
    /// Compute variable names for all instructions that generate a value.
    fn compute_variable_names(&self) {
        debug_assert!(!*self.var_names_computed.borrow());
        // Note that because the on-disk IR is conceptually immutable, so we don't have to worry
        // about keeping the names up to date.
        for f in &self.funcs {
            for (bb_idx, bb) in f.blocks.iter().enumerate() {
                for (inst_idx, inst) in bb.instrs.iter().enumerate() {
                    if self.instr_generates_value(inst) {
                        *inst.name.borrow_mut() = Some(format!("{}_{}", bb_idx, inst_idx));
                    }
                }
            }
        }
        *self.var_names_computed.borrow_mut() = true;
    }

    pub(crate) fn func_idx(&self, find_func: &str) -> Option<FuncIdx> {
        // OPT: create a cache in the Module.
        self.funcs
            .iter()
            .enumerate()
            .find(|(_, f)| f.name == find_func)
            .map(|(f_idx, _)| FuncIdx(f_idx))
    }

    /// Look up a `FuncType` by its index.
    ///
    /// # Panics
    ///
    /// Panics if the type index is either out of bounds, or the corresponding type is not a
    /// function type.
    pub(crate) fn func_ty(&self, func_idx: FuncIdx) -> &FuncType {
        match self.types[self.funcs[func_idx.to_usize()].type_idx.to_usize()] {
            Type::Func(ref ft) => &ft,
            _ => panic!(),
        }
    }

    pub(crate) fn block(&self, bid: &BlockID) -> Option<&Block> {
        self.funcs
            .get(bid.func_idx.to_usize())?
            .block(bid.block_idx)
    }

    /// Fill in the function index of local variable operands of instructions.
    ///
    /// FIXME: It may be possible to do this as we deserialise, instead of after the fact:
    /// https://github.com/sharksforarms/deku/issues/363
    fn compute_local_operand_func_indices(&mut self) {
        for (f_idx, f) in self.funcs.iter_mut().enumerate() {
            for bb in &mut f.blocks {
                for inst in &mut bb.instrs {
                    for op in &mut inst.operands {
                        if let Operand::LocalVariable(ref mut lv) = op {
                            lv.instr_id_mut().func_idx = FuncIdx(f_idx);
                        }
                    }
                }
            }
        }
    }

    /// Get the type of the instruction.
    ///
    /// It is UB to pass an `instr` that is not from the `Module` referenced by `self`.
    pub(crate) fn instr_type(&self, instr: &Instruction) -> &Type {
        &self.types[instr.type_idx.to_usize()]
    }

    // FIXME: rename this to `is_def()`, which we've decided is a beter name.
    // FIXME: also move this to the `Instruction` type.
    fn instr_generates_value(&self, i: &Instruction) -> bool {
        self.instr_type(i) != &Type::Void
    }

    pub(crate) fn to_str(&self) -> String {
        let mut ret = String::new();
        ret.push_str(&format!("# IR format version: {}\n", self.version));
        ret.push_str(&format!("# Num funcs: {}\n", self.funcs.len()));
        ret.push_str(&format!("# Num consts: {}\n", self.consts.len()));
        ret.push_str(&format!("# Num types: {}\n", self.types.len()));

        for func in &self.funcs {
            ret.push_str(&format!("\n{}", func.to_str(self)));
        }
        ret
    }

    #[allow(dead_code)]
    pub(crate) fn dump(&self) {
        eprintln!("{}", self.to_str());
    }
}

#[cfg(test)]
impl Module {
    pub(crate) fn push_func(&mut self, func: Function) -> FuncIdx {
        let idx = self.funcs.len();
        self.funcs.push(func);
        FuncIdx(idx)
    }

    pub(crate) fn push_type(&mut self, ty: Type) -> TypeIdx {
        let idx = self.types.len();
        self.types.push(ty);
        TypeIdx(idx)
    }
}

/// Deserialise an AOT module from the slice `data`.
pub(crate) fn deserialise_module(data: &[u8]) -> Result<Module, Box<dyn Error>> {
    match Module::from_bytes((data, 0)) {
        Ok(((_, _), mut modu)) => {
            modu.compute_local_operand_func_indices();
            Ok(modu)
        }
        Err(e) => Err(e.to_string().into()),
    }
}

/// Deserialise and print IR from an on-disk file.
///
/// Used for support tooling (in turn used by tests too).
pub fn print_from_file(path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let data = fs::read(path)?;
    let ir = deserialise_module(&data)?;
    println!("{}", ir.to_str());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        deserialise_module, deserialise_string, Constant, IntegerType, Opcode, TypeIdx,
        FORMAT_VERSION, MAGIC, OPKIND_BLOCK, OPKIND_CONST, OPKIND_FUNCTION, OPKIND_LOCAL_VARIABLE,
        OPKIND_TYPE, OPKIND_UNIMPLEMENTED, TYKIND_FUNC, TYKIND_INTEGER, TYKIND_PTR, TYKIND_STRUCT,
        TYKIND_UNIMPLEMENTED, TYKIND_VOID,
    };
    use byteorder::{NativeEndian, WriteBytesExt};
    use std::ffi::CString;

    #[cfg(target_pointer_width = "64")]
    fn write_native_usize(d: &mut Vec<u8>, v: usize) {
        d.write_u64::<NativeEndian>(v as u64).unwrap(); // `as` is safe: u64 == usize
    }

    fn write_str(d: &mut Vec<u8>, s: &str) {
        d.extend(s.as_bytes());
        d.push(0); // null terminator.
    }

    /// Note that this test only checks valid IR encodings and not for valid IR semantics. For
    /// example, nonsensical instruction arguments (incorrect arg counts, incorrect arg types etc.)
    /// are not checked.
    ///
    /// FIXME: implement an IR validator for this purpose.
    #[test]
    fn deser_and_display() {
        let mut data = Vec::new();

        // HEADER
        // magic:
        data.write_u32::<NativeEndian>(MAGIC).unwrap();
        // version:
        data.write_u32::<NativeEndian>(FORMAT_VERSION).unwrap();

        // num_functions:
        write_native_usize(&mut data, 2);

        // FUNCTION 0
        // name:
        write_str(&mut data, "foo");
        // type_idx:
        write_native_usize(&mut data, 4);
        // num_blocks:
        write_native_usize(&mut data, 2);

        // BLOCK 0
        // num_instrs:
        write_native_usize(&mut data, 3);

        // INSTRUCTION 0
        // type_idx:
        write_native_usize(&mut data, 2);
        // opcode:
        data.write_u8(Opcode::Alloca as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(1).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_CONST).unwrap();
        // const_idx
        write_native_usize(&mut data, 0);

        // INSTRUCTION 1
        // type_idx:
        write_native_usize(&mut data, 0);
        // opcode:
        data.write_u8(Opcode::Nop as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(0).unwrap();

        // INSTRUCTION 2
        // type_idx:
        write_native_usize(&mut data, 0);
        // opcode:
        data.write_u8(Opcode::CondBr as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(3).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_LOCAL_VARIABLE as u8).unwrap();
        // bb_idx:
        write_native_usize(&mut data, 0);
        // inst_idx:
        write_native_usize(&mut data, 0);
        // OPERAND 1
        // operand_kind:
        data.write_u8(OPKIND_BLOCK as u8).unwrap();
        // bb_idx:
        write_native_usize(&mut data, 0);
        // OPERAND 2
        // operand_kind:
        data.write_u8(OPKIND_BLOCK as u8).unwrap();
        // bb_idx:
        write_native_usize(&mut data, 1);

        // BLOCK 1
        // num_instrs:
        write_native_usize(&mut data, 6);

        // INSTRUCTION 0
        // type_idx:
        write_native_usize(&mut data, 0);
        // opcode:
        data.write_u8(Opcode::Unimplemented as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(1).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_UNIMPLEMENTED as u8).unwrap();
        // unimplemented description:
        write_str(&mut data, "%3 = some_llvm_instruction ...");

        // INSTRUCTION 1
        // type_idx:
        write_native_usize(&mut data, 2);
        // opcode:
        data.write_u8(Opcode::GetElementPtr as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(1).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_CONST as u8).unwrap();
        // const_idx:
        write_native_usize(&mut data, 1);

        // INSTRUCTION 2
        // type_idx:
        write_native_usize(&mut data, 2);
        // opcode:
        data.write_u8(Opcode::Alloca as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(2).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_TYPE).unwrap();
        // type_idx:
        write_native_usize(&mut data, 3);
        // OPERAND 1
        // operand_kind:
        data.write_u8(OPKIND_CONST as u8).unwrap();
        // const_idx:
        write_native_usize(&mut data, 2);

        // INSTRUCTION 3
        // type_idx:
        write_native_usize(&mut data, 2);
        // opcode:
        data.write_u8(Opcode::Call as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(3).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_FUNCTION).unwrap();
        // func_idx:
        write_native_usize(&mut data, 1);
        // OPERAND 1
        // operand_kind:
        data.write_u8(OPKIND_CONST).unwrap();
        // const_idx:
        write_native_usize(&mut data, 2);
        // OPERAND 2
        // operand_kind:
        data.write_u8(OPKIND_CONST).unwrap();
        // const_idx:
        write_native_usize(&mut data, 2);

        // INSTRUCTION 4
        // type_idx:
        write_native_usize(&mut data, 0);
        // opcode:
        data.write_u8(Opcode::Br as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(0).unwrap();

        // INSTRUCTION 5
        // type_idx:
        write_native_usize(&mut data, 6);
        // opcode:
        data.write_u8(Opcode::Nop as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(0).unwrap();

        // FUNCTION 1
        // name:
        write_str(&mut data, "bar");
        // type_idx:
        write_native_usize(&mut data, 5);
        // num_blocks:
        write_native_usize(&mut data, 0);

        // CONSTANTS
        // num_consts:
        write_native_usize(&mut data, 3);

        // CONSTANT 0
        // type_idx:
        write_native_usize(&mut data, 1);
        // num_bytes:
        write_native_usize(&mut data, 0);

        // CONSTANT 1
        // type_idx:
        write_native_usize(&mut data, 3);
        // num_bytes:
        write_native_usize(&mut data, 4);
        // bytes:
        data.write_u32::<NativeEndian>(u32::MAX).unwrap();

        // CONSTANT 2
        // type_idx:
        write_native_usize(&mut data, 3);
        // num_bytes:
        write_native_usize(&mut data, 4);
        // bytes:
        data.write_u32::<NativeEndian>(50).unwrap();

        // TYPES
        // num_types:
        write_native_usize(&mut data, 7);

        // TYPE 0
        // type_kind:
        data.write_u8(TYKIND_VOID).unwrap();

        // TYPE 1
        // type_kind:
        data.write_u8(TYKIND_UNIMPLEMENTED).unwrap();
        // unimplemented description:
        write_str(&mut data, "a_type");

        // TYPE 2
        // type_kind:
        data.write_u8(TYKIND_PTR).unwrap();

        // TYPE 3
        // type_kind:
        data.write_u8(TYKIND_INTEGER).unwrap();
        // num_bits:
        data.write_u32::<NativeEndian>(32).unwrap();

        // TYPE 4
        // type_kind:
        data.write_u8(TYKIND_FUNC).unwrap();
        // num_args:
        write_native_usize(&mut data, 2);
        // arg_ty_idxs:
        write_native_usize(&mut data, 2);
        write_native_usize(&mut data, 3);
        // ret_ty:
        write_native_usize(&mut data, 3);
        // is_vararg:
        data.write_u8(0).unwrap();

        // TYPE 5
        // type_kind:
        data.write_u8(TYKIND_FUNC).unwrap();
        // num_args:
        write_native_usize(&mut data, 0);
        // ret_ty:
        write_native_usize(&mut data, 0);
        // is_vararg:
        data.write_u8(0).unwrap();

        // TYPE 6
        // type_kind:
        data.write_u8(TYKIND_STRUCT).unwrap();
        // num_fields:
        write_native_usize(&mut data, 2);
        // field_ty_idxs[0]:
        write_native_usize(&mut data, 2);
        // field_ty_idxs[1]:
        write_native_usize(&mut data, 3);
        // field_bit_offs[0]:
        write_native_usize(&mut data, 0);
        // field_bit_offs[0]:
        write_native_usize(&mut data, 24);

        let test_mod = deserialise_module(data.as_slice()).unwrap();
        let string_mod = test_mod.to_str();

        println!("{}", string_mod);
        let expect = "\
# IR format version: 0
# Num funcs: 2
# Num consts: 3
# Num types: 7

func foo($arg0: ptr, $arg1: i32) -> i32 {
  bb0:
    $0_0: ptr = alloca ?cst<a_type>
    nop
    condbr $0_0, bb0, bb1
  bb1:
    ?inst<%3 = some_llvm_instruction ...>
    $1_1: ptr = getelementptr -1i32
    $1_2: ptr = alloca i32, 50i32
    $1_3: ptr = call bar(50i32, 50i32)
    br
    $1_5: {0: ptr, 24: i32} = nop
}

func bar();
";
        assert_eq!(expect, string_mod);
    }

    #[test]
    fn string_deser() {
        let check = |s: &str| {
            assert_eq!(
                &deserialise_string(CString::new(s).unwrap().into_bytes_with_nul()).unwrap(),
                s
            );
        };
        check("testing");
        check("the quick brown fox jumped over the lazy dog");
        check("");
        check("         ");
    }

    #[test]
    fn const_int_strings() {
        use num_traits::cast::NumCast;
        use std::mem;

        // Check (in an endian neutral manner) that a `num-bits`-sized integer of value `num`, when
        // converted to a constant IR integer, then stringified, results in the string `expect`.
        //
        // When `num` has a bit size greater than `num_bits` the most significant bits of `num` are
        // treated as undefined: they can be any value as IR stringification will ignore them.
        fn check<T: NumCast + Sized>(num_bits: u32, num: T, expect: &str) {
            assert!(mem::size_of::<T>() * 8 >= usize::try_from(num_bits).unwrap());

            // Get a byte-vector for `num`.
            let mut bytes: Vec<u8> = Vec::new();
            match mem::size_of::<T>() {
                1 => bytes
                    .write_u8(<u8 as NumCast>::from::<T>(num).unwrap())
                    .unwrap(),
                2 => bytes
                    .write_u16::<NativeEndian>(<u16 as NumCast>::from(num).unwrap())
                    .unwrap(),
                4 => bytes
                    .write_u32::<NativeEndian>(<u32 as NumCast>::from(num).unwrap())
                    .unwrap(),
                8 => bytes
                    .write_u64::<NativeEndian>(<u64 as NumCast>::from(num).unwrap())
                    .unwrap(),
                _ => todo!("{}", mem::size_of::<T>()),
            }

            // Construct an IR constant and check it stringifies ok.
            let it = IntegerType { num_bits };
            let c = Constant {
                type_idx: TypeIdx::new(0),
                bytes,
            };
            assert_eq!(it.const_to_str(&c), expect);
        }

        check(1, 1u8, "1i1");
        check(1, 0u8, "0i1");
        check(1, 254u8, "0i1");
        check(1, 255u8, "1i1");
        check(1, 254u64, "0i1");
        check(1, 255u64, "1i1");

        check(16, 0u16, "0i16");
        check(16, u16::MAX, "-1i16");
        check(16, 12345u16, "12345i16");
        check(16, 12345u64, "12345i16");
        check(16, i16::MIN as u16, &format!("{}i16", i16::MIN));
        check(16, i16::MIN as u64, &format!("{}i16", i16::MIN));

        check(32, 0u32, "0i32");
        check(32, u32::MAX, "-1i32");
        check(32, 12345u32, "12345i32");
        check(32, 12345u64, "12345i32");
        check(32, i32::MIN as u32, &format!("{}i32", i32::MIN));
        check(32, i32::MIN as u64, &format!("{}i32", i32::MIN));

        check(64, 0u64, "0i64");
        check(64, u64::MAX, "-1i64");
        check(64, 12345678u64, "12345678i64");
        check(64, i64::MIN as u64, &format!("{}i64", i64::MIN));
    }
}
