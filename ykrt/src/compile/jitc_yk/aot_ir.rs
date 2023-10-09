//! Yk's AOT IR deserialiser.
//!
//! This is a parser for the on-disk (in the ELF binary) IR format used to express the
//! (immutable) ahead-of-time compiled interpreter.

use byteorder::{NativeEndian, ReadBytesExt};
use deku::prelude::*;
use std::{cell::RefCell, error::Error, ffi::CStr, io::Cursor};

/// A magic number that all bytecode payloads begin with.
const MAGIC: u32 = 0xedd5f00d;
/// The version of the bytecode format.
const FORMAT_VERSION: u32 = 0;

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
    fn to_str(&self, m: &AOTModule) -> String;

    /// Print myself to stderr in human-readable form.
    ///
    /// This is provided as a debugging convenience.
    fn dump(&self, m: &AOTModule) {
        eprintln!("{}", self.to_str(m));
    }
}

/// An instruction opcode.
#[deku_derive(DekuRead)]
#[derive(Debug, Eq, PartialEq)]
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
    Unimplemented = 255,
}

impl IRDisplay for Opcode {
    fn to_str(&self, _m: &AOTModule) -> String {
        format!("{:?}", self).to_lowercase()
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct ConstantOperand {
    constant_idx: usize,
}

impl IRDisplay for ConstantOperand {
    fn to_str(&self, m: &AOTModule) -> String {
        m.consts[self.constant_idx].to_str(m)
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct LocalVariableOperand {
    #[deku(skip)] // computed after deserialisation.
    func_idx: usize,
    bb_idx: usize,
    inst_idx: usize,
}

impl IRDisplay for LocalVariableOperand {
    fn to_str(&self, m: &AOTModule) -> String {
        format!(
            "${}_{}: {}",
            self.bb_idx,
            self.inst_idx,
            m.local_var_operand_type(self).to_str(m)
        )
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct TypeOperand {
    type_idx: usize,
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct BlockOperand {
    bb_idx: usize,
}

impl IRDisplay for BlockOperand {
    fn to_str(&self, _m: &AOTModule) -> String {
        format!("bb{}", self.bb_idx)
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct FunctionOperand {
    func_idx: usize,
}

const OPKIND_CONST: u8 = 0;
const OPKIND_LOCAL_VARIABLE: u8 = 1;
const OPKIND_TYPE: u8 = 2;
const OPKIND_FUNCTION: u8 = 3;
const OPKIND_BLOCK: u8 = 4;
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
    #[deku(id = "OPKIND_UNIMPLEMENTED")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "deserialise_string")] String),
}

impl IRDisplay for Operand {
    fn to_str(&self, m: &AOTModule) -> String {
        match self {
            Self::Constant(c) => c.to_str(m),
            Self::LocalVariable(l) => l.to_str(m),
            Self::Type(t) => m.types[t.type_idx].to_str(m),
            Self::Function(f) => m.funcs[f.func_idx].name.to_owned(),
            Self::Block(bb) => bb.to_str(m),
            Self::Unimplemented(s) => format!("?op<{}>", s),
        }
    }
}

/// A bytecode instruction.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Instruction {
    type_index: usize,
    opcode: Opcode,
    #[deku(temp)]
    num_operands: u32,
    #[deku(count = "num_operands")]
    operands: Vec<Operand>,
    /// A variable name, only computed if the instruction is ever printed.
    #[deku(skip)]
    name: RefCell<Option<String>>,
}

impl IRDisplay for Instruction {
    fn to_str(&self, m: &AOTModule) -> String {
        if self.opcode == Opcode::Unimplemented {
            debug_assert!(self.operands.len() == 1);
            if let Operand::Unimplemented(s) = &self.operands[0] {
                return format!("?inst<{}>", s);
            } else {
                // This would be an invalid serialisation.
                panic!();
            }
        }

        if self.name.borrow().is_none() {
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
    instrs: Vec<Instruction>,
}

impl IRDisplay for Block {
    fn to_str(&self, m: &AOTModule) -> String {
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
    #[deku(temp)]
    num_blocks: usize,
    #[deku(count = "num_blocks")]
    blocks: Vec<Block>,
}

impl Function {
    fn is_declaration(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl IRDisplay for Function {
    fn to_str(&self, m: &AOTModule) -> String {
        let mut ret = String::new();
        if self.is_declaration() {
            // declarations have no body, so print it as such.
            ret.push_str(&format!("func {};\n", self.name));
        } else {
            ret.push_str(&format!("func {} {{\n", self.name));
            for (i, b) in self.blocks.iter().enumerate() {
                ret.push_str(&format!("  bb{}:\n{}", i, b.to_str(m)));
            }
            ret.push_str("}\n");
        }
        ret
    }
}

// A fixed-width two's compliment integer.
//
// Signedness is not specified.
#[deku_derive(DekuRead)]
#[derive(Debug, PartialEq, Eq)]
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
    fn to_str(&self, _m: &AOTModule) -> String {
        format!("i{}", self.num_bits)
    }
}

const TYKIND_VOID: u8 = 0;
const TYKIND_INTEGER: u8 = 1;
const TYKIND_PTR: u8 = 2;
const TYKIND_UNIMPLEMENTED: u8 = 255;

/// A type.
#[deku_derive(DekuRead)]
#[derive(Debug, PartialEq, Eq)]
#[deku(type = "u8")]
pub(crate) enum Type {
    #[deku(id = "TYKIND_VOID")]
    Void,
    #[deku(id = "TYKIND_INTEGER")]
    Integer(IntegerType),
    #[deku(id = "TYKIND_PTR")]
    Ptr,
    #[deku(id = "TYKIND_UNIMPLEMENTED")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "deserialise_string")] String),
}

impl Type {
    fn const_to_str(&self, c: &Constant) -> String {
        match self {
            Self::Void => "void".to_owned(),
            Self::Integer(it) => it.const_to_str(c),
            Self::Ptr => "const_ptr".to_owned(),
            Self::Unimplemented(s) => format!("?cst<{}>", s),
        }
    }
}

impl IRDisplay for Type {
    fn to_str(&self, m: &AOTModule) -> String {
        match self {
            Self::Void => "void".to_owned(),
            Self::Integer(i) => i.to_str(m),
            Self::Ptr => "ptr".to_owned(),
            Self::Unimplemented(s) => format!("?ty<{}>", s),
        }
    }
}

/// A constant.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Constant {
    type_index: usize,
    #[deku(temp)]
    num_bytes: usize,
    #[deku(count = "num_bytes")]
    bytes: Vec<u8>,
}

impl IRDisplay for Constant {
    fn to_str(&self, m: &AOTModule) -> String {
        m.types[self.type_index].const_to_str(self)
    }
}

/// A bytecode module.
///
/// This is the top-level container for the bytecode.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct AOTModule {
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
}

impl AOTModule {
    /// Compute variable names for all instructions that generate a value.
    fn compute_variable_names(&self) {
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
    }

    /// Fill in the function index of local variable operands of instructions.o
    ///
    /// FIXME: It may be possible to do this as we deserialise, instead of after the fact:
    /// https://github.com/sharksforarms/deku/issues/363
    fn compute_local_operand_func_indices(&mut self) {
        for (f_idx, f) in self.funcs.iter_mut().enumerate() {
            for bb in &mut f.blocks {
                for inst in &mut bb.instrs {
                    for op in &mut inst.operands {
                        if let Operand::LocalVariable(ref mut lv) = op {
                            lv.func_idx = f_idx;
                        }
                    }
                }
            }
        }
    }

    /// Get the type of the instruction.
    ///
    /// It is UB to pass an `instr` that is not from the `AOTModule` referenced by `self`.
    fn instr_type(&self, instr: &Instruction) -> &Type {
        &self.types[instr.type_index]
    }

    /// Get the type of the local variable operand.
    ///
    /// It is UB to pass an operand that is not from an instruction in the `AOTModule` referenced
    /// by `self`.
    fn local_var_operand_type(&self, o: &LocalVariableOperand) -> &Type {
        let instr = &self.funcs[o.func_idx].blocks[o.bb_idx].instrs[o.inst_idx];
        self.instr_type(instr)
    }

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

/// Deserialise an AOT module from the slice `data`.
pub(crate) fn deserialise_module(data: &[u8]) -> Result<AOTModule, Box<dyn Error>> {
    match AOTModule::from_bytes((data, 0)) {
        Ok(((_, _), mut modu)) => {
            modu.compute_local_operand_func_indices();
            Ok(modu)
        }
        Err(e) => Err(e.to_string().into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        deserialise_module, deserialise_string, Constant, IntegerType, Opcode, FORMAT_VERSION,
        MAGIC, OPKIND_BLOCK, OPKIND_CONST, OPKIND_FUNCTION, OPKIND_LOCAL_VARIABLE, OPKIND_TYPE,
        OPKIND_UNIMPLEMENTED, TYKIND_INTEGER, TYKIND_PTR, TYKIND_UNIMPLEMENTED, TYKIND_VOID,
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
        // num_blocks:
        write_native_usize(&mut data, 2);

        // BLOCK 0
        // num_instrs:
        write_native_usize(&mut data, 3);

        // INSTRUCTION 0
        // type_index:
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
        // type_index:
        write_native_usize(&mut data, 0);
        // opcode:
        data.write_u8(Opcode::Nop as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(0).unwrap();

        // INSTRUCTION 2
        // type_index:
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
        write_native_usize(&mut data, 5);

        // INSTRUCTION 0
        // type_index:
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
        // type_index:
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
        // type_index:
        write_native_usize(&mut data, 2);
        // opcode:
        data.write_u8(Opcode::Alloca as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(2).unwrap();
        // OPERAND 0
        // operand_kind:
        data.write_u8(OPKIND_TYPE).unwrap();
        // type_index:
        write_native_usize(&mut data, 3);
        // OPERAND 1
        // operand_kind:
        data.write_u8(OPKIND_CONST as u8).unwrap();
        // const_idx:
        write_native_usize(&mut data, 2);

        // INSTRUCTION 3
        // type_index:
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
        // type_index:
        write_native_usize(&mut data, 0);
        // opcode:
        data.write_u8(Opcode::Br as u8).unwrap();
        // num_operands:
        data.write_u32::<NativeEndian>(0).unwrap();

        // FUNCTION 1
        // name:
        write_str(&mut data, "bar");
        // num_blocks:
        write_native_usize(&mut data, 0);

        // CONSTANTS
        // num_consts:
        write_native_usize(&mut data, 3);

        // CONSTANT 0
        // type_index:
        write_native_usize(&mut data, 1);
        // num_bytes:
        write_native_usize(&mut data, 0);

        // CONSTANT 1
        // type_index:
        write_native_usize(&mut data, 3);
        // num_bytes:
        write_native_usize(&mut data, 4);
        // bytes:
        data.write_u32::<NativeEndian>(u32::MAX).unwrap();

        // CONSTANT 2
        // type_index:
        write_native_usize(&mut data, 3);
        // num_bytes:
        write_native_usize(&mut data, 4);
        // bytes:
        data.write_u32::<NativeEndian>(50).unwrap();

        // TYPES
        // num_types:
        write_native_usize(&mut data, 4);

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

        let test_mod = deserialise_module(data.as_slice()).unwrap();
        let string_mod = test_mod.to_str();

        println!("{}", string_mod);
        let expect = "\
# IR format version: 0
# Num funcs: 2
# Num consts: 3
# Num types: 4

func foo {
  bb0:
    $0_0: ptr = alloca ?cst<a_type>
    nop
    condbr $0_0: ptr, bb0, bb1
  bb1:
    ?inst<%3 = some_llvm_instruction ...>
    $1_1: ptr = getelementptr -1i32
    $1_2: ptr = alloca i32, 50i32
    $1_3: ptr = call bar(50i32, 50i32)
    br
}

func bar;
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
                type_index: 0,
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
