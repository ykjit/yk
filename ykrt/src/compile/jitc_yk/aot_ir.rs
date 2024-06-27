//! The AOT Intermediate Representation (IR).
//!
//! This is the IR created by ykllvm: this module contains both the IR itself and a deserialiser
//! from ykllvm's output into the IR itself. The IR can feel a little odd at first because we store
//! indexes into vectors rather than use direct references.
//!
//! The IR uses two general terminological conventions:
//!   * A "definition" is something for which the IR contains complete knowledge.
//!   * A "declaration" is something compiled externally for which we only know the minimal
//!     external structure (e.g. a function signature), with the body being present elsewhere in
//!     the binary.
//!
//! Because using the IR can often involve getting hold of data nested several layers deep, we also
//! use a number of abbreviations/conventions to keep the length of source down to something
//! manageable (in alphabetical order):
//!
//!  * `Const` and `const_`: a "constant"
//!  * `decl`: a "declaration" (e.g. a "function declaration" is a reference to an existing
//!    function somewhere else in the address space)
//!  * `m`: the name conventionally given to the shared [Module] instance (i.e. `m: Module`)
//!  * `Idx`: "index"
//!  * `Inst`: "instruction"
//!  * `Ty`: "type"
//!
//! Textual IR can be generated in the same way as in the JIT IR (i.e. using `std::fmt::Display`
//! and/or `display()`). The same naming conventions are used in the textual AOT IR as in the
//! textual JIT IR. See the docstring for the [super::jit_ir] module.

use byteorder::{NativeEndian, ReadBytesExt};
use deku::prelude::*;
use std::{
    error::Error,
    ffi::CString,
    fmt::{self, Display},
    fs,
    path::PathBuf,
};
use typed_index_collections::TiVec;

/// A magic number that all bytecode payloads begin with.
const MAGIC: u32 = 0xedd5f00d;
/// The version of the bytecode format.
const FORMAT_VERSION: u32 = 0;

/// The symbol name of the control point function (after ykllvm has transformed it).
const CONTROL_POINT_NAME: &str = "__ykrt_control_point";
const LLVM_DEBUG_CALL_NAME: &str = "llvm.dbg.value";

/// The argument index of the trace inputs (live variables) struct at call-sites to the control
/// point call.
const CTRL_POINT_ARGIDX_INPUTS: usize = 2;

/// An AOT IR module.
///
/// This is the top-level container for the AOT IR.
///
/// A module is platform dependent, as type sizes and alignment are baked-in.
#[deku_derive(DekuRead)]
#[derive(Debug, Default)]
pub(crate) struct Module {
    #[deku(assert = "*magic == MAGIC", temp)]
    magic: u32,
    #[deku(assert = "*version == FORMAT_VERSION")]
    version: u32,
    /// The bit-size of what LLVM calls "the pointer indexing type", for address space zero.
    ///
    /// This is the signed integer LLVM uses for computing GEP offsets in the default pointer
    /// address space. This is needed because in certain cases we are required to sign-extend or
    /// truncate to this width.
    ptr_off_bitsize: u8,
    #[deku(temp)]
    num_funcs: usize,
    #[deku(count = "num_funcs", map = "map_to_tivec")]
    funcs: TiVec<FuncIdx, Func>,
    #[deku(temp)]
    num_consts: usize,
    #[deku(count = "num_consts", map = "map_to_tivec")]
    consts: TiVec<ConstIdx, Const>,
    #[deku(temp)]
    num_global_decls: usize,
    #[deku(count = "num_global_decls", map = "map_to_tivec")]
    global_decls: TiVec<GlobalDeclIdx, GlobalDecl>,
    #[deku(temp)]
    num_types: usize,
    #[deku(count = "num_types", map = "map_to_tivec")]
    types: TiVec<TyIdx, Ty>,
}

impl Module {
    /// Find a function by its name.
    ///
    /// # Panics
    ///
    /// Panics if no function exists with that name.
    pub(crate) fn funcidx(&self, find_func: &str) -> FuncIdx {
        // OPT: create a cache in the Module.
        self.funcs
            .iter()
            .enumerate()
            .find(|(_, f)| f.name == find_func)
            .map(|(f_idx, _)| FuncIdx(f_idx))
            .unwrap()
    }

    pub(crate) fn ptr_off_bitsize(&self) -> u8 {
        self.ptr_off_bitsize
    }

    /// Return the block uniquely identified (in this module) by the specified [BBlockId].
    pub(crate) fn bblock(&self, bid: &BBlockId) -> &BBlock {
        self.funcs[bid.funcidx].bblock(bid.bbidx)
    }

    pub(crate) fn const_type(&self, c: &Const) -> &Ty {
        &self.types[c.unwrap_val().tyidx]
    }

    /// Lookup a constant by its index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn const_(&self, ci: ConstIdx) -> &Const {
        &self.consts[ci]
    }

    /// Lookup a type by its index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn type_(&self, idx: TyIdx) -> &Ty {
        &self.types[idx]
    }

    /// Lookup a function by its index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn func(&self, idx: FuncIdx) -> &Func {
        &self.funcs[idx]
    }

    /// Lookup a global variable declaration by its index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn global_decl(&self, idx: GlobalDeclIdx) -> &GlobalDecl {
        &self.global_decls[idx]
    }

    /// Return the number of global variable declarations.
    pub(crate) fn global_decls_len(&self) -> usize {
        self.global_decls.len()
    }

    #[allow(dead_code)]
    pub(crate) fn dump(&self) {
        eprintln!("{}", self);
    }
}

impl std::fmt::Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("# IR format version: {}\n", self.version))?;
        f.write_fmt(format_args!("# Num funcs: {}\n", self.funcs.len()))?;
        f.write_fmt(format_args!("# Num consts: {}\n", self.consts.len()))?;
        f.write_fmt(format_args!(
            "# Num global decls: {}\n",
            self.global_decls.len()
        ))?;
        f.write_fmt(format_args!("# Num types: {}\n", self.types.len()))?;

        if !self.global_decls.is_empty() {
            for gd in &self.global_decls {
                writeln!(f, "{}", gd)?;
            }
        }

        for func in &self.funcs {
            write!(f, "\n{}", func.display(self))?;
        }
        Ok(())
    }
}

/// Deserialise an AOT module from the slice `data`.
pub(crate) fn deserialise_module(data: &[u8]) -> Result<Module, Box<dyn Error>> {
    let ((_, _), modu) = Module::from_bytes((data, 0))?;
    Ok(modu)
}

/// Deserialise and print IR from an on-disk file.
///
/// Used for support tooling (in turn used by tests too).
pub fn print_from_file(path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let data = fs::read(path)?;
    let ir = deserialise_module(&data)?;
    println!("{}", ir);
    Ok(())
}

// Generate common methods for index types.
macro_rules! index {
    ($struct:ident) => {
        impl $struct {
            #[allow(dead_code)]
            pub(crate) fn new(v: usize) -> Self {
                Self(v)
            }
        }

        impl From<usize> for $struct {
            fn from(idx: usize) -> Self {
                Self(idx)
            }
        }

        impl From<$struct> for usize {
            fn from(s: $struct) -> usize {
                s.0
            }
        }
    };
}

/// An index into [Module::funcs].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct FuncIdx(usize);
index!(FuncIdx);

/// An index into [Module::types].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TyIdx(usize);
index!(TyIdx);

/// An index into [Func::bblocks].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BBlockIdx(usize);
index!(BBlockIdx);

/// An index into [BBlock::insts].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct InstIdx(usize);
index!(InstIdx);

/// An index into [Module::consts].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ConstIdx(usize);
index!(ConstIdx);

/// An index into [Module::global_decls].
///
/// Note: these are "declarations" and not "definitions" because they all been AOT code-generated
/// already, and thus come "pre-initialised".
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlobalDeclIdx(usize);
index!(GlobalDeclIdx);

/// An index into [FuncTy::arg_tyidxs].
/// ^ FIXME: no it's not! But it should be!
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ArgIdx(usize);
index!(ArgIdx);

/// Helper function for deku `map` attribute. It is necessary to write all the types out in full to
/// avoid type inference errors, so it's easier to have a single helper function rather than inline
/// this into each `map` attribute.
fn map_to_string(v: Vec<u8>) -> Result<String, DekuError> {
    if let Ok(x) = CString::from_vec_with_nul(v) {
        if let Ok(x) = x.into_string() {
            return Ok(x);
        }
    }
    Err(DekuError::Parse("Couldn't map string".to_owned()))
}

/// Helper function for deku `map` attribute. It is necessary to write all the types out in full to
/// avoid type inference errors, so it's easier to have a single helper function rather than inline
/// this into each `map` attribute.
fn map_to_tivec<I, T>(v: Vec<T>) -> Result<TiVec<I, T>, DekuError> {
    Ok(TiVec::from(v))
}

/// A binary operator.
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[deku(type = "u8")]
pub(crate) enum BinOp {
    Add = 0,
    Sub,
    Mul,
    Or,
    And,
    Xor,
    Shl,
    AShr,
    FAdd,
    FDiv,
    FMul,
    FRem,
    FSub,
    LShr,
    SDiv,
    SRem,
    UDiv,
    URem,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Or => "or",
            Self::And => "and",
            Self::Xor => "xor",
            Self::Shl => "shl",
            Self::AShr => "ashr",
            Self::FAdd => "fadd",
            Self::FDiv => "fdiv",
            Self::FMul => "fmul",
            Self::FRem => "frem",
            Self::FSub => "fsub",
            Self::LShr => "lshr",
            Self::SDiv => "sdiv",
            Self::SRem => "srem",
            Self::UDiv => "udiv",
            Self::URem => "urem",
        };
        write!(f, "{}", s)
    }
}

/// Uniquely identifies an instruction within a [Module].
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) struct InstID {
    /// The index of the parent function.
    funcidx: FuncIdx,
    bbidx: BBlockIdx,
    iidx: InstIdx,
}

impl InstID {
    pub(crate) fn new(funcidx: FuncIdx, bbidx: BBlockIdx, iidx: InstIdx) -> Self {
        Self {
            funcidx,
            bbidx,
            iidx,
        }
    }
}

/// Uniquely identifies a basic block within a [Module].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BBlockId {
    funcidx: FuncIdx,
    bbidx: BBlockIdx,
}

impl BBlockId {
    pub(crate) fn new(funcidx: FuncIdx, bbidx: BBlockIdx) -> Self {
        Self { funcidx, bbidx }
    }

    pub(crate) fn funcidx(&self) -> FuncIdx {
        self.funcidx
    }

    pub(crate) fn bbidx(&self) -> BBlockIdx {
        self.bbidx
    }

    pub(crate) fn is_entry(&self) -> bool {
        self.bbidx == BBlockIdx(0)
    }
}

/// Predicates for use in numeric comparisons.
#[deku_derive(DekuRead)]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[deku(type = "u8")]
pub(crate) enum Predicate {
    Equal = 0,
    NotEqual,
    UnsignedGreater,
    UnsignedGreaterEqual,
    UnsignedLess,
    UnsignedLessEqual,
    SignedGreater,
    SignedGreaterEqual,
    SignedLess,
    SignedLessEqual,
    // FIXME: add floating-point-specific predicates.
}

impl Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Following LLVM's precedent, use short predicate names for formatting.
        match self {
            Self::Equal => write!(f, "eq"),
            Self::NotEqual => write!(f, "ne"),
            Self::UnsignedGreater => write!(f, "ugt"),
            Self::UnsignedGreaterEqual => write!(f, "uge"),
            Self::UnsignedLess => write!(f, "ult"),
            Self::UnsignedLessEqual => write!(f, "ule"),
            Self::SignedGreater => write!(f, "sgt"),
            Self::SignedGreaterEqual => write!(f, "sge"),
            Self::SignedLess => write!(f, "slt"),
            Self::SignedLessEqual => write!(f, "sle"),
        }
    }
}

/// The operations that a [Inst::Cast] can perform.
///
/// FIXME: There are many other operations that we can add here on-demand. See the inheritance
/// hierarchy here: https://llvm.org/doxygen/classllvm_1_1CastInst.html
#[deku_derive(DekuRead)]
#[derive(Debug, Clone, Copy)]
#[deku(type = "u8")]
pub(crate) enum CastKind {
    SExt = 0,
    ZeroExtend = 1,
    Trunc = 2,
    SIToFP = 3,
    FPExt = 4,
}

impl Display for CastKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::SExt => "sext",
            Self::ZeroExtend => "zext",
            Self::Trunc => "trunc",
            Self::SIToFP => "si_to_fp",
            Self::FPExt => "fp_ext",
        };
        write!(f, "{}", s)
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(type = "u8")]
pub(crate) enum Operand {
    #[deku(id = "0")]
    Const(ConstIdx),
    // FIXME: rename this to `Local` for consistency with ykllvm's serialiser.
    #[deku(id = "1")]
    LocalVariable(InstID),
    #[deku(id = "2")]
    Global(GlobalDeclIdx),
    #[deku(id = "3")]
    Func(FuncIdx),
}

impl Operand {
    /// For a [Self::LocalVariable] operand return the instruction that defines the variable.
    ///
    /// Panics for other kinds of operand.
    ///
    /// OPT: This is expensive.
    pub(crate) fn to_inst<'a>(&self, aotmod: &'a Module) -> &'a Inst {
        let Self::LocalVariable(iid) = self else {
            panic!()
        };
        &aotmod.funcs[iid.funcidx].bblocks[iid.bbidx].insts[iid.iidx]
    }

    /// Returns the [Ty] of the operand.
    pub(crate) fn type_<'a>(&self, m: &'a Module) -> &'a Ty {
        match self {
            Self::LocalVariable(_) => {
                // The `unwrap` can't fail for a `LocalVariable`.
                self.to_inst(m).def_type(m).unwrap()
            }
            Self::Const(cidx) => m.type_(m.const_(*cidx).unwrap_val().tyidx()),
            _ => todo!(),
        }
    }

    /// Return the `InstID` of a local variable operand. Panics if called on other kinds of
    /// operands.
    pub(crate) fn to_inst_id(&self) -> InstID {
        let Self::LocalVariable(iid) = self else {
            panic!()
        };
        iid.clone()
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableOperand<'a> {
        DisplayableOperand { operand: self, m }
    }
}

pub(crate) struct DisplayableOperand<'a> {
    operand: &'a Operand,
    m: &'a Module,
}

impl fmt::Display for DisplayableOperand<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.operand {
            Operand::Const(cidx) => {
                write!(f, "{}", self.m.consts[*cidx].display(self.m))
            }
            Operand::LocalVariable(iid) => {
                write!(f, "%{}_{}", usize::from(iid.bbidx), usize::from(iid.iidx))
            }
            Operand::Global(gidx) => write!(f, "@{}", self.m.global_decls[*gidx].name()),
            Operand::Func(fidx) => write!(f, "{}", self.m.funcs[*fidx].name()),
        }
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct DeoptSafepoint {
    pub(crate) id: Operand,
    #[deku(temp)]
    num_lives: u32,
    #[deku(count = "num_lives")]
    pub(crate) lives: Vec<Operand>,
}

impl DeoptSafepoint {
    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableDeoptSafepoint<'a> {
        DisplayableDeoptSafepoint { safepoint: self, m }
    }
}

pub(crate) struct DisplayableDeoptSafepoint<'a> {
    safepoint: &'a DeoptSafepoint,
    m: &'a Module,
}

impl fmt::Display for DisplayableDeoptSafepoint<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lives_s = self
            .safepoint
            .lives
            .iter()
            .map(|a| a.display(self.m).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "[safepoint: {}, ({})]",
            self.safepoint.id.display(self.m),
            lives_s
        )
    }
}

/// An instruction.
///
/// An instruction is conceptually an [Opcode] and a list of [Operand]s. The semantics of the
/// instruction, and the meaning of the operands, are determined by the opcode.
///
/// Insts that compute a value define a new local variable in the parent [Func]. In such a
/// case the newly defined variable can be referenced in the operands of later instructions by the
/// [InstID] of the [Inst] that defined the variable.
///
/// In other words, an instruction and the variable it defines are both identified by the same
/// [InstID].
///
/// The type of the variable defined by an instruction (if any) can be determined by
/// [Inst::def_type()].
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[repr(u8)]
#[deku(type = "u8")]
pub(crate) enum Inst {
    #[deku(id = "0")]
    Nop,
    #[deku(id = "1")]
    Load {
        ptr: Operand,
        tyidx: TyIdx,
        volatile: bool,
    },
    #[deku(id = "2")]
    Store {
        val: Operand,
        tgt: Operand,
        volatile: bool,
    },
    #[deku(id = "3")]
    Alloca {
        tyidx: TyIdx,
        count: usize,
        align: u64,
    },
    #[deku(id = "4")]
    Call {
        callee: FuncIdx,
        #[deku(temp)]
        num_args: u32,
        #[deku(count = "num_args")]
        args: Vec<Operand>,
        #[deku(temp)]
        has_safepoint: u8,
        #[deku(cond = "*has_safepoint != 0", default = "None")]
        safepoint: Option<DeoptSafepoint>,
    },
    #[deku(id = "5")]
    Br {
        /// The block this branch points to.
        succ: BBlockIdx,
    },
    #[deku(id = "6")]
    CondBr {
        cond: Operand,
        true_bb: BBlockIdx,
        false_bb: BBlockIdx,
        safepoint: DeoptSafepoint,
    },
    #[deku(id = "7")]
    ICmp {
        tyidx: TyIdx,
        lhs: Operand,
        pred: Predicate,
        rhs: Operand,
    },
    #[deku(id = "8")]
    Ret {
        #[deku(temp)]
        has_val: u8,
        #[deku(cond = "*has_val != 0", default = "None")]
        val: Option<Operand>,
    },
    #[deku(id = "9")]
    InsertValue { agg: Operand, elem: Operand },
    /// This opcode adds to the `ptr` operand:
    ///  - a constant offset
    ///  - zero or more dynamic offsets.
    ///
    /// where each dynamic offset is:
    ///  - A potentially dynamic element count.
    ///  - A constant element size.
    ///
    /// A dynamic offset is computed at runtime by multiplying the element count by the element
    /// size.
    #[deku(id = "10")]
    PtrAdd {
        // The type index of a pointer.
        //
        // FIXME: the type will always be `ptr`, so this field could be elided if we provide a way
        // for us to find the pointer type index quickly.
        tyidx: TyIdx,
        /// The pointer to offset from.
        ptr: Operand,
        /// The constant offset (in bytes).
        ///
        /// This is signed to allow for negative array indices and negative pointer arithmetic.
        const_off: isize,
        /// The number of dynamic offsets.
        #[deku(temp)]
        num_dyn_offs: usize,
        /// The element counts for the dynamic offsets.
        ///
        /// These are interpreted as signed values to allow negative indexing and negative pointer
        /// arithmetic.
        #[deku(count = "num_dyn_offs")]
        dyn_elem_counts: Vec<Operand>,
        /// The element sizes for the dynamic offsets (in bytes).
        ///
        /// These are unsigned values.
        #[deku(count = "num_dyn_offs")]
        dyn_elem_sizes: Vec<usize>,
    },
    #[deku(id = "11")]
    BinaryOp {
        lhs: Operand,
        binop: BinOp,
        rhs: Operand,
    },
    /// An opcode that is designed to cover cast-like operations. E.g. bitcasts, sign extends, zero
    /// extends etc.
    #[deku(id = "12")]
    Cast {
        /// The cast-like operation to perform.
        cast_kind: CastKind,
        /// The value to be operated upon.
        val: Operand,
        /// The resulting type of the operation.
        dest_tyidx: TyIdx,
    },
    #[deku(id = "13")]
    Switch {
        test_val: Operand,
        default_dest: BBlockIdx,
        #[deku(temp)]
        num_cases: usize,
        /// The values for each switch block. FIXME: These are currently cast by ykllvm to a `u64`
        /// no matter what the original type was: in other words, these should be interpreted as
        /// bit patterns consuming 64-bits, not integer types of `u64`. Currently ykllvm prevents
        /// types bigger than 64 bits being serialised, but the original integer type may require
        /// fewer than 64-bits.
        #[deku(count = "num_cases")]
        case_values: Vec<u64>,
        #[deku(count = "num_cases")]
        case_dests: Vec<BBlockIdx>,
        safepoint: DeoptSafepoint,
    },
    #[deku(id = "14")]
    Phi {
        #[deku(temp)]
        num_incoming: usize,
        #[deku(count = "num_incoming")]
        incoming_bbs: Vec<BBlockIdx>,
        #[deku(count = "num_incoming")]
        incoming_vals: Vec<Operand>,
    },
    #[deku(id = "15")]
    IndirectCall {
        ftyidx: TyIdx,
        callop: Operand,
        #[deku(temp)]
        num_args: u32,
        #[deku(count = "num_args")]
        args: Vec<Operand>,
    },
    #[deku(id = "16")]
    Select {
        cond: Operand,
        trueval: Operand,
        falseval: Operand,
    },
    #[deku(id = "17")]
    LoadArg { arg_idx: usize, ty_idx: TyIdx },
    #[deku(id = "255")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "map_to_string")] String),
}

impl Inst {
    /// Find the name of a local variable.
    ///
    /// This is used when stringifying the instruction.
    ///
    /// FIXME: This is very slow and could be optimised.
    fn local_name(&self, m: &Module) -> String {
        for f in m.funcs.iter() {
            for (bbidx, bb) in f.bblocks.iter().enumerate() {
                for (iidx, inst) in bb.insts.iter().enumerate() {
                    if std::ptr::addr_eq(inst, self) {
                        return format!("%{}_{}", bbidx, iidx);
                    }
                }
            }
        }
        panic!(); // malformed IR.
    }

    /// Returns the [Ty] of the local variable defined by this instruction or `None` if this
    /// instruction does not define a new local variable.
    pub(crate) fn def_type<'a>(&self, m: &'a Module) -> Option<&'a Ty> {
        match self {
            Self::Alloca { .. } => Some(&Ty::Ptr),
            Self::BinaryOp { lhs, .. } => Some(lhs.type_(m)),
            Self::Br { .. } => None,
            Self::Call { callee, .. } => {
                // The type of the newly-defined local is the return type of the callee.
                if let Ty::Func(ft) = m.type_(m.func(*callee).tyidx) {
                    let ty = m.type_(ft.ret_ty);
                    if ty != &Ty::Void {
                        Some(ty)
                    } else {
                        None
                    }
                } else {
                    panic!(); // IR malformed.
                }
            }
            Self::CondBr { .. } => None,
            Self::InsertValue { agg, .. } => Some(agg.type_(m)),
            Self::ICmp { tyidx, .. } => Some(m.type_(*tyidx)),
            Self::Load { tyidx, .. } => Some(m.type_(*tyidx)),
            Self::PtrAdd { tyidx, .. } => Some(m.type_(*tyidx)),
            Self::Ret { .. } => {
                // Subtle: although `Ret` might make a value, that's not a local value in the
                // parent function.
                None
            }
            Self::Store { .. } => None,
            Self::Cast { dest_tyidx, .. } => Some(m.type_(*dest_tyidx)),
            Self::Switch { .. } => None,
            Self::Phi { incoming_vals, .. } => {
                // Indexing cannot crash: correct PHI nodes have at least one incoming value.
                Some(incoming_vals[0].type_(m))
            }
            Self::IndirectCall { ftyidx, .. } => {
                // The type of the newly-defined local is the return type of the callee.
                if let Ty::Func(ft) = m.type_(*ftyidx) {
                    let ty = m.type_(ft.ret_ty);
                    if ty != &Ty::Void {
                        Some(ty)
                    } else {
                        None
                    }
                } else {
                    panic!(); // IR malformed.
                }
            }
            Self::Select {
                cond: _, trueval, ..
            } => Some(trueval.type_(m)),
            Self::LoadArg { arg_idx: _, ty_idx } => Some(m.type_(*ty_idx)),
            Self::Unimplemented(_) => None,
            Self::Nop => None,
        }
    }

    pub(crate) fn is_mappable_call(&self, m: &Module) -> bool {
        match self {
            Self::Call { callee, .. } => !m.func(*callee).is_declaration(),
            _ => false,
        }
    }

    /// If `self` is a call to the control point, then return the live variables struct argument
    /// being passed to it. Otherwise return None.
    pub(crate) fn control_point_call_trace_inputs(&self, m: &Module) -> Option<&Operand> {
        match self {
            Self::Call { callee, args, .. } => {
                if m.func(*callee).name == CONTROL_POINT_NAME {
                    let arg = &args[CTRL_POINT_ARGIDX_INPUTS];
                    // It should be a pointer (to a struct, but we can't check that).
                    debug_assert!(matches!(arg.type_(m), &Ty::Ptr));
                    Some(arg)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn safepoint(&self) -> Option<&DeoptSafepoint> {
        match self {
            Self::Call { safepoint, .. } => safepoint.as_ref(),
            Self::CondBr { ref safepoint, .. } => Some(safepoint),
            _ => None,
        }
    }

    pub(crate) fn is_debug_call(&self, m: &Module) -> bool {
        match self {
            Self::Call { callee, .. } => m.func(*callee).name == LLVM_DEBUG_CALL_NAME,
            _ => false,
        }
    }

    /// Determine if two instructions in the (immutable) AOT IR are the same based on pointer
    /// identity.
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableInst<'a> {
        DisplayableInst {
            instruction: self,
            m,
        }
    }
}

pub(crate) struct DisplayableInst<'a> {
    instruction: &'a Inst,
    m: &'a Module,
}

impl fmt::Display for DisplayableInst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(t) = self.instruction.def_type(self.m) {
            // If the instruction defines a local, we will format the instruction like it's an
            // assignment. Here we print the left-hand side.
            write!(
                f,
                "{}: {} = ",
                self.instruction.local_name(self.m),
                t.display(self.m)
            )?;
        }

        match self.instruction {
            Inst::Alloca {
                tyidx,
                count,
                align,
            } => write!(
                f,
                "alloca {}, {}, {}",
                self.m.type_(*tyidx).display(self.m),
                count,
                align
            ),
            Inst::BinaryOp { lhs, binop, rhs } => {
                write!(
                    f,
                    "{binop} {}, {}",
                    lhs.display(self.m),
                    rhs.display(self.m)
                )
            }
            Inst::Br { succ } => write!(f, "br bb{}", usize::from(*succ)),
            Inst::Call {
                callee,
                args,
                safepoint,
            } => {
                let args_s = args
                    .iter()
                    .map(|a| a.display(self.m).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let safepoint_s = safepoint
                    .as_ref()
                    .map_or("".to_string(), |sp| format!(" {}", sp.display(self.m)));
                write!(
                    f,
                    "call {}({}){}",
                    self.m.func(*callee).name(),
                    args_s,
                    safepoint_s
                )
            }
            Inst::CondBr {
                cond,
                true_bb,
                false_bb,
                safepoint,
            } => write!(
                f,
                "condbr {}, bb{}, bb{} {}",
                cond.display(self.m),
                usize::from(*true_bb),
                usize::from(*false_bb),
                safepoint.display(self.m)
            ),
            Inst::ICmp { lhs, pred, rhs, .. } => {
                write!(f, "{pred} {}, {}", lhs.display(self.m), rhs.display(self.m))
            }
            Inst::Load {
                ptr,
                tyidx: _,
                volatile,
            } => {
                let vol = if *volatile { ", volatile" } else { "" };
                write!(f, "load {}{}", ptr.display(self.m), vol)
            }
            Inst::PtrAdd {
                ptr,
                const_off,
                dyn_elem_counts,
                dyn_elem_sizes,
                ..
            } => {
                if dyn_elem_counts.is_empty() {
                    write!(f, "ptr_add {}, {}", ptr.display(self.m), const_off)
                } else {
                    let dyns = dyn_elem_counts
                        .iter()
                        .zip(dyn_elem_sizes)
                        .map(|(c, s)| format!("({} * {})", c.display(self.m), s))
                        .collect::<Vec<_>>();
                    write!(
                        f,
                        "ptr_add {}, {} + {}",
                        ptr.display(self.m),
                        const_off,
                        dyns.join(" + ")
                    )
                }
            }
            Inst::Ret { val } => match val {
                None => write!(f, "ret"),
                Some(v) => write!(f, "ret {}", v.display(self.m)),
            },
            Inst::Store { tgt, val, volatile } => {
                let vol = if *volatile { ", volatile" } else { "" };
                write!(
                    f,
                    "*{} = {}{}",
                    tgt.display(self.m),
                    val.display(self.m),
                    vol
                )
            }
            Inst::InsertValue { agg, elem } => write!(
                f,
                "insert_val {}, {}",
                agg.display(self.m),
                elem.display(self.m)
            ),
            Inst::Cast {
                cast_kind,
                val,
                dest_tyidx,
            } => write!(
                f,
                "{cast_kind} {}, {}",
                val.display(self.m),
                self.m.types[*dest_tyidx].display(self.m)
            ),
            Inst::Switch {
                test_val,
                default_dest,
                case_values,
                case_dests,
                safepoint,
            } => {
                let cases = case_values
                    .iter()
                    .zip(case_dests)
                    .map(|(val, dest)| format!("{} -> bb{}", val, usize::from(*dest)))
                    .collect::<Vec<_>>();
                write!(
                    f,
                    "switch {}, bb{}, [{}] {}",
                    test_val.display(self.m),
                    usize::from(*default_dest),
                    cases.join(", "),
                    safepoint.display(self.m)
                )
            }
            Inst::Phi {
                incoming_vals,
                incoming_bbs,
            } => {
                let args = incoming_bbs
                    .iter()
                    .zip(incoming_vals)
                    .map(|(bb, val)| format!("bb{} -> {}", usize::from(*bb), val.display(self.m)))
                    .collect::<Vec<_>>();
                write!(f, "phi {}", args.join(", "))
            }
            Inst::IndirectCall {
                ftyidx: _,
                callop,
                args,
            } => {
                let args_s = args
                    .iter()
                    .map(|a| a.display(self.m).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "icall {}({})", callop.display(self.m), args_s)
            }
            Inst::Select {
                cond,
                trueval,
                falseval,
            } => {
                write!(
                    f,
                    "select {}, {}, {}",
                    cond.display(self.m),
                    trueval.display(self.m),
                    falseval.display(self.m)
                )
            }
            Inst::LoadArg { arg_idx, ty_idx: _ } => write!(f, "arg({})", arg_idx,),
            Inst::Unimplemented(s) => write!(f, "unimplemented <<{}>>", s),
            Inst::Nop => write!(f, "nop"),
        }
    }
}

/// A basic block containing IR [Inst]s.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct BBlock {
    #[deku(temp)]
    num_insts: usize,
    #[deku(count = "num_insts", map = "map_to_tivec")]
    pub(crate) insts: TiVec<InstIdx, Inst>,
}

impl BBlock {
    // Returns true if this block is terminated by a return, false otherwise.
    pub fn is_return(&self) -> bool {
        matches!(self.insts.last().unwrap(), Inst::Ret { .. })
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableBBlock<'a> {
        DisplayableBBlock { bblock: self, m }
    }
}

pub(crate) struct DisplayableBBlock<'a> {
    bblock: &'a BBlock,
    m: &'a Module,
}

impl fmt::Display for DisplayableBBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x in &self.bblock.insts {
            writeln!(f, "    {}", x.display(self.m))?;
        }
        Ok(())
    }
}

/// A function definition or declaration.
///
/// If the function was compiled by ykllvm as part of the interpreter binary, then we have IR for
/// the function body, and the function is said to be a *function definition*.
///
/// Conversely, if the function was *not* compiled by ykllvm as part of the interpreter binary (as
/// is the case for shared library functions), then we have no IR for the function body, and the
/// function is said to be a *function declaration*.
///
/// [Func::is_declaration()] can be used to determine if the [Func] is a definition or a
/// declaration.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct Func {
    #[deku(until = "|v: &u8| *v == 0", map = "map_to_string")]
    name: String,
    tyidx: TyIdx,
    outline: bool,
    #[deku(temp)]
    num_bblocks: usize,
    #[deku(count = "num_bblocks", map = "map_to_tivec")]
    bblocks: TiVec<BBlockIdx, BBlock>,
}

impl Func {
    fn is_declaration(&self) -> bool {
        self.bblocks.is_empty()
    }

    pub(crate) fn is_outline(&self) -> bool {
        self.outline
    }

    /// Return the [BBlock] at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of range.
    pub(crate) fn bblock(&self, bbidx: BBlockIdx) -> &BBlock {
        &self.bblocks[bbidx]
    }

    /// Return the name of the function.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Return the type index of the function.
    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableFunc<'a> {
        DisplayableFunc { func_: self, m }
    }
}

pub(crate) struct DisplayableFunc<'a> {
    func_: &'a Func,
    m: &'a Module,
}

impl fmt::Display for DisplayableFunc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = &self.m.types[self.func_.tyidx];
        if let Ty::Func(fty) = ty {
            write!(
                f,
                "func {}({}",
                self.func_.name,
                fty.arg_tyidxs
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("%arg{}: {}", i, self.m.types[*t].display(self.m)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
            if fty.is_vararg {
                write!(f, ", ...")?;
            }
            write!(f, ")")?;
            let ret_ty = &self.m.types[fty.ret_ty];
            if ret_ty != &Ty::Void {
                write!(f, " -> {}", ret_ty.display(self.m))?;
            }
            if self.func_.is_declaration() {
                // declarations have no body, so print it as such.
                writeln!(f, ";")
            } else {
                writeln!(f, " {{")?;
                for (i, b) in self.func_.bblocks.iter().enumerate() {
                    write!(f, "  bb{}:\n{}", i, b.display(self.m))?;
                }
                writeln!(f, "}}")
            }
        } else {
            unreachable!();
        }
    }
}

/// Return the stringified constant integer obtained by interpreting `bytes` as `num-bits`-wide
/// constant integer.
///
/// FIXME: For now we just handle common integer types, but eventually we will need to
/// implement printing of aribitrarily-sized (in bits) integers. Consider using a bigint
/// library so we don't have to do it ourself?
///
/// This discussion may help:
/// https://rust-lang.zulipchat.com/#narrow/stream/122651-general/topic/.E2.9C.94.20Big.20Integer.20library.20with.20bit.20granularity/near/393733327
pub(crate) fn const_int_bytes_to_string(num_bits: u32, bytes: &[u8]) -> String {
    // All of the unwraps below are safe due to:
    debug_assert!(bytes.len() * 8 >= usize::try_from(num_bits).unwrap());

    let mut bytes = bytes;
    match num_bits {
        1 => format!("{}i1", bytes.read_i8().unwrap() & 1),
        8 => format!("{}i8", bytes.read_i8().unwrap()),
        16 => format!("{}i16", bytes.read_i16::<NativeEndian>().unwrap()),
        32 => format!("{}i32", bytes.read_i32::<NativeEndian>().unwrap()),
        64 => format!("{}i64", bytes.read_i64::<NativeEndian>().unwrap()),
        _ => todo!("{}", num_bits),
    }
}

/// A fixed-width integer type.
///
/// Note:
///   1. These integers range in size from 1..2^23 (inc.) bits. This is inherited [from LLVM's
///      integer type](https://llvm.org/docs/LangRef.html#integer-type).
///   2. Signedness is not specified. Interpretation of the bit pattern is delegated to operations
///      upon the integer.
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IntegerTy {
    pub(crate) num_bits: u32,
}

impl IntegerTy {
    /// Create a new integer type with the specified number of bits.
    #[cfg(test)]
    pub(crate) fn new(num_bits: u32) -> Self {
        debug_assert!(num_bits > 0 && num_bits <= 0x800000);
        Self { num_bits }
    }

    /// Return the number of bits (1..2^23 (inc.)) this integer spans.
    pub(crate) fn num_bits(&self) -> u32 {
        debug_assert!(self.num_bits > 0 && self.num_bits <= 0x800000);
        self.num_bits
    }

    /// Return the number of bytes required to store this integer type.
    ///
    /// Padding for alignment is not included.
    #[cfg(test)]
    pub(crate) fn byte_size(&self) -> usize {
        let bits = self.num_bits();
        let mut ret = bits / 8;
        // If it wasn't an exactly byte-sized thing, round up to the next byte.
        if bits % 8 != 0 {
            ret += 1;
        }
        usize::try_from(ret).unwrap()
    }

    /// Format a constant integer value that is of the type described by `self`.
    fn const_to_string(&self, c: &ConstVal) -> String {
        const_int_bytes_to_string(self.num_bits, c.bytes())
    }
}

impl Display for IntegerTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i{}", self.num_bits)
    }
}

#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FuncTy {
    /// The number of formal arguments the function takes.
    #[deku(temp)]
    num_args: usize,
    /// Ty indices for the function's formal arguments.
    #[deku(count = "num_args")]
    arg_tyidxs: Vec<TyIdx>,
    /// Ty index of the function's return type.
    ret_ty: TyIdx,
    /// Is the function vararg?
    is_vararg: bool,
}

impl FuncTy {
    #[cfg(test)]
    fn new(arg_tyidxs: Vec<TyIdx>, ret_tyidx: TyIdx, is_vararg: bool) -> Self {
        Self {
            arg_tyidxs,
            ret_ty: ret_tyidx,
            is_vararg,
        }
    }

    pub(crate) fn arg_tyidxs(&self) -> &[TyIdx] {
        &self.arg_tyidxs
    }

    pub(crate) fn ret_ty(&self) -> TyIdx {
        self.ret_ty
    }

    pub(crate) fn is_vararg(&self) -> bool {
        self.is_vararg
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableFuncTy<'a> {
        DisplayableFuncTy { func_type: self, m }
    }
}

pub(crate) struct DisplayableFuncTy<'a> {
    func_type: &'a FuncTy,
    m: &'a Module,
}

impl fmt::Display for DisplayableFuncTy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut args = self
            .func_type
            .arg_tyidxs
            .iter()
            .map(|t| self.m.types[*t].display(self.m).to_string())
            .collect::<Vec<_>>();
        if self.func_type.is_vararg() {
            args.push("...".to_owned());
        }
        write!(f, "func({})", args.join(", "))?;
        let rty = self.m.type_(self.func_type.ret_ty);
        if rty != &Ty::Void {
            write!(f, " -> {}", rty.display(self.m))?
        }
        Ok(())
    }
}

/// Floating point types.
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[deku(type = "u8")]
pub(crate) enum FloatTy {
    // 32-bit floating point.
    #[deku(id = "0")]
    Float,
    // 64-bit floating point.
    #[deku(id = "1")]
    Double,
}

impl Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Float => "float",
            Self::Double => "double",
        };
        write!(f, "{}", s)
    }
}

#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StructTy {
    /// The number of fields the struct has.
    #[deku(temp)]
    num_fields: usize,
    /// The types of the fields.
    #[deku(count = "num_fields")]
    field_tyidxs: Vec<TyIdx>,
    /// The bit offsets of the fields (taking into account any required padding for alignment).
    #[deku(count = "num_fields")]
    field_bit_offs: Vec<usize>,
}

impl StructTy {
    /// Returns the type index of the specified field index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn field_tyidx(&self, idx: usize) -> TyIdx {
        self.field_tyidxs[idx]
    }

    /// Returns the byte offset of the specified field index.
    ///
    /// # Panics
    ///
    /// Panics if the field is not byte-aligned or the index is out of bounds.
    pub(crate) fn field_byte_off(&self, idx: usize) -> usize {
        let bit_off = self.field_bit_offs[idx];
        if bit_off % 8 != 0 {
            todo!();
        }
        bit_off / 8
    }

    /// Returns the number of fields in the struct.
    pub(crate) fn num_fields(&self) -> usize {
        self.field_tyidxs.len()
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableStructTy<'a> {
        DisplayableStructTy {
            struct_type: self,
            m,
        }
    }
}

pub(crate) struct DisplayableStructTy<'a> {
    struct_type: &'a StructTy,
    m: &'a Module,
}

impl Display for DisplayableStructTy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.struct_type
                .field_tyidxs
                .iter()
                .enumerate()
                .map(|(i, ti)| format!(
                    "{}: {}",
                    self.struct_type.field_bit_offs[i],
                    self.m.types[*ti].display(self.m)
                ))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

const TYKIND_VOID: u8 = 0;
const TYKIND_INTEGER: u8 = 1;
const TYKIND_PTR: u8 = 2;
const TYKIND_FUNC: u8 = 3;
const TYKIND_STRUCT: u8 = 4;
const TYKIND_FLOAT: u8 = 5;
const TYKIND_UNIMPLEMENTED: u8 = 255;

/// A type.
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
#[deku(type = "u8")]
pub(crate) enum Ty {
    #[deku(id = "TYKIND_VOID")]
    Void,
    #[deku(id = "TYKIND_INTEGER")]
    Integer(IntegerTy),
    #[deku(id = "TYKIND_PTR")]
    Ptr,
    #[deku(id = "TYKIND_FUNC")]
    Func(FuncTy),
    #[deku(id = "TYKIND_STRUCT")]
    Struct(StructTy),
    #[deku(id = "TYKIND_FLOAT")]
    Float(FloatTy),
    #[deku(id = "TYKIND_UNIMPLEMENTED")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "map_to_string")] String),
}

impl Ty {
    fn const_to_string(&self, c: &ConstVal) -> String {
        match self {
            Self::Void => "void".to_owned(),
            Self::Integer(it) => it.const_to_string(c),
            Self::Ptr => {
                let ptr_size = std::mem::size_of::<usize>();
                debug_assert_eq!(c.bytes().len(), ptr_size);
                // unwrap is safe: constant is malformed if there are too few bytes for a chunk.
                let pval = usize::from_ne_bytes(*c.bytes().first_chunk().unwrap());
                format!("{:#x}", pval)
            }
            Self::Func(_) => unreachable!(), // No such thing as a constant function in our IR.
            Self::Struct(_) => {
                // FIXME: write a stringifier for constant structs.
                "const_struct".to_owned()
            }
            Self::Float(ft) => {
                // Note that floats are stored at rest as a doubles for now.
                // unwrap safe: constant malformed if there are too few bytes for a chunk.
                let dval = f64::from_ne_bytes(*c.bytes().first_chunk().unwrap());
                match ft {
                    FloatTy::Float => format!("{}float", dval as f32),
                    FloatTy::Double => format!("{}double", dval),
                }
            }
            Self::Unimplemented(s) => format!("?cst<{}>", s),
        }
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableTy<'a> {
        DisplayableTy { type_: self, m }
    }
}

#[derive(Debug)]
pub(crate) struct DisplayableTy<'a> {
    type_: &'a Ty,
    m: &'a Module,
}

impl fmt::Display for DisplayableTy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.type_ {
            Ty::Void => write!(f, "void"),
            Ty::Integer(x) => write!(f, "{}", x),
            Ty::Ptr => write!(f, "ptr"),
            Ty::Func(ft) => write!(f, "{}", ft.display(self.m)),
            Ty::Struct(st) => write!(f, "{}", st.display(self.m)),
            Ty::Float(ft) => write!(f, "{}", ft),
            Ty::Unimplemented(s) => write!(f, "?ty<{}>", s),
        }
    }
}

/// A (potentially implemented) constant.
///
/// Constants not handled by the ykllvm serialiser become `Const::Unimplemented`.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(type = "u8")]
pub(crate) enum Const {
    #[deku(id = "0")]
    Val(ConstVal),
    #[deku(id = "1")]
    Unimplemented(#[deku(until = "|v: &u8| *v == 0", map = "map_to_string")] String),
}

impl Const {
    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableConst<'a> {
        DisplayableConst { constant: self, m }
    }

    /// Returns a `ConstVal` iff `self` is `Cosnt::Val`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is `Cosnt::Unimplemented`. The panic message indicates the problematic
    /// constant that requires implementation in the ykllvm serialiser.
    pub(crate) fn unwrap_val(&self) -> &ConstVal {
        match self {
            Const::Val(v) => v,
            Const::Unimplemented(m) => panic!("unimplemented const: {}", m),
        }
    }
}

pub(crate) struct DisplayableConst<'a> {
    constant: &'a Const,
    m: &'a Module,
}

impl Display for DisplayableConst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.constant {
            Const::Val(cv) => write!(f, "{}", cv.display(self.m)),
            Const::Unimplemented(m) => write!(f, "unimplemented <<{}>>", m),
        }
    }
}

/// A constant value.
#[deku_derive(DekuRead)]
#[derive(Debug)]
pub(crate) struct ConstVal {
    tyidx: TyIdx,
    #[deku(temp)]
    num_bytes: usize,
    #[deku(count = "num_bytes")]
    bytes: Vec<u8>,
}

impl ConstVal {
    /// Return a byte slice of the constant's value.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the type index of the constant.
    pub(crate) fn tyidx(&self) -> TyIdx {
        self.tyidx
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableConstVal<'a> {
        DisplayableConstVal { cv: self, m }
    }
}

pub(crate) struct DisplayableConstVal<'a> {
    m: &'a Module,
    cv: &'a ConstVal,
}

impl Display for DisplayableConstVal<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.m.types[self.cv.tyidx].const_to_string(self.cv)
        )
    }
}

/// A global variable declaration, identified by its symbol name.
///
/// Since the AOT IR doesn't capture the initialisers of global variables (externally compiled or
/// otherwise), all global variables are considered *declarations*.
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlobalDecl {
    is_threadlocal: bool,
    #[deku(until = "|v: &u8| *v == 0", map = "map_to_string")]
    name: String,
}

impl Display for GlobalDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tl = if self.is_threadlocal { " tls" } else { "" };
        write!(f, "global_decl{} @{}", tl, self.name,)
    }
}

impl GlobalDecl {
    pub(crate) fn is_threadlocal(&self) -> bool {
        self.is_threadlocal
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{PrimInt, ToBytes};
    use std::mem;

    #[test]
    fn string_deser() {
        let check = |s: &str| {
            assert_eq!(
                &map_to_string(CString::new(s).unwrap().into_bytes_with_nul()).unwrap(),
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
        // Check (in an endian neutral manner) that a `num-bits`-sized integer of value `num`, when
        // converted to a constant IR integer, then stringified, results in the string `expect`.
        //
        // When `num` has a bit size greater than `num_bits` the most significant bits of `num` are
        // treated as undefined: they can be any value as IR stringification will ignore them.
        fn check<T: ToBytes + PrimInt>(num_bits: u32, num: T, expect: &str) {
            assert!(mem::size_of::<T>() * 8 >= usize::try_from(num_bits).unwrap());

            // Get a byte-vector for `num`.
            let bytes = ToBytes::to_ne_bytes(&num).as_ref().to_vec();

            // Construct an IR constant and check it stringifies ok.
            let it = IntegerTy { num_bits };
            let c = ConstVal {
                tyidx: TyIdx::new(0),
                bytes,
            };
            assert_eq!(it.const_to_string(&c), expect);
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

    #[test]
    fn stringify_const_ptr() {
        let mut m = Module::default();
        m.types.push(Ty::Ptr);
        let ptr_tyidx = TyIdx(0);
        // Build a constant pointer with higher valued bytes towards the most-significant byte.
        // Careful now: big endian stores the most significant byte first!
        let rng = 0u8..(mem::size_of::<usize>() as u8);
        #[cfg(target_endian = "little")]
        let bytes = rng.clone().collect::<Vec<u8>>();
        #[cfg(target_endian = "big")]
        let bytes = rng.clone().rev().collect::<Vec<u8>>();

        let cp = ConstVal {
            tyidx: ptr_tyidx,
            bytes,
        };

        let expect_bytes = rng.rev().map(|i| format!("{:02x}", i)).collect::<String>();
        let expect_usize = usize::from_str_radix(&expect_bytes, 16).unwrap();
        assert_eq!(
            format!("{}", cp.display(&m)),
            format!("{:#x}", expect_usize)
        );
    }

    #[test]
    fn stringify_const_ptr2() {
        let mut m = Module::default();
        m.types.push(Ty::Ptr);
        let ptr_tyidx = TyIdx(0);
        let ptr_val = stringify_const_ptr2 as *const u8 as usize;
        let cp = ConstVal {
            tyidx: ptr_tyidx,
            bytes: ptr_val.to_ne_bytes().to_vec(),
        };
        assert_eq!(format!("{}", cp.display(&m)), format!("{:#x}", ptr_val));
    }

    #[test]
    fn stringify_unimplemented_consts() {
        let c = Const::Unimplemented("someoperand".into());
        let m = Module::default();
        assert_eq!(c.display(&m).to_string(), "unimplemented <<someoperand>>");
    }

    #[test]
    fn integer_type_sizes() {
        for i in 1..8 {
            assert_eq!(IntegerTy::new(i).byte_size(), 1);
        }
        for i in 9..16 {
            assert_eq!(IntegerTy::new(i).byte_size(), 2);
        }
        assert_eq!(IntegerTy::new(127).byte_size(), 16);
        assert_eq!(IntegerTy::new(128).byte_size(), 16);
        assert_eq!(IntegerTy::new(129).byte_size(), 17);
    }

    #[test]
    fn stringify_func_types() {
        let mut m = Module::default();

        let i8_tyidx = TyIdx::new(m.types.len());
        m.types.push(Ty::Integer(IntegerTy { num_bits: 8 }));
        let void_tyidx = TyIdx::new(m.types.len());
        m.types.push(Ty::Void);

        let fty = Ty::Func(FuncTy::new(vec![i8_tyidx], i8_tyidx, false));
        assert_eq!(fty.display(&m).to_string(), "func(i8) -> i8");

        let fty = Ty::Func(FuncTy::new(vec![i8_tyidx], i8_tyidx, true));
        assert_eq!(fty.display(&m).to_string(), "func(i8, ...) -> i8");

        let fty = Ty::Func(FuncTy::new(vec![], i8_tyidx, false));
        assert_eq!(fty.display(&m).to_string(), "func() -> i8");

        let fty = Ty::Func(FuncTy::new(vec![], void_tyidx, false));
        assert_eq!(fty.display(&m).to_string(), "func()");
    }
}
