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
#![allow(dead_code)]

use byteorder::{NativeEndian, ReadBytesExt};
use deku::prelude::*;
use std::{
    borrow::Cow,
    collections::HashMap,
    error::Error,
    ffi::{CStr, CString, OsStr},
    fmt::{self, Display},
    fs,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use typed_index_collections::{TiSlice, TiVec};

/// A magic number that all bytecode payloads begin with.
const MAGIC: u32 = 0xedd5f00d;
/// The version of the bytecode format.
const FORMAT_VERSION: u32 = 0;

const LLVM_DEBUG_CALL_NAME: &str = "llvm.dbg.value";

/// A struct identifying a line in a source file.
#[derive(Debug)]
#[deku_derive(DekuRead)]
struct LineInfoLoc {
    /// The path to the containing source file.
    ///
    /// The file existed at AOT compile time and there is no guarantee that it still exists now.
    pathidx: PathIdx,
    /// The line number in the source file.
    line_num: usize,
}

/// A "raw" line info record as initially seen by the deserialiser.
#[derive(Debug)]
#[deku_derive(DekuRead)]
struct RawLineInfoRec {
    /// The instruction record is for.
    inst_id: InstId,
    /// The location in the source file.
    line: LineInfoLoc,
}

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
    #[deku(temp)]
    num_paths: usize,
    #[deku(count = "num_paths", map = "map_to_tivec_of_pathbuf")]
    paths: TiVec<PathIdx, PathBuf>,
    #[deku(temp)]
    num_lineinfos: usize,
    #[deku(count = "num_lineinfos", map = "map_to_lineinfo")]
    line_infos: HashMap<InstId, LineInfoLoc>,
    #[deku(skip)]
    source_files: Arc<Mutex<HashMap<PathBuf, Option<BufReader<fs::File>>>>>,
    #[deku(skip)]
    /// A cache mapping function names to function indices.
    func_cache: HashMap<CString, FuncIdx>,
}

impl Module {
    pub(crate) fn create_func_cache(&mut self) {
        for (fidx, f) in self.funcs.iter().enumerate() {
            self.func_cache
                .insert(CString::new(f.name()).unwrap(), FuncIdx(fidx));
        }
    }

    /// Find a function by its name.
    ///
    /// # Panics
    ///
    /// Panics if no function exists with that name.
    pub(crate) fn funcidx(&self, find_func: &CStr) -> FuncIdx {
        self.func_cache[find_func]
    }

    pub(crate) fn ptr_off_bitsize(&self) -> u8 {
        self.ptr_off_bitsize
    }

    /// Return the AOT instruction for the given instruction id.
    pub(crate) fn inst(&self, instid: &InstId) -> &Inst {
        let f = self.func(instid.funcidx);
        let b = f.bblock(instid.bbidx);
        &b.insts[instid.iidx]
    }

    /// Return the block uniquely identified (in this module) by the specified [BBlockId].
    pub(crate) fn bblock(&self, bid: &BBlockId) -> &BBlock {
        self.funcs[bid.funcidx].bblock(bid.bbidx)
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

    /// Lookup a path by its index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    fn path(&self, pathidx: PathIdx) -> &Path {
        &self.paths[pathidx]
    }

    #[allow(dead_code)]
    pub(crate) fn dump(&self) {
        eprintln!("{self}");
    }

    /// If possible, retrieve the source code line described by `path` and `line_num`.
    ///
    /// Returns the empty string on failure.
    fn source_line(&self, path: &Path, line_num: usize) -> String {
        if let Ok(mut files) = self.source_files.lock() {
            // Open the source source file if it isn't already open.
            //
            // If we fail to open the source file, we just record `None` so that we don't try to
            // open it again later.
            //
            // Not using the entry API here to avoid unnecessary clones of `path`.
            if !files.contains_key(path) {
                let val = fs::File::open(path).ok().map(BufReader::new);
                files.insert(path.to_owned(), val);
            }
            // lookup cannot fail due to insertion above.
            let mut f = files.get_mut(path).unwrap().as_mut();
            if let Some(ref mut f) = f
                && f.seek(SeekFrom::Start(0)).is_ok()
                && let Some(Ok(line)) = f.lines().nth(line_num - 1)
            {
                // success
                return line.trim().to_string();
            }
        }
        // failure
        "".into()
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
                writeln!(f, "{gd}")?;
            }
        }

        for (funcidx, func) in self.funcs.iter_enumerated() {
            write!(f, "\n{}", func.display(self, Some(funcidx)))?;
        }
        Ok(())
    }
}

/// Deserialise an AOT module from the slice `data`.
pub(crate) fn deserialise_module(data: &[u8]) -> Result<Module, Box<dyn Error>> {
    let ((_, _), mut modu) = Module::from_bytes((data, 0))?;
    modu.create_func_cache();
    Ok(modu)
}

/// Deserialise and print IR from an on-disk file.
///
/// Used for support tooling (in turn used by tests too).
pub fn print_from_file(path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let data = fs::read(path)?;
    let ir = deserialise_module(&data)?;
    println!("{ir}");
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
pub(crate) struct BBlockInstIdx(usize);
index!(BBlockInstIdx);

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

/// An index into [Module::paths].
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PathIdx(usize);
index!(PathIdx);

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
    if let Ok(x) = CString::from_vec_with_nul(v)
        && let Ok(x) = x.into_string()
    {
        return Ok(x);
    }
    Err(DekuError::Parse(Cow::Borrowed("Couldn't map string")))
}

/// Convert the bytes of a null-terminated string into a PathBuf.
fn map_to_pathbuf(v: Vec<u8>) -> Result<PathBuf, DekuError> {
    Ok(PathBuf::from(map_to_string(v)?))
}

/// Helper function for deku `map` attribute. It is necessary to write all the types out in full to
/// avoid type inference errors, so it's easier to have a single helper function rather than inline
/// this into each `map` attribute.
fn map_to_tivec<I, T>(v: Vec<T>) -> Result<TiVec<I, T>, DekuError> {
    Ok(TiVec::from(v))
}

#[deku_derive(DekuRead)]
#[derive(Debug)]
struct RawPath {
    #[deku(until = "|v: &u8| *v == 0", map = "map_to_pathbuf")]
    path: PathBuf,
}

fn map_to_tivec_of_pathbuf(v: Vec<RawPath>) -> Result<TiVec<PathIdx, PathBuf>, DekuError> {
    let mut paths = TiVec::new();
    for path in v {
        paths.push(path.path);
    }
    Ok(paths)
}

/// Maps a flat vector of line-level debug info records into hashmap for easy lookups.
fn map_to_lineinfo(v: Vec<RawLineInfoRec>) -> Result<HashMap<InstId, LineInfoLoc>, DekuError> {
    Ok(v.into_iter()
        .map(|r| (r.inst_id, r.line))
        .collect::<HashMap<_, _>>())
}

/// A binary operator.
#[deku_derive(DekuRead)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[deku(id_type = "u8")]
pub(crate) enum BinOp {
    /// The canonicalised form of `Add` in JIT IR is (Var, Var) or (Var, Const).
    Add = 0,
    Sub,
    /// The canonicalised form of `Mul` in JIT IR is (Var, Var) or (Var, Const).
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
        write!(f, "{s}")
    }
}

/// Uniquely identifies an instruction within a [Module].
#[deku_derive(DekuRead)]
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) struct InstId {
    /// The index of the parent function.
    funcidx: FuncIdx,
    bbidx: BBlockIdx,
    iidx: BBlockInstIdx,
}

impl InstId {
    pub(crate) fn new(funcidx: FuncIdx, bbidx: BBlockIdx, iidx: BBlockInstIdx) -> Self {
        Self {
            funcidx,
            bbidx,
            iidx,
        }
    }

    pub(crate) fn funcidx(&self) -> FuncIdx {
        self.funcidx
    }

    pub(crate) fn bbidx(&self) -> BBlockIdx {
        self.bbidx
    }

    pub(crate) fn iidx(&self) -> BBlockInstIdx {
        self.iidx
    }
}

/// Uniquely identifies a basic block within a [Module].
#[derive(Clone, Copy, Debug, PartialEq)]
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

    /// Check whether `self` and `other` identify basic blocks from the same function and `self` is
    /// a static successor of `other`.
    pub(crate) fn static_intraprocedural_successor_of(&self, other: &Self, m: &Module) -> bool {
        let other_bb = m.bblock(other);
        let term_inst = other_bb.insts().last().unwrap();
        match term_inst {
            Inst::Br { succ } => *self == BBlockId::new(other.funcidx(), *succ),
            Inst::CondBr {
                true_bb, false_bb, ..
            } => {
                *self == BBlockId::new(other.funcidx(), *true_bb)
                    || *self == BBlockId::new(other.funcidx(), *false_bb)
            }
            Inst::Switch {
                default_dest,
                case_dests,
                ..
            } => {
                for bbidx in case_dests {
                    if *self == BBlockId::new(other.funcidx(), *bbidx) {
                        return true;
                    }
                }
                *self == BBlockId::new(other.funcidx(), *default_dest)
            }
            Inst::Ret { .. } => false,
            _ => panic!("invalid block terminator: {term_inst:?}"),
        }
    }
}

/// Predicates for use in numeric comparisons. These are directly based on [LLVM's `icmp`
/// semantics](https://llvm.org/docs/LangRef.html#icmp-instruction). All quotes below are taken
/// from there.
#[deku_derive(DekuRead)]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[deku(id_type = "u8")]
pub(crate) enum Predicate {
    /// "eq: yields true if the operands are equal, false otherwise. No sign
    /// interpretation is necessary or performed."
    Equal = 0,
    /// "ne: yields true if the operands are unequal, false otherwise. No sign
    /// interpretation is necessary or performed."
    NotEqual,
    /// "ugt: interprets the operands as unsigned values and yields true if op1 is
    /// greater than op2."
    UnsignedGreater,
    /// "uge: interprets the operands as unsigned values and yields true if op1 is
    /// greater than or equal to op2."
    UnsignedGreaterEqual,
    /// "ule: interprets the operands as unsigned values and yields true if op1 is
    /// less than or equal to op2."
    UnsignedLess,
    /// "ule: interprets the operands as unsigned values and yields true if op1 is
    /// less than or equal to op2."
    UnsignedLessEqual,
    /// "sgt: interprets the operands as signed values and yields true if op1 is greater
    /// than op2."
    SignedGreater,
    /// "sge: interprets the operands as signed values and yields true if op1 is
    /// greater than or equal to op2."
    SignedGreaterEqual,
    /// "slt: interprets the operands as signed values and yields true if op1 is less
    /// than op2."
    SignedLess,
    /// "sle: interprets the operands as signed values and yields true if op1 is less
    /// than or equal to op2."
    SignedLessEqual,
}

impl Predicate {
    /// Returns whether the comparison is signed.
    ///
    /// Not that [Self::Equal] and [Self::NotEqual] are considered not unsigned, since such
    /// comparisons are signedness agnostic.
    pub(crate) fn signed(&self) -> bool {
        match self {
            Self::Equal
            | Self::NotEqual
            | Self::UnsignedGreater
            | Self::UnsignedGreaterEqual
            | Self::UnsignedLess
            | Self::UnsignedLessEqual => false,
            Self::SignedGreater
            | Self::SignedGreaterEqual
            | Self::SignedLess
            | Self::SignedLessEqual => true,
        }
    }
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

/// Predicates for use in numeric comparisons.
#[deku_derive(DekuRead)]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[deku(id_type = "u8")]
pub(crate) enum FloatPredicate {
    // FIXME: eventually remove False/True (always false/always true) predicates. LLVM has them,
    // but we can lower these to constants in our IR.
    False = 0,
    OrderedEqual,
    OrderedGreater,
    OrderedGreaterEqual,
    OrderedLess,
    OrderedLessEqual,
    OrderedNotEqual,
    Ordered,
    Unordered,
    UnorderedEqual,
    UnorderedGreater,
    UnorderedGreaterEqual,
    UnorderedLess,
    UnorderedLessEqual,
    UnorderedNotEqual,
    True,
}

impl Display for FloatPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The `f_` prefix ensures there is no ambiguity with integer predicates.
        //
        // For example "ule" could mean "unsigned less or equal" (for integers) or "unordered less
        // or equal" (for floats). Without prefixing the float version comparison instructions
        // using those would both stringify as: `%res: ty = ule %op1, %op2`.
        let s = match self {
            Self::False => "f_false",
            Self::OrderedEqual => "f_oeq",
            Self::OrderedGreater => "f_ogt",
            Self::OrderedGreaterEqual => "f_oge",
            Self::OrderedLess => "f_olt",
            Self::OrderedLessEqual => "f_ole",
            Self::OrderedNotEqual => "f_one",
            Self::Ordered => "f_ord",
            Self::Unordered => "f_uno",
            Self::UnorderedEqual => "f_ueq",
            Self::UnorderedGreater => "f_ugt",
            Self::UnorderedGreaterEqual => "f_uge",
            Self::UnorderedLess => "f_ult",
            Self::UnorderedLessEqual => "f_ule",
            Self::UnorderedNotEqual => "f_une",
            Self::True => "f_true",
        };
        write!(f, "{s}")
    }
}

/// The operations that a [Inst::Cast] can perform.
///
/// FIXME: There are many other operations that we can add here on-demand. See the inheritance
/// hierarchy here: https://llvm.org/doxygen/classllvm_1_1CastInst.html
#[deku_derive(DekuRead)]
#[derive(Debug, Clone, Copy)]
#[deku(id_type = "u8")]
pub(crate) enum CastKind {
    SExt = 0,
    ZeroExtend = 1,
    Trunc = 2,
    SIToFP = 3,
    FPExt = 4,
    FPToSI = 5,
    BitCast = 6,
    PtrToInt = 7,
    IntToPtr = 8,
    UIToFP = 9,
}

impl Display for CastKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::SExt => "sext",
            Self::ZeroExtend => "zext",
            Self::Trunc => "trunc",
            Self::SIToFP => "si_to_fp",
            Self::FPExt => "fp_ext",
            Self::FPToSI => "fp_to_si",
            Self::BitCast => "bitcast",
            Self::IntToPtr => "int_to_ptr",
            Self::PtrToInt => "ptr_to_int",
            Self::UIToFP => "ui_to_fp",
        };
        write!(f, "{s}")
    }
}

#[deku_derive(DekuRead)]
#[derive(Debug, Clone, Hash)]
#[deku(id_type = "u8")]
pub(crate) enum Operand {
    #[deku(id = "0")]
    Const(ConstIdx),
    #[deku(id = "1")]
    Local(InstId),
    #[deku(id = "2")]
    Global(GlobalDeclIdx),
    #[deku(id = "3")]
    Func(FuncIdx),
}

impl Operand {
    /// For a [Self::Variable] operand return the instruction that defines the variable.
    ///
    /// Panics for other kinds of operand.
    ///
    /// OPT: This is expensive.
    pub(crate) fn to_inst<'a>(&self, aotmod: &'a Module) -> &'a Inst {
        let Self::Local(iid) = self else { panic!() };
        &aotmod.funcs[iid.funcidx].bblocks[iid.bbidx].insts[iid.iidx]
    }

    /// Returns the [Ty] of the operand.
    pub(crate) fn type_<'a>(&self, m: &'a Module) -> &'a Ty {
        match self {
            Self::Local(_) => {
                // The `unwrap` can't fail for a `LocalVariable`.
                self.to_inst(m).def_type(m).unwrap()
            }
            Self::Const(cidx) => m.type_(m.const_(*cidx).tyidx()),
            Self::Global(_) => {
                // As is the case for LLVM IR, globals are always pointer-typed in Yk AOT IR.
                &Ty::Ptr
            }
            Self::Func(funcidx) => m.type_(m.func(*funcidx).tyidx()),
        }
    }

    /// Return the `InstId` of a local variable operand. Panics if called on other kinds of
    /// operands.
    pub(crate) fn to_inst_id(&self) -> InstId {
        let Self::Local(iid) = self else { panic!() };
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
            Operand::Local(iid) => {
                write!(f, "%{}_{}", usize::from(iid.bbidx), usize::from(iid.iidx))
            }
            Operand::Global(gid) => write!(f, "@{}", self.m.global_decls[*gid].name()),
            Operand::Func(fidx) => write!(f, "{}", self.m.funcs[*fidx].name()),
        }
    }
}

#[deku_derive(DekuRead)]
#[derive(Clone, Debug)]
pub(crate) struct DeoptSafepoint {
    pub(crate) id: u64,
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
        write!(f, "[safepoint: {}i64, ({})]", self.safepoint.id, lives_s)
    }
}

/// An instruction.
///
/// An instruction is conceptually an [Opcode] and a list of [Operand]s. The semantics of the
/// instruction, and the meaning of the operands, are determined by the opcode.
///
/// Insts that compute a value define a new local variable in the parent [Func]. In such a
/// case the newly defined variable can be referenced in the operands of later instructions by the
/// [InstId] of the [Inst] that defined the variable.
///
/// In other words, an instruction and the variable it defines are both identified by the same
/// [InstId].
///
/// The type of the variable defined by an instruction (if any) can be determined by
/// [Inst::def_type()].
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[repr(u8)]
#[deku(id_type = "u8")]
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
        tyidx: TyIdx,
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
        safepoint: DeoptSafepoint,
    },
    #[deku(id = "16")]
    Select {
        cond: Operand,
        trueval: Operand,
        falseval: Operand,
    },
    #[deku(id = "17")]
    LoadArg { arg_idx: usize, ty_idx: TyIdx },
    #[deku(id = "18")]
    FCmp {
        tyidx: TyIdx,
        lhs: Operand,
        pred: FloatPredicate,
        rhs: Operand,
    },
    #[deku(id = "19")]
    Promote {
        tyidx: TyIdx,
        val: Operand,
        safepoint: DeoptSafepoint,
    },
    #[deku(id = "20")]
    FNeg { val: Operand },
    #[deku(id = "21")]
    DebugStr { msg: Operand },
    #[deku(id = "22")]
    ExtractValue {
        tyidx: TyIdx,
        op: Operand,
        #[deku(temp)]
        num_indices: usize,
        #[deku(count = "num_indices")]
        indices: Vec<usize>,
    },
    #[deku(id = "255")]
    Unimplemented {
        tyidx: TyIdx,
        #[deku(until = "|v: &u8| *v == 0", map = "map_to_string")]
        llvm_inst_str: String,
    },
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
                        return format!("%{bbidx}_{iidx}");
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
                    if ty != &Ty::Void { Some(ty) } else { None }
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
            Self::Phi { tyidx, .. } => {
                // Indexing cannot crash: correct PHI nodes have at least one incoming value.
                Some(m.type_(*tyidx))
            }
            Self::IndirectCall { ftyidx, .. } => {
                // The type of the newly-defined local is the return type of the callee.
                if let Ty::Func(ft) = m.type_(*ftyidx) {
                    let ty = m.type_(ft.ret_ty);
                    if ty != &Ty::Void { Some(ty) } else { None }
                } else {
                    panic!(); // IR malformed.
                }
            }
            Self::Select {
                cond: _, trueval, ..
            } => Some(trueval.type_(m)),
            Self::LoadArg { arg_idx: _, ty_idx } => Some(m.type_(*ty_idx)),
            Self::Unimplemented { tyidx, .. } => {
                let ty = m.type_(*tyidx);
                if ty != &Ty::Void { Some(ty) } else { None }
            }
            Self::Nop => None,
            Self::FCmp { tyidx, .. } => Some(m.type_(*tyidx)),
            Self::Promote { tyidx, .. } => Some(m.type_(*tyidx)),
            Self::FNeg { val } => Some(val.type_(m)),
            Self::DebugStr { .. } => None,
            Self::ExtractValue {
                op: _,
                tyidx,
                indices: _,
            } => Some(m.type_(*tyidx)),
        }
    }

    /// Returns whether `self` is a call to the control point.
    pub(crate) fn is_control_point(&self, m: &Module) -> bool {
        // FIXME: We don't really expect any other patchpoints here, but it would be nice to check
        // the third argument is actually the control point. This would require the ability to
        // reconstitute a string from constant bytes though.
        match self {
            Self::Call { callee, .. } => {
                m.func(*callee).name == "llvm.experimental.patchpoint.void"
            }
            _ => false,
        }
    }

    pub(crate) fn safepoint(&'static self) -> Option<&'static DeoptSafepoint> {
        match self {
            Self::Call { safepoint, .. } => safepoint.as_ref(),
            Self::CondBr { safepoint, .. } => Some(safepoint),
            _ => None,
        }
    }

    pub(crate) fn is_debug_call(&self, m: &Module) -> bool {
        match self {
            Self::Call { callee, .. } => m.func(*callee).name == LLVM_DEBUG_CALL_NAME,
            _ => false,
        }
    }

    pub(crate) fn display<'a>(
        &'a self,
        m: &'a Module,
        instid: Option<InstId>,
    ) -> DisplayableInst<'a> {
        DisplayableInst {
            instruction: self,
            instid,
            m,
        }
    }
}

pub(crate) struct DisplayableInst<'a> {
    instruction: &'a Inst,
    /// The ID of the instruction.
    ///
    /// Required to find line-level debugging info for the instruction.
    instid: Option<InstId>,
    m: &'a Module,
}

impl fmt::Display for DisplayableInst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If we have source-level line info for the instruction, print a comment line containing
        // the source line.
        if let Some(instid) = &self.instid
            && let Some(li) = self.m.line_infos.get(instid)
        {
            let path = self.m.path(li.pathidx);
            // Sometimes the filename cannot be determined.
            let basename = path.file_name().unwrap_or(OsStr::new("<unknown-filename>"));
            // We assume the filename is expressible in UTF-8.
            let src = self.m.source_line(path, li.line_num);
            write!(
                f,
                "# {}:{}: {}\n    ",
                basename.to_str().unwrap(),
                li.line_num,
                src
            )?;
        }
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
                let func = self.m.func(*callee);
                let idem = if func.is_idempotent() {
                    "idempotent "
                } else {
                    ""
                };
                write!(f, "call {}{}({}){}", idem, func.name(), args_s, safepoint_s)
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
                ..
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
                safepoint,
            } => {
                let args_s = args
                    .iter()
                    .map(|a| a.display(self.m).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "icall {}({}) {}",
                    callop.display(self.m),
                    args_s,
                    safepoint.display(self.m)
                )
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
            Inst::LoadArg { arg_idx, ty_idx: _ } => write!(f, "arg({arg_idx})",),
            Inst::Unimplemented { llvm_inst_str, .. } => {
                write!(f, "unimplemented <<{llvm_inst_str}>>")
            }
            Inst::Nop => write!(f, "nop"),
            Inst::FCmp { lhs, pred, rhs, .. } => {
                write!(f, "{pred} {}, {}", lhs.display(self.m), rhs.display(self.m))
            }
            Inst::Promote { val, safepoint, .. } => {
                write!(
                    f,
                    "promote {} {}",
                    val.display(self.m),
                    safepoint.display(self.m)
                )
            }
            Inst::FNeg { val } => {
                write!(f, "fneg {}", val.display(self.m),)
            }
            Inst::DebugStr { msg } => write!(f, "debug_str {}", msg.display(self.m)),
            Inst::ExtractValue {
                op,
                tyidx: _,
                indices,
            } => {
                write!(f, "extractvalue {}, {:?}", op.display(self.m), indices,)
            }
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
    pub(crate) insts: TiVec<BBlockInstIdx, Inst>,
}

impl BBlock {
    // Returns true if this block is terminated by a return, false otherwise.
    pub fn is_return(&self) -> bool {
        matches!(self.insts.last().unwrap(), Inst::Ret { .. })
    }

    pub(crate) fn display<'a>(
        &'a self,
        m: &'a Module,
        bbid: Option<BBlockId>,
    ) -> DisplayableBBlock<'a> {
        DisplayableBBlock {
            bblock: self,
            m,
            bbid,
        }
    }

    pub(crate) fn insts(&self) -> &TiSlice<BBlockInstIdx, Inst> {
        self.insts.as_slice()
    }
}

pub(crate) struct DisplayableBBlock<'a> {
    bblock: &'a BBlock,
    /// The ID of the basic block.
    ///
    /// Required to find line-level debugging info for the instructions in the block.
    bbid: Option<BBlockId>,
    m: &'a Module,
}

impl fmt::Display for DisplayableBBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (instidx, inst) in self.bblock.insts.iter_enumerated() {
            let instid = self
                .bbid
                .as_ref()
                .map(|bbid| InstId::new(bbid.funcidx, bbid.bbidx, instidx));
            writeln!(f, "    {}", inst.display(self.m, instid))?;
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
    flags: u8,
    #[deku(temp)]
    num_bblocks: usize,
    #[deku(count = "num_bblocks", map = "map_to_tivec")]
    bblocks: TiVec<BBlockIdx, BBlock>,
}

const FUNCFLAG_OUTLINE: u8 = 1;
const FUNCFLAG_IDEMPOTENT: u8 = 1 << 1;
const FUNCFLAG_INDIRECT_INLINE: u8 = 1 << 2;

impl Func {
    pub(crate) fn is_declaration(&self) -> bool {
        self.bblocks.is_empty()
    }

    pub(crate) fn is_outline(&self) -> bool {
        self.flags & FUNCFLAG_OUTLINE != 0
    }

    pub(crate) fn is_idempotent(&self) -> bool {
        self.flags & FUNCFLAG_IDEMPOTENT != 0
    }

    pub(crate) fn is_indirect_inline(&self) -> bool {
        self.flags & FUNCFLAG_INDIRECT_INLINE != 0
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

    pub(crate) fn display<'a>(
        &'a self,
        m: &'a Module,
        funcidx: Option<FuncIdx>,
    ) -> DisplayableFunc<'a> {
        DisplayableFunc {
            func_: self,
            m,
            funcidx,
        }
    }

    /// Determine if the function contains any calls to the named function.
    pub(crate) fn contains_call_to(&self, m: &Module, func_name: &str) -> bool {
        for bb in &self.bblocks {
            for inst in bb.insts() {
                if let Inst::Call { callee, .. } = inst
                    && m.func(*callee).name() == func_name
                {
                    return true;
                }
            }
        }
        false
    }
}

pub(crate) struct DisplayableFunc<'a> {
    func_: &'a Func,
    /// The index of the function in the module.
    ///
    /// Required to locate line-level debugging info for the instructions in the function.
    funcidx: Option<FuncIdx>,
    m: &'a Module,
}

impl fmt::Display for DisplayableFunc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ty = &self.m.types[self.func_.tyidx];
        if let Ty::Func(fty) = ty {
            let mut attrs = Vec::new();
            if self.func_.is_idempotent() {
                attrs.push("yk_idempotent");
            }
            if self.func_.is_outline() {
                attrs.push("yk_outline");
            }
            if self.func_.is_indirect_inline() {
                attrs.push("yk_indirect_inline");
            }
            let attrs = if !attrs.is_empty() {
                &format!("#[{}]\n", attrs.join(", "))
            } else {
                ""
            };
            write!(
                f,
                "{}func {}({}",
                attrs,
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
                for (bbidx, b) in self.func_.bblocks.iter_enumerated() {
                    let bbid = self.funcidx.map(|funcidx| BBlockId::new(funcidx, bbidx));
                    write!(
                        f,
                        "  bb{}:\n{}",
                        usize::from(bbidx),
                        b.display(self.m, bbid)
                    )?;
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
    bitw: u32,
}

impl IntegerTy {
    /// Create a new integer type with the specified number of bits.
    #[cfg(test)]
    pub(crate) fn new(bitw: u32) -> Self {
        debug_assert!(bitw > 0 && bitw <= 0x800000);
        Self { bitw }
    }

    /// Return the number of bits (1..2^23 (inc.)) this integer spans.
    pub(crate) fn bitw(&self) -> u32 {
        debug_assert!(self.bitw > 0 && self.bitw <= 0x800000);
        self.bitw
    }

    /// Return the number of bytes required to store this integer type.
    ///
    /// Padding for alignment is not included.
    pub(crate) fn bytew(&self) -> u32 {
        self.bitw().div_ceil(8)
    }

    /// Format a constant integer value that is of the type described by `self`.
    fn const_to_string(&self, c: &ConstVal) -> String {
        const_int_bytes_to_string(self.bitw, c.bytes())
    }
}

impl Display for IntegerTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i{}", self.bitw)
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
#[deku(id_type = "u8")]
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
        write!(f, "{s}")
    }
}

#[deku_derive(DekuRead)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StructTy {
    /// Total size in bits of this struct including alignment.
    bit_size: u64,
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
    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableStructTy<'a> {
        DisplayableStructTy {
            struct_type: self,
            m,
        }
    }

    pub(crate) fn bit_size(&self) -> u64 {
        self.bit_size
    }

    pub(crate) fn field_tyidxs(&self) -> &Vec<TyIdx> {
        &self.field_tyidxs
    }

    pub(crate) fn field_bit_offs(&self) -> &Vec<usize> {
        &self.field_bit_offs
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
#[deku(id_type = "u8")]
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
                format!("{pval:#x}")
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
                    FloatTy::Double => format!("{dval}double"),
                }
            }
            Self::Unimplemented(s) => format!("?cst<{s}>"),
        }
    }

    pub(crate) fn display<'a>(&'a self, m: &'a Module) -> DisplayableTy<'a> {
        DisplayableTy { type_: self, m }
    }

    pub(crate) fn bitw(&self) -> u32 {
        match self {
            Self::Integer(it) => it.bitw(),
            Self::Ptr => todo!(),
            _ => todo!(),
        }
    }

    pub(crate) fn bytew(&self) -> u32 {
        match self {
            Self::Integer(it) => it.bytew(),
            Self::Ptr => u32::try_from(std::mem::size_of::<*const ()>()).unwrap(),
            _ => todo!(),
        }
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
            Ty::Integer(x) => write!(f, "{x}"),
            Ty::Ptr => write!(f, "ptr"),
            Ty::Func(ft) => write!(f, "{}", ft.display(self.m)),
            Ty::Struct(st) => write!(f, "{}", st.display(self.m)),
            Ty::Float(ft) => write!(f, "{ft}"),
            Ty::Unimplemented(s) => write!(f, "?ty<{s}>"),
        }
    }
}

/// A (potentially implemented) constant.
///
/// Constants not handled by the ykllvm serialiser become `Const::Unimplemented`.
#[deku_derive(DekuRead)]
#[derive(Debug)]
#[deku(id_type = "u8")]
pub(crate) enum Const {
    #[deku(id = "0")]
    Val(ConstVal),
    #[deku(id = "1")]
    Unimplemented {
        tyidx: TyIdx,
        #[deku(until = "|v: &u8| *v == 0", map = "map_to_string")]
        llvm_const_str: String,
    },
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
            Const::Unimplemented { llvm_const_str, .. } => {
                panic!("unimplemented const: {llvm_const_str}")
            }
        }
    }

    pub(crate) fn tyidx(&self) -> TyIdx {
        match self {
            Self::Val(cv) => cv.tyidx(),
            Self::Unimplemented { tyidx, .. } => *tyidx,
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
            Const::Unimplemented { llvm_const_str, .. } => {
                write!(f, "unimplemented <<{llvm_const_str}>>")
            }
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
            let it = IntegerTy { bitw: num_bits };
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

        let expect_bytes = rng
            .rev()
            .fold("".to_string(), |acc, i| format!("{acc}{i:02x}"));
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
        let mut m = Module::default();
        m.types.push(Ty::Integer(IntegerTy::new(8)));
        let c = Const::Unimplemented {
            tyidx: TyIdx::new(0),
            llvm_const_str: "someoperand".into(),
        };
        assert_eq!(c.display(&m).to_string(), "unimplemented <<someoperand>>");
    }

    #[test]
    fn integer_type_sizes() {
        for i in 1..8 {
            assert_eq!(IntegerTy::new(i).bytew(), 1);
        }
        for i in 9..16 {
            assert_eq!(IntegerTy::new(i).bytew(), 2);
        }
        assert_eq!(IntegerTy::new(127).bytew(), 16);
        assert_eq!(IntegerTy::new(128).bytew(), 16);
        assert_eq!(IntegerTy::new(129).bytew(), 17);
    }

    #[test]
    fn stringify_func_types() {
        let mut m = Module::default();

        let i8_tyidx = TyIdx::new(m.types.len());
        m.types.push(Ty::Integer(IntegerTy { bitw: 8 }));
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
