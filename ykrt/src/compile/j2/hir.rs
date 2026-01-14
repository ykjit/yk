//! High-level trace IR (HIR).
//!
//! HIR is a representation of a trace that can be thought of as a highly simplified version of
//! LLVM IR. There is no perfect IR: HIR's aim is to favour simplicity and regularity over other
//! goals (e.g. reducing the number of instructions in a trace). For example, instruction operands
//! can only refer to other instructions. That means that operands do not need to encode a type,
//! and they cannot refer to a constant. Roughly speaking this LLVM IR:
//!
//! ```text
//! %4 = ...
//! %5 = add i32 %4, 8i32
//! ```
//!
//! is represented in HIR as:
//!
//! ```text
//! %4: i32 = ...
//! %5: i32 = 8
//! %6: i32 = add %4, %5
//! ```
//!
//! From our perspective, the main advantage of the latter is that pattern matching HIR is simple
//! and uniform.
//!
//! In general, HIR instructions can represent the most common aspects of their LLVM equivalents:
//! we do not try to represent every possibility that LLVM IR can represent. That means there is
//! AOT code that we cannot (currently) represent as HIR: we will have to deal with such cases as
//! and when we see them.
//!
//!
//! ## Terminology
//!
//! "Variables", "instructions", and "values" are near-synonyms in our context: they *do* mean
//! slightly different things, but using the "wrong" term never leads to ambiguity so we do not
//! obsess over it. "Variables" are `%4` and so on. "Instructions" are `add` and so on. "Values"
//! are the run-time values produced by an instruction and assigned to a variable.
//!
//!
//! ## Differences from LLVM's IR
//!
//! In a small number of cases, we deliberately diverge from LLVM IR. Notably, the catch-all
//! `getelementptr` instruction has no single equivalent in HIR: instead, we represent it as
//! [PtrAdd] or [DynPtrAdd] as appropriate.
//!
//!
//! ## Instruction numbering / referencing
//!
//! Instructions are always referenced by an [InstIdx]: this is deliberately an index into an array
//! for efficiency reasons. Thus instructions are always numbered 0..*n*.
//!
//!
//! ## [TyIdx]s can be replied on for type comparisons
//!
//! HIR _must_ map the same [Ty] to a single [TyIdx] (i.e. no matter how often a given type is
//! inserted into a [Module], it must produce the same [TyIdx]). Thus type comparisons are simple
//! integer comparisons.
//!
//!
//! ## [ConstIdx]s cannot be relied upon for comparisons
//!
//! HIR should try to deduplicate [Const]s where possible, but this cannot be relied upon: in
//! particular, floating point numbers do not a have simple notion of "is equivalent". Where
//! [Const] deduplication does occur, it must be done with care to ensure that only constants which
//! in all cases could be considered equal are deduplicated. In practise, this means that floating
//! point numbers should be considered as bit patterns.
//!
//!
//! ## Modules vs. blocks
//!
//! A HIR [Module] is a complete, high-level, representation of our intuitive notion of a "trace".
//!
//! Traces come in different flavours, which we represent as a pair ([TraceStart], [TraceEnd]).
//! This allows us to subsume the traditional notion of a "loop trace" and so on. 5 of the 6
//! possibilities are valid ([TraceStart]s in rows, [TraceEnd]s in columns):
//!
//! |              | Coupler | Loop | Return |
//! |--------------|---------|------|--------|
//! | ControlPoint | ✓       | ✓    | ✓      |
//! | Guard        | ✓       | ✗    | ✓      |
//!
//! Note: this doesn't mean that a guard trace can't end up back at the start of the loop the guard
//! was in. It does mean that we don't need to treat this as a special option: it is a guard
//! coupler that happens to end up back at the same trace.
//!
//! Depending on the kind of trace (loop etc.), the [TraceEnd] will contain one or more [Block]s
//! (where a block is a sequence of instructions). Blocks can (or, more accurately, "will")
//! represent two subtly different things: "body" blocks (including peeled blocks) represent the
//! core of a trace; "guard" blocks represent side-exits (note: we currently don't do this!). Only
//! body blocks can contain guards and thus reference guard blocks.
//!
//! All blocks have an entry and exit point, with corresponding [VarLocs], though the details
//! differ between body and guard blocks. Body blocks start with 0 or more [Arg] / [Const]
//! instructions and end with a [Term] instruction. Guard blocks' entry is defined by the
//! corresponding [Guard] instruction and their exit by the relevant AOT [DeoptSafepoint].
//!
//! All blocks share [Ty]s: thus a [TyIdx] is relative to a module, not a block.
//!
//!
//! ## Arguments
//!
//! Blocks start with 0 or more [Arg] or [Const] instructions. These have a 1:1 correspondance to
//! the number of [VarLocs] for that block.
//!
//!
//! ## Guards
//!
//! Guards in a trace map to a [GuardRestore], which contains the information necessary for both
//! deopt and side-tracing. Note that multiple guards may map to a single [GuardRestore] (e.g. due
//! to loop peeling).
//!
//!
//! ## Canonicalisation
//!
//! Canonicalisation in HIR is only possible during the optimisation phase. It does two things:
//!
//! 1. It rewrites all operands to their most recently identified equivalents.
//! 2. It optionally rewrites the instruction (e.g. to favour constants on the RHS).
//!
//! The first of these is objective, but the second is subjective. That said, common-sense suggests
//! that regularity helps understanding: for example, whenever possible, canonicalisation on binary
//! operations (`add`, `and`, `icmp eq` etc.) favours references to constants on the right-hand
//! side. Note that canonicalisation never changes the _kind_ of an instruction.
//!
//! For example, for this trace:
//!
//! ```text
//! %0: i8 = arg
//! %1: i8 = 10
//! %2: i8 = add %1, %0
//! ```
//!
//! the last instruction will be canonicalised to `add %0, %1` i.e. favouring the constant on the
//! right-hand side of the `add`. Canonicalisation is inherently subjective and you should check
//! the details of each instruction for how it canonicalises itself: `add`, for example, could just
//! as easily, have favoured the constant being on the left-hand side. That said, common-sense
//! suggests that regularity helps understanding: for example, whenever possible, canonicalisation
//! on binary operations (`add`, `and`, `eq` etc.) favours references to constants on the
//! right-hand side.
//!
//!
//! ## To / from text
//!
//! HIR can be pretty-printed to a string and, via [super::hir_parser::str_to_mod] created from a
//! string. The textual representations are non-normative: the text created as output and text
//! accepted as input deliberately have slightly different syntaxes (mostly so that the input
//! format can be easier to write), and those may change at any point in the future.
//!
//!
//! ## Well-formedness
//!
//! A HIR module can be checked for well-formedness with [Mod::assert_well_formed]. Well-formedness
//! is a lightweight check that HIR's basic rules have been adhered to: it does not guarantee that
//! the trace is semantically well-formed.

use crate::{
    compile::{
        j2::{
            compiled_trace::J2CompiledTrace,
            hir_to_asm::AsmGuardIdx,
            opt::EquivIIdxT,
            regalloc::{RegT, VarLocs},
        },
        jitc_yk::{
            aot_ir::{self, DeoptSafepoint},
            arbbitint::ArbBitInt,
        },
    },
    mt::TraceId,
};
use enum_dispatch::enum_dispatch;
use index_vec::IndexVec;
use smallvec::SmallVec;
use std::{
    assert_matches::assert_matches,
    cmp::min,
    collections::HashMap,
    ffi::c_void,
    fmt::{Display, Formatter},
    ops::{Bound, RangeBounds},
    sync::Arc,
};
use strum::{EnumCount, EnumDiscriminants};

/// A representation of a "module like" object.
///
/// Abstracting this out as a trait allows us to use the same interface for both "fully complete"
/// modules (i.e. a literal [Mod]) as well as "being built" modules (i.e. for [super::opt::OptT]).
pub(super) trait ModLikeT {
    fn ty(&self, tyidx: TyIdx) -> &Ty;

    /// Returns the [FuncTy] at `tyidx`. This is a convenience function over [Self::ty].
    ///
    /// # Panics
    ///
    /// If `tyidx` does not reference a [FuncTy].
    fn func_ty(&self, tyidx: TyIdx) -> &FuncTy {
        let Ty::Func(x) = self.ty(tyidx) else {
            panic!()
        };
        x
    }

    /// Return a reference to the [GuardExtra] `geidx`.
    fn gextra(&self, geidx: GuardExtraIdx) -> &GuardExtra;

    /// Return a mutable reference to the [GuardExtra] `geidx`.
    fn gextra_mut(&mut self, geidx: GuardExtraIdx) -> &mut GuardExtra;

    /// If logging was enabled, returns `Some(linker_name)` if `addr` has a known name, or `None`
    /// otherwise.
    fn addr_to_name(&self, addr: usize) -> Option<&str>;
}

/// A representation of a "block like" object.
///
/// Abstracting this out as a trait allows us to use the same interface for both "fully complete"
/// blocks (i.e. a literal [Block]) as well as "being built" blocks (i.e. for [super::opt::OptT]).
pub(super) trait BlockLikeT {
    /// Return the instruction at `iidx`.
    ///
    /// # Panics
    ///
    /// If `iidx` is out of bounds.
    fn inst(&self, iidx: InstIdx) -> &Inst;

    /// Return the bit width of the instruction `iidx`. This is a convenience function over other
    /// public functions.
    ///
    /// # Panics
    ///
    /// If `iidx` is out of bounds.
    fn inst_bitw(&self, m: &dyn ModLikeT, iidx: InstIdx) -> u32 {
        self.inst(iidx).ty(m).bitw()
    }
}

/// A HIR module representing an intuitive notion of a "trace". Roughly speaking a module contains
/// one or more [Block]s (depending on the kind of trace represented) and zero or more
/// [GuardRestore]s.
#[derive(Debug)]
pub(super) struct Mod<Reg: RegT> {
    pub trid: TraceId,
    pub trace_start: TraceStart<Reg>,
    pub trace_end: TraceEnd<Reg>,
    pub tys: IndexVec<TyIdx, Ty>,
    /// Extra information that is too big to fit in a [Guard] instruction.
    pub guard_extras: IndexVec<GuardExtraIdx, GuardExtra>,
    /// A map of names to pointers. Will be `None` if logging was disabled.
    pub addr_name_map: Option<HashMap<usize, Option<String>>>,
}

impl<Reg: RegT> Mod<Reg> {
    /// Check that this module is well-formed, panicking if it is not.
    #[allow(dead_code)]
    pub(super) fn assert_well_formed(&self) {
        // Guarantee that a given [Ty] appears only once in `self.tys`.
        for (i, ty) in self.tys.iter().enumerate() {
            for ty2 in self.tys.iter().skip(i + 1) {
                assert_ne!(ty, ty2, "Type '{ty:?}' appears more than once");
            }
        }

        match &self.trace_end {
            TraceEnd::Coupler { .. } | TraceEnd::Loop { .. } | TraceEnd::Return { .. } => todo!(),
            #[cfg(test)]
            TraceEnd::Test { entry_vlocs, block } => {
                block.assert_well_formed(self, entry_vlocs, entry_vlocs);
            }
        }
    }

    pub(super) fn guard_extra(&self, geidx: GuardExtraIdx) -> &GuardExtra {
        &self.guard_extras[geidx]
    }

    pub(super) fn guard_extras(&self) -> &[GuardExtra] {
        self.guard_extras.as_raw_slice()
    }
}

impl<Reg: RegT> Display for Mod<Reg> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match &self.trace_end {
            TraceEnd::Coupler { entry, .. } => write!(f, "{}", entry.to_string(self)),
            TraceEnd::Loop { entry, .. } => write!(f, "{}", entry.to_string(self)),
            TraceEnd::Return { entry, .. } => write!(f, "{}", entry.to_string(self)),
            #[cfg(test)]
            TraceEnd::Test { block, .. } => write!(f, "{}", block.to_string(self)),
        }
    }
}

impl<Reg: RegT> ModLikeT for Mod<Reg> {
    fn addr_to_name(&self, addr: usize) -> Option<&str> {
        self.addr_name_map
            .as_ref()
            .and_then(|x| x.get(&addr))
            .and_then(|x| x.as_ref().map(|y| y.as_str()))
    }

    fn gextra(&self, geidx: GuardExtraIdx) -> &GuardExtra {
        &self.guard_extras[geidx]
    }

    fn gextra_mut(&mut self, geidx: GuardExtraIdx) -> &mut GuardExtra {
        &mut self.guard_extras[geidx]
    }

    fn ty(&self, tyidx: TyIdx) -> &Ty {
        &self.tys[tyidx]
    }
}

/// Where did this trace start from?
#[derive(Debug)]
pub(super) enum TraceStart<Reg: RegT> {
    /// This trace started from a control point.
    ControlPoint {
        entry_safepoint: &'static DeoptSafepoint,
    },
    /// This trace started from a guard failing.
    Guard {
        src_ctr: Arc<J2CompiledTrace<Reg>>,
        src_gidx: AsmGuardIdx,
        entry_vlocs: Vec<VarLocs<Reg>>,
    },
    /// This is a trace intended for unit testing.
    #[cfg(test)]
    Test,
}

/// Where did this trace end?
#[derive(Debug)]
pub(super) enum TraceEnd<Reg: RegT> {
    /// This trace ended at a different [crate::location::Location] than it started from.
    Coupler {
        entry: Block,
        tgt_ctr: Arc<J2CompiledTrace<Reg>>,
    },
    /// This trace ended at the same [crate::location::Location] that it started from, thus forming
    /// a loop.
    Loop { entry: Block, peel: Option<Block> },
    /// This trace ended up early when a `return` statement in the control point loop was
    /// encountered.
    Return {
        entry: Block,
        exit_safepoint: &'static DeoptSafepoint,
    },
    /// This is a trace intended for unit testing.
    #[cfg(test)]
    Test {
        entry_vlocs: Vec<VarLocs<Reg>>,
        block: Block,
    },
}

/// An ordered sequence of instructions.
#[derive(Debug)]
pub(super) struct Block {
    pub insts: IndexVec<InstIdx, Inst>,
}

impl Block {
    #[allow(dead_code)]
    fn assert_well_formed<Reg: RegT>(
        &self,
        m: &dyn ModLikeT,
        entry_vlocs: &[VarLocs<Reg>],
        exit_vlocs: &[VarLocs<Reg>],
    ) {
        for (iidx, inst) in self.insts_iter(..) {
            if iidx < InstIdx::from(entry_vlocs.len()) {
                assert_matches!(
                    inst,
                    Inst::Arg(_) | Inst::Const(_),
                    "%{iidx:?}: entry instructions must be 'arg' or constants"
                );
            } else if let Inst::Arg(_) = inst {
                panic!("%{iidx:?}: 'arg' instructions cannot appear after trace entry");
            } else if iidx == self.insts.last_idx()
                && let Inst::Term(Term(term_vars)) = inst
            {
                assert_eq!(
                    exit_vlocs.len(),
                    term_vars.len(),
                    "%{iidx:?}: number of term vars does not match number of term VarLocs"
                );

                for (i, (x, y)) in entry_vlocs.iter().zip(exit_vlocs.iter()).enumerate() {
                    let entry_ty = self.insts[InstIdx::from(i)].ty(m);
                    let exit_ty = self.insts[term_vars[i]].ty(m);
                    if entry_ty != exit_ty {
                        panic!(
                            "%{iidx:?}: term var '%{:?}' at position '{i}' does match type of %{i}",
                            term_vars[i]
                        );
                    }

                    if x != y {
                        panic!(
                            "%{iidx:?}: term var '%{:?}' at position '{i}' does not match entry VarLoc",
                            term_vars[i]
                        );
                    }
                }
            }

            for op_iidx in inst.iter_iidxs(m) {
                assert!(
                    op_iidx < iidx,
                    "%{iidx:?}: forward reference to %{op_iidx:?}"
                )
            }
            if let Inst::Term(_) = inst {
                assert_eq!(
                    iidx,
                    self.insts.last_idx(),
                    "%{iidx:?}: term must be the last instruction in a trace"
                );
            }
            inst.assert_well_formed(m, self, iidx);
        }
    }

    pub(super) fn insts_iter<T>(
        &self,
        range: T,
    ) -> impl DoubleEndedIterator<Item = (InstIdx, &Inst)> + '_
    where
        T: RangeBounds<InstIdx>,
    {
        let start = match range.start_bound() {
            Bound::Included(x) => min(usize::from(*x), self.insts.len()),
            Bound::Excluded(x) => min(usize::from(*x + 1), self.insts.len()),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(x) => min(usize::from(*x + 1), self.insts.len()),
            Bound::Excluded(x) => min(usize::from(*x), self.insts.len()),
            Bound::Unbounded => self.insts.len(),
        };
        (start..end).map(|i| {
            let i = InstIdx::from(i);
            (i, &self.insts[i])
        })
    }

    pub(super) fn insts_len(&self) -> usize {
        self.insts.len()
    }

    /// Return true if there are heap effects in `range` that interfere with the instruction at
    /// `on`.
    pub(super) fn heap_effects_on<T: RangeBounds<InstIdx>>(&self, _on: InstIdx, range: T) -> bool {
        // Heap effects aren't currently implemented, so we do the ultra-conservative thing: if
        // anything in `range` has a possible heap effect, we return `true`.
        for (_, inst) in self.insts_iter(range) {
            if let Inst::Call(_) | Inst::Load(_) | Inst::Store(_) = inst {
                return true;
            }
        }
        false
    }

    /// If `iidx` is a constant Pointer, return a reference to that pointer or `None` otherwise.
    pub(super) fn inst_ptr<Reg: RegT>(&self, _m: &Mod<Reg>, iidx: InstIdx) -> Option<usize> {
        if let Inst::Const(Const {
            kind: ConstKind::Ptr(x),
            ..
        }) = self.inst(iidx)
        {
            Some(*x)
        } else {
            None
        }
    }

    /// Return the bit width of the instruction `iidx`. This is a convenience function over other
    /// public functions.
    pub(super) fn inst_ty<'a, Reg: RegT>(&'a self, m: &'a Mod<Reg>, iidx: InstIdx) -> &'a Ty {
        self.inst(iidx).ty(m)
    }

    pub(super) fn to_string<Reg: RegT>(&self, m: &Mod<Reg>) -> String {
        let mut out = Vec::with_capacity(self.insts.len());
        for (iidx, inst) in self.insts.iter_enumerated() {
            let ty = inst.ty(m);
            if ty == &Ty::Void {
                out.push(inst.to_string(m, self));
            } else {
                out.push(format!(
                    "%{}: {} = {}",
                    usize::from(iidx),
                    ty.to_string(m),
                    inst.to_string(m, self)
                ));
            }
        }
        out.join("\n")
    }
}

impl BlockLikeT for Block {
    fn inst(&self, idx: InstIdx) -> &Inst {
        &self.insts[usize::from(idx)]
    }
}

/// A HIR type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum Ty {
    // As in LLVM IR: a 64-bit floating-point value (IEEE-754 binary64).
    Double,
    // As in LLVM IR: a 32-bit floating-point value (IEEE-754 binary32).
    Float,
    /// A function type.
    ///
    /// Because these are rather big, and also used fairly rarely, we defer the details to a `Box`
    /// so that the common case has little memory overhead.
    Func(Box<FuncTy>),
    // An integer `u32` bits wide, where `u > 0 && u <= 24`.
    Int(u32),
    /// A pointer in an address space. LLVM allows 24 bits to be used.
    Ptr(u32),
    Void,
}

impl Ty {
    /// Return the bit width of a type.
    pub(super) fn bitw(&self) -> u32 {
        match self {
            Ty::Double => 64,
            Ty::Float => 32,
            Ty::Func(_func_ty) => todo!(),
            Ty::Int(bitw) => *bitw,
            Ty::Ptr(addrspace) => {
                assert_eq!(*addrspace, 0);
                #[cfg(target_arch = "x86_64")]
                64
            }
            Ty::Void => todo!(),
        }
    }

    pub(super) fn to_string<Reg: RegT>(&self, _m: &Mod<Reg>) -> String {
        match self {
            Ty::Double => "double".to_string(),
            Ty::Float => "float".to_string(),
            Ty::Func(_func_ty) => todo!(),
            Ty::Int(bitw) => format!("i{bitw}"),
            Ty::Ptr(addrspace) => {
                if *addrspace == 0 {
                    "ptr".to_string()
                } else {
                    todo!()
                }
            }
            Ty::Void => "void".to_string(),
        }
    }
}

/// A function type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct FuncTy {
    /// The type of each argument.
    pub args_tyidxs: SmallVec<[TyIdx; 4]>,
    /// True if this function's final argument is varargs.
    pub has_varargs: bool,
    /// The function's return type.
    pub rtn_tyidx: TyIdx,
}

index_vec::define_index_type! {
    pub(super) struct BlockIdx = u16;
}

index_vec::define_index_type! {
    /// The index type for extra information about guards that doesn't fit into a [Guard] object.
    pub(super) struct GuardExtraIdx = u16;
}

impl GuardExtraIdx {
    /// The maximum representable [GuardExtraIdx].
    pub(super) const MAX: GuardExtraIdx = GuardExtraIdx::from_raw_unchecked(u16::MAX);
}

// Note: if you change the `u32` here, `MAX` must also be updated.
index_vec::define_index_type! {
    pub(super) struct InstIdx = u32;
}

impl InstIdx {
    /// The maximum representable [InstIdx].
    pub(super) const MAX: InstIdx = InstIdx::from_raw_unchecked(u32::MAX);
}

index_vec::define_index_type! {
    pub(super) struct TyIdx = u16;
}

index_vec::define_index_type! {
    pub(super) struct ThreadLocalIdx = u16;
}

/// The trait that HIR instructions must conform to.
#[enum_dispatch]
pub(super) trait InstT: std::fmt::Debug {
    /// This might be unused in some configurations.
    #[allow(unused)]
    fn assert_well_formed(&self, _m: &dyn ModLikeT, _b: &dyn BlockLikeT, _iidx: InstIdx) {}

    /// Canonicalise this instruction. This rewrites operands to their most recent equivalent
    /// [InstIdx]s and performs other, basic, canonicalisation on a per-instruction kind basis.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, _opt: &mut T);

    /// For the purposes of common subexpression elimination is `other` equivalent to `self`?
    /// `other` must be canonicalised for this comparison to return true in all cases where
    /// it should.
    ///
    /// This should use [OptT::equiv_iidx] for [InstIdx]s. Other attributes should be treated on a
    /// case-by-case basis.
    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool;

    /// Produce each of this instruction's operands: note no order is guaranteed.
    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a>;

    /// Apply the function `iidx_map` to each of this instruction's operands, mutating `self` with
    /// the result.
    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, m: &mut dyn ModLikeT, iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx;

    /// Return a pretty printed version of `self`.
    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, m: &M, b: &B) -> String;

    /// Return the [Ty] of `self`.
    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty;
}

#[enum_dispatch(InstT)]
#[derive(Clone, Debug, EnumCount, EnumDiscriminants)]
pub(super) enum Inst {
    Abs,
    Add,
    And,
    Arg,
    AShr,
    #[cfg(test)]
    BlackBox,
    Call,
    Const,
    CtPop,
    DynPtrAdd,
    FAdd,
    FCmp,
    FDiv,
    Floor,
    FMul,
    FNeg,
    FSub,
    FPExt,
    FPToSI,
    Guard,
    ICmp,
    IntToPtr,
    Load,
    LShr,
    MemCpy,
    MemSet,
    Mul,
    Or,
    PtrAdd,
    PtrToInt,
    SDiv,
    Select,
    SExt,
    Shl,
    SIToFP,
    SMax,
    SMin,
    SRem,
    Store,
    Sub,
    Term,
    ThreadLocal,
    Trunc,
    UDiv,
    UIToFP,
    Xor,
    ZExt,
}

/// Abs with the same semantics as `llvm.abs.`.
#[derive(Clone, Debug)]
pub(super) struct Abs {
    pub tyidx: TyIdx,
    pub val: InstIdx,
    /// What LLVM calls `is_int_min_poison`
    pub int_min_poison: bool,
}

impl InstT for Abs {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            m.ty(self.tyidx).bitw(),
            b.inst_bitw(m, self.val),
            "%{iidx:?}: inconsistent bit widths for return type and val"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Abs(Abs {
            tyidx,
            val,
            int_min_poison,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
            && self.int_min_poison == *int_min_poison
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "abs %{}{}",
            usize::from(self.val),
            if self.int_min_poison {
                ", int_min_poison"
            } else {
                ""
            }
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// `+` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct Add {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub nuw: bool,
    pub nsw: bool,
}

impl InstT for Add {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Add(Add {
            tyidx,
            lhs,
            rhs,
            nuw,
            nsw,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.nuw == *nuw
            && self.nsw == *nsw
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("add %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// `&` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct And {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for And {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::And(And { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("and %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// An instruction representing argument *n* passed to a [Block]. In a sense, this is a
/// pseudo-instruction: it doesn't "do" anything directly, but it does (a) provide a consistent way
/// of viewing block arguments and it gives each of them a HIR type.
#[derive(Clone, Debug)]
pub(super) struct Arg {
    pub tyidx: TyIdx,
}

impl InstT for Arg {
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, _opt: &mut T) {}

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::none(m)
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut _iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        "arg".to_string()
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Arithmetic shift right with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct AShr {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub exact: bool,
}

impl InstT for AShr {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::AShr(AShr {
            tyidx,
            lhs,
            rhs,
            exact,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.exact == *exact
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "ashr %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Keep a value alive. Black boxes cannot be optimised away, and thus the value they refer to
/// cannot be optimised away. This is only used for testing purposes.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(super) struct BlackBox {
    pub val: InstIdx,
}

#[cfg(test)]
impl InstT for BlackBox {
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("blackbox %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Void
    }
}

/// `call` of a known function with the semantics of LLVM calls where the follow LLVM
/// attributes are implicitly set/unset:
///   1.`tail` and `musttail` are false (i.e. not a tail call),
///   2. `fast-math` is false,
///   3. `cconv` is false,
///   4. `zeroext`, `signext`, `noext`, and `inreg` are false,
///   5. addrspace is 0,
///   6. no function attributes,
///   7. no operand bundles.
#[derive(Clone, Debug)]
pub(super) struct Call {
    pub tgt: InstIdx,
    pub func_tyidx: TyIdx,
    pub args: SmallVec<[InstIdx; 1]>,
}

impl InstT for Call {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.tgt).ty(m) == &Ty::Ptr(0),
            "%{iidx:?}: call target is not a pointer"
        );

        let fty = m.func_ty(self.func_tyidx);
        if self.args.len() < fty.args_tyidxs.len() {
            panic!("%{iidx:?}: call has too few arguments");
        }

        if self.args.len() > fty.args_tyidxs.len() && !fty.has_varargs {
            panic!("%{iidx:?}: call has too many arguments");
        }

        for (i, iidx) in self.args.iter().enumerate().take(fty.args_tyidxs.len()) {
            assert_eq!(
                b.inst(*iidx).ty(m),
                m.ty(fty.args_tyidxs[i]),
                "%{iidx:?}: argument {i} has wrong type"
            )
        }
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.tgt = opt.equiv_iidx(self.tgt);
        for x in self.args.iter_mut() {
            *x = opt.equiv_iidx(*x);
        }
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::with_kind(m, IterIidxsIteratorKind::Call(self))
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.tgt = iidx_map(self.tgt);
        for x in self.args.iter_mut() {
            *x = iidx_map(*x);
        }
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, m: &M, b: &B) -> String {
        let fname = if let Inst::Const(Const {
            kind: ConstKind::Ptr(addr),
            ..
        }) = b.inst(self.tgt)
            && let Some(n) = m.addr_to_name(*addr)
        {
            format!(" ; @{n}")
        } else {
            "".to_owned()
        };
        format!(
            "call %{}({}){fname}",
            usize::from(self.tgt),
            self.args
                .iter()
                .map(|x| format!("%{}", usize::from(*x)))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(m.func_ty(self.func_tyidx).rtn_tyidx)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Const {
    pub tyidx: TyIdx,
    pub kind: ConstKind,
}

impl InstT for Const {
    fn assert_well_formed(&self, m: &dyn ModLikeT, _b: &dyn BlockLikeT, iidx: InstIdx) {
        let tyidx_bitw = m.ty(self.tyidx).bitw();
        let t = match &self.kind {
            ConstKind::Double(_) => {
                if tyidx_bitw == 64 {
                    None
                } else {
                    Some(64)
                }
            }
            ConstKind::Float(_) => {
                if tyidx_bitw == 32 {
                    None
                } else {
                    Some(32)
                }
            }
            ConstKind::Int(x) => {
                if tyidx_bitw == x.bitw() {
                    None
                } else {
                    Some(x.bitw())
                }
            }
            ConstKind::Ptr(_) => {
                if tyidx_bitw == usize::BITS {
                    None
                } else {
                    Some(usize::BITS)
                }
            }
        };
        if let Some(is_bits) = t {
            panic!("%{iidx:?}: type is {tyidx_bitw} bits but the constant is {is_bits} bits");
        }
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, _opt: &mut T) {}

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Const(Const { kind, .. }) = other
            && &self.kind == kind
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::none(m)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut _iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, m: &M, _b: &B) -> String {
        match &self.kind {
            ConstKind::Double(x) => x.to_string(),
            ConstKind::Float(x) => x.to_string(),
            ConstKind::Int(x) => x.to_string(),
            ConstKind::Ptr(x) => {
                if let Some(n) = m.addr_to_name(*x) {
                    format!("0x{x:X} ; @{n}")
                } else {
                    format!("0x{x:X}")
                }
            }
        }
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ConstKind {
    Double(f64),
    Float(f32),
    Int(ArbBitInt),
    Ptr(usize),
}

/// Count with the number of set bits, with the same semantics as `llvm.ctpop.`.
#[derive(Clone, Debug)]
pub(super) struct CtPop {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for CtPop {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            m.ty(self.tyidx).bitw(),
            b.inst_bitw(m, self.val),
            "%{iidx:?}: inconsistent bit widths for return type and val"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val)
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::CtPop(CtPop { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("ctpop %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

#[derive(Clone, Debug)]
pub(super) struct DynPtrAdd {
    pub ptr: InstIdx,
    pub num_elems: InstIdx,
    pub elem_size: u32,
}

impl InstT for DynPtrAdd {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.ptr).ty(m) == &Ty::Ptr(0),
            "%{iidx:?}: pointer is not a ptr type"
        );

        assert_matches!(
            b.inst(self.num_elems).ty(m),
            Ty::Int(_bitw),
            "%{iidx:?}: num_elems is not an integer type"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.ptr = opt.equiv_iidx(self.ptr);
        self.num_elems = opt.equiv_iidx(self.num_elems);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::DynPtrAdd(DynPtrAdd {
            ptr,
            num_elems,
            elem_size,
        }) = other
            && opt.equiv_iidx(self.ptr) == *ptr
            && opt.equiv_iidx(self.num_elems) == *num_elems
            && self.elem_size == *elem_size
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.ptr, self.num_elems)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.ptr = iidx_map(self.ptr);
        self.num_elems = iidx_map(self.num_elems);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "dynptradd %{}, %{}, {}",
            usize::from(self.ptr),
            usize::from(self.num_elems),
            self.elem_size
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Ptr(0)
    }
}

/// Floating point `+` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct FAdd {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for FAdd {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FAdd(FAdd { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "fadd %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Floating point comparison, with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct FCmp {
    /// What LLVM calls `cond`.
    pub pred: FPred,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for FCmp {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            b.inst(self.lhs).ty(m),
            b.inst(self.rhs).ty(m),
            "%{iidx:?}: inconsistent lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FCmp(FCmp { pred, lhs, rhs }) = other
            && self.pred == *pred
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "fcmp {} %{}, %{}",
            self.pred.to_str(),
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Int(1)
    }
}

/// Floating point comparison predicate with the same semantics as their LLVM IR equivalents.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum FPred {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

impl FPred {
    fn to_str(self) -> &'static str {
        match self {
            FPred::False => "false",
            FPred::Oeq => "oeq",
            FPred::Ogt => "ogt",
            FPred::Oge => "oge",
            FPred::Olt => "olt",
            FPred::Ole => "ole",
            FPred::One => "one",
            FPred::Ord => "ord",
            FPred::Ueq => "ueq",
            FPred::Ugt => "ugt",
            FPred::Uge => "uge",
            FPred::Ult => "ult",
            FPred::Ule => "ule",
            FPred::Une => "une",
            FPred::Uno => "uno",
            FPred::True => "true",
        }
    }
}

/// Floating point `/` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct FDiv {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for FDiv {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FDiv(FDiv { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "fdiv %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Floor with the same semantics as `llvm.floor.`.
#[derive(Clone, Debug)]
pub(super) struct Floor {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for Floor {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            m.ty(self.tyidx),
            b.inst(self.val).ty(m),
            "%{iidx:?}: inconsistent types for return type and val"
        );

        assert_matches!(
            m.ty(self.tyidx),
            Ty::Float | Ty::Double,
            "%{iidx:?}: float / double required"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Floor(Floor { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("floor %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Floating point `*` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct FMul {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for FMul {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FMul(FMul { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "fmul %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Floating point negation with the same semantics as LLVM's `fneg`.
#[derive(Clone, Debug)]
pub(super) struct FNeg {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub val: InstIdx,
}

impl InstT for FNeg {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            *m.ty(self.tyidx),
            *b.inst(self.val).ty(m),
            "%{iidx:?}: inconsistent return / val types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FNeg(FNeg { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("fneg %{}", usize::from(self.val),)
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Floating point `-` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct FSub {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for FSub {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FSub(FSub { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "fsub %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Cast from a smaller to a larger floating point type with the same semantics as LLVM's `fpext`.
#[derive(Clone, Debug)]
pub(super) struct FPExt {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for FPExt {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        // Right now, we can only possibly go from Float -> Double with `fpext`. If we add other
        // floating point types, that will change.
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Double,
            "%{iidx:?}: return type is not a floating point type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Float,
            "%{iidx:?}: val is not an integer"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FPExt(FPExt { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("fpext %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Cast a floating point number to a signed integer with the same semantics as LLVM's `fptosi`.
#[derive(Clone, Debug)]
pub(super) struct FPToSI {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for FPToSI {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Double | Ty::Float,
            "%{iidx:?}: return type is not a floating point type"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::FPToSI(FPToSI { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("fptosi %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// A guard that the value produced by `cond` is `expect_true`. If not, the remainder of the
/// trace is invalid for this execution.
#[derive(Clone, Debug)]
pub(super) struct Guard {
    pub expect: bool,
    pub cond: InstIdx,
    /// The [Guardextra] that this guard maps to. Before optimisation, this will be set to
    /// [GuardExtra::MAX].
    pub geidx: GuardExtraIdx,
}

impl InstT for Guard {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            b.inst(self.cond).ty(m),
            &Ty::Int(1),
            "%{iidx:?}: guard references a non-i1 for its condition"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.cond = opt.equiv_iidx(self.cond);
        // To avoid allocating, we swap in an empty vec, mutate what we've taken out, and put it
        // back in just below.
        let mut entry_vars = std::mem::take(&mut opt.gextra_mut(self.geidx).entry_vars);
        for x in entry_vars.iter_mut() {
            *x = opt.equiv_iidx(*x);
        }
        opt.gextra_mut(self.geidx).entry_vars = entry_vars;
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::with_kind(m, IterIidxsIteratorKind::Guard(self))
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.cond = iidx_map(self.cond);
        for x in m.gextra_mut(self.geidx).entry_vars.iter_mut() {
            *x = iidx_map(*x);
        }
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, m: &M, _b: &B) -> String {
        format!(
            "guard {}, %{}, [{}]",
            if self.expect { "true" } else { "false" },
            usize::from(self.cond),
            m.gextra(self.geidx)
                .entry_vars
                .iter()
                .map(|iidx| format!("%{}", usize::from(*iidx)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Void
    }
}

/// Extra information for guard instructions that is too big to fit into [Guard].
#[derive(Clone, Debug)]
pub(super) struct GuardExtra {
    pub bid: aot_ir::BBlockId,
    /// If this guard:
    ///
    ///   1. is the first guard in a trace,
    ///   2. relates to an LLVM-level `switch`,
    ///
    /// then this records the information necessary for subsequent sidetraces to deal with the
    /// switch properly.
    pub switch: Option<Switch>,
    /// The variables used on entry to the guard. These are stored as an ordered, flat, sequence,
    /// corresponding to the sequence of `exit_frames`. For example, if `exit_frames` has 2 frames,
    /// the first of which needs 3 variables and the second 1 variable entry_vars would look as
    /// follows:
    /// ```text
    /// [a, b, c, d]
    ///  ^^^^^^^
    ///     |     ^
    ///     |   exit_frames[1] variable
    ///     |
    ///  exit_frames[0] variables
    /// ```
    pub entry_vars: Vec<InstIdx>,
    /// The frames needed for deopt and side-tracing with the most recent frame at the tail-end of
    /// this list. This is a 1:1 mapping with the call frames at the point of the respective guard
    /// *except* that the most recent call frame is replaced with the deopt information for the
    /// branch (etc) that failed. Brief experiments suggest that, depending on the benchmark and
    /// interpreter, the depth of frames decreases exponentially: in ~50-90% (and 90% is more
    /// common than 50%) of cases there is 1 frame, about 10x fewer have 2 frames, and so on.
    pub exit_frames: SmallVec<[Frame; 2]>,
}

/// If a guard relates to an AOT `switch`, this struct records the extra information we need to
/// process it correctly.
#[derive(Clone, Debug)]
pub(super) struct Switch {
    /// The [InstId] of the AOT switch instruction.
    pub iid: aot_ir::InstId,
    /// The destination blocks we've seen so far in the chain of `switch` cases. Recording this
    /// allows us to optimise away part, or all, of guard checks in later sidetraces.
    pub seen_bbidxs: Vec<aot_ir::BBlockIdx>,
}

/// Integer comparison, with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct ICmp {
    /// What LLVM calls `cond`.
    pub pred: IPred,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub samesign: bool,
}

impl InstT for ICmp {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            b.inst(self.lhs).ty(m),
            b.inst(self.rhs).ty(m),
            "%{iidx:?}: inconsistent lhs / rhs types"
        );
    }

    /// For [IPred::Eq] and [IPred::Ne], canonicalise to favour references to constants on the RHS of
    /// the addition.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if (self.pred == IPred::Eq || self.pred == IPred::Ne)
            && matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::ICmp(ICmp {
            pred,
            lhs,
            rhs,
            samesign,
        }) = other
            && self.pred == *pred
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.samesign == *samesign
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "icmp {} %{}, %{}",
            self.pred.as_str(),
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Int(1)
    }
}

/// Integer comparison predicate with the same semantics as their LLVM IR equivalents.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum IPred {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

impl IPred {
    /// Does this predicate do signed comparison (i.e. requiring sign extension of arguments).
    pub(super) fn is_signed(&self) -> bool {
        match self {
            IPred::Eq | IPred::Ne | IPred::Ugt | IPred::Uge | IPred::Ult | IPred::Ule => false,
            IPred::Sgt | IPred::Sge | IPred::Slt | IPred::Sle => true,
        }
    }

    fn as_str(&self) -> &str {
        match self {
            IPred::Eq => "eq",
            IPred::Ne => "ne",
            IPred::Ugt => "ugt",
            IPred::Uge => "uge",
            IPred::Ult => "ult",
            IPred::Ule => "ule",
            IPred::Sgt => "sgt",
            IPred::Sge => "sge",
            IPred::Slt => "slt",
            IPred::Sle => "sle",
        }
    }
}

/// Cast a pointer to an int with the same semantics as LLVM's `ptrtoint`.
#[derive(Clone, Debug)]
pub(super) struct IntToPtr {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for IntToPtr {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Ptr(_),
            "%{iidx:?}: return type is not a pointer"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer type"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::IntToPtr(IntToPtr { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("inttoptr %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Load {
    pub tyidx: TyIdx,
    pub ptr: InstIdx,
    pub is_volatile: bool,
}

impl InstT for Load {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            b.inst(self.ptr).ty(m),
            &Ty::Ptr(_),
            "%{iidx:?}: ptr is not a pointer"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.ptr = opt.equiv_iidx(self.ptr);
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.ptr)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.ptr = iidx_map(self.ptr);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("load %{}", usize::from(self.ptr))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Logical shift right with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct LShr {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub exact: bool,
}

impl InstT for LShr {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::LShr(LShr {
            tyidx,
            lhs,
            rhs,
            exact,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.exact == *exact
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "lshr %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// `memcpy` with the same semantics as LLVM's `llvm.memcpy`.
#[derive(Clone, Debug)]
pub(super) struct MemCpy {
    pub dst: InstIdx,
    pub src: InstIdx,
    pub len: InstIdx,
    pub volatile: bool,
}

impl InstT for MemCpy {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.dst).ty(m) == b.inst(self.src).ty(m),
            "%{iidx:?}: inconsistent dst / src types"
        );
        assert_matches!(
            b.inst(self.len).ty(m),
            &Ty::Int(32 | 64),
            "%{iidx:?}: len has wrong type"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.dst = opt.equiv_iidx(self.dst);
        self.src = opt.equiv_iidx(self.src);
        self.len = opt.equiv_iidx(self.len);
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::three(m, self.dst, self.src, self.len)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.dst = iidx_map(self.dst);
        self.src = iidx_map(self.src);
        self.len = iidx_map(self.len);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "memcpy %{}, %{}, %{}, {}",
            usize::from(self.dst),
            usize::from(self.src),
            usize::from(self.len),
            if self.volatile { "true" } else { "false" }
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Void
    }
}

/// `memcpy` with the same semantics as LLVM's `llvm.memcpy`.
#[derive(Clone, Debug)]
pub(super) struct MemSet {
    pub dst: InstIdx,
    pub val: InstIdx,
    pub len: InstIdx,
    pub volatile: bool,
}

impl InstT for MemSet {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            b.inst(self.dst).ty(m),
            &Ty::Ptr(_),
            "%{iidx:?}: dst has wrong type"
        );
        assert_matches!(
            b.inst(self.val).ty(m),
            &Ty::Int(8),
            "%{iidx:?}: val has wrong type"
        );
        assert_matches!(
            b.inst(self.len).ty(m),
            &Ty::Int(32 | 64),
            "%{iidx:?}: len has wrong type"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.dst = opt.equiv_iidx(self.dst);
        self.val = opt.equiv_iidx(self.val);
        self.len = opt.equiv_iidx(self.len);
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::three(m, self.dst, self.val, self.len)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.dst = iidx_map(self.dst);
        self.val = iidx_map(self.val);
        self.len = iidx_map(self.len);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "memset %{}, %{}, %{}, {}",
            usize::from(self.dst),
            usize::from(self.val),
            usize::from(self.len),
            if self.volatile { "true" } else { "false" }
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Void
    }
}

/// `*` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct Mul {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub nuw: bool,
    pub nsw: bool,
}

impl InstT for Mul {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Mul(Mul {
            tyidx,
            lhs,
            rhs,
            nuw,
            nsw,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.nuw == *nuw
            && self.nsw == *nsw
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("mul %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// `|` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct Or {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub disjoint: bool,
}

impl InstT for Or {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Or(Or {
            tyidx,
            lhs,
            rhs,
            disjoint,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.disjoint == *disjoint
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("or %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

#[derive(Clone, Debug)]
pub(super) struct PtrAdd {
    pub ptr: InstIdx,
    pub off: i32,
    pub in_bounds: bool,
    pub nusw: bool,
    pub nuw: bool,
}

impl InstT for PtrAdd {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.ptr).ty(m) == &Ty::Ptr(0),
            "%{iidx:?}: pointer is not a ptr type"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.ptr = opt.equiv_iidx(self.ptr);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::PtrAdd(PtrAdd {
            ptr,
            off,
            in_bounds,
            nusw,
            nuw,
        }) = other
            && opt.equiv_iidx(self.ptr) == *ptr
            && self.off == *off
            && self.in_bounds == *in_bounds
            && self.nusw == *nusw
            && self.nuw == *nuw
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.ptr)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.ptr = iidx_map(self.ptr);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("ptradd %{}, {}", usize::from(self.ptr), self.off)
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Ptr(0)
    }
}

/// Cast a pointer to an int with the same semantics as LLVM's `ptrtoint`.
#[derive(Clone, Debug)]
pub(super) struct PtrToInt {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for PtrToInt {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Int(_),
            "%{iidx:?}: return type is not an integer type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Ptr(_),
            "%{iidx:?}: val is not a pointer"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::PtrToInt(PtrToInt { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("ptrtoint %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Return the quotient from signed division with the same semantics as LLVM's `sdiv`.
#[derive(Clone, Debug)]
pub(super) struct SDiv {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub exact: bool,
}

impl InstT for SDiv {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::SDiv(SDiv {
            tyidx,
            lhs,
            rhs,
            exact,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.exact == *exact
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "sdiv %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

#[derive(Clone, Debug)]
pub(super) struct Select {
    pub tyidx: TyIdx,
    pub cond: InstIdx,
    pub truev: InstIdx,
    pub falsev: InstIdx,
}

impl InstT for Select {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_eq!(
            b.inst(self.cond).ty(m),
            &Ty::Int(1),
            "%{iidx:?}: select references a non-i1 for its condition"
        );

        assert!(
            b.inst(self.truev).ty(m) == b.inst(self.falsev).ty(m)
                && b.inst(self.falsev).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / truev / falsev types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.cond = opt.equiv_iidx(self.cond);
        self.truev = opt.equiv_iidx(self.truev);
        self.falsev = opt.equiv_iidx(self.falsev);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Select(Select {
            tyidx,
            cond,
            truev,
            falsev,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.cond) == *cond
            && opt.equiv_iidx(self.truev) == *truev
            && opt.equiv_iidx(self.falsev) == *falsev
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::three(m, self.cond, self.truev, self.falsev)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.cond = iidx_map(self.cond);
        self.truev = iidx_map(self.truev);
        self.falsev = iidx_map(self.falsev);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "select %{}, %{}, %{}",
            usize::from(self.cond),
            usize::from(self.truev),
            usize::from(self.falsev)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Sign extend with the same semantics as LLVM's `sext`.
#[derive(Clone, Debug)]
pub(super) struct SExt {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for SExt {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Int(_),
            "%{iidx:?}: return type is not an integer type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer type"
        );

        assert!(
            b.inst_bitw(m, self.val) < m.ty(self.tyidx).bitw(),
            "%{iidx:?}: val bitw must be strictly less than the return type bitw"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::SExt(SExt { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("sext %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Shift left with the same semantics as LLVM's `shl`.
#[derive(Clone, Debug)]
pub(super) struct Shl {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub nuw: bool,
    pub nsw: bool,
}

impl InstT for Shl {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Shl(Shl {
            tyidx,
            lhs,
            rhs,
            nuw,
            nsw,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.nuw == *nuw
            && self.nsw == *nsw
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("shl %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Cast a signed integer to floating point with the same semantics as LLVM's `sitofp`.
#[derive(Clone, Debug)]
pub(super) struct SIToFP {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for SIToFP {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Double | Ty::Float,
            "%{iidx:?}: return type is not a floating point type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::SIToFP(SIToFP { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("sitofp %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Return the signed maximum of two values, with the same semantics as LLVM's `llvm.smax`.
#[derive(Clone, Debug)]
pub(super) struct SMax {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for SMax {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::SMax(SMax { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "smax %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Return the signed maximum of two values, with the same semantics as LLVM's `llvm.smin`.
#[derive(Clone, Debug)]
pub(super) struct SMin {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for SMin {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::SMin(SMin { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "smin %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Return the remainder from signed division with the same semantics as LLVM's `srem`.
#[derive(Clone, Debug)]
pub(super) struct SRem {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for SRem {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::SRem(SRem { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "srem %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Store `val` into `ptr` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct Store {
    pub val: InstIdx,
    pub ptr: InstIdx,
    #[allow(unused)]
    pub is_volatile: bool,
}

impl InstT for Store {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            b.inst(self.ptr).ty(m),
            &Ty::Ptr(_),
            "%{iidx:?}: ptr is not a pointer"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
        self.ptr = opt.equiv_iidx(self.ptr);
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.val, self.ptr)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
        self.ptr = iidx_map(self.ptr);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "store %{}, %{}",
            usize::from(self.val),
            usize::from(self.ptr)
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Void
    }
}

/// `-` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct Sub {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub nuw: bool,
    pub nsw: bool,
}

impl InstT for Sub {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Sub(Sub {
            tyidx,
            lhs,
            rhs,
            nuw,
            nsw,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.nuw == *nuw
            && self.nsw == *nsw
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("sub %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// A block terminator.
#[derive(Clone, Debug)]
pub(super) struct Term(pub(super) Vec<InstIdx>);

impl InstT for Term {
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        for x in self.0.iter_mut() {
            *x = opt.equiv_iidx(*x);
        }
    }

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, _other: &Inst) -> bool {
        panic!();
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::with_kind(m, IterIidxsIteratorKind::Term(self))
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "term [{}]",
            self.0
                .iter()
                .map(|x| format!("%{}", usize::from(*x)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        for x in self.0.iter_mut() {
            *x = iidx_map(*x);
        }
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        // `Term` is a pseudo-instruction, but it makes various things simpler if we pretend it has
        // a type.
        &Ty::Void
    }
}

#[derive(Clone, Debug)]
pub(super) struct ThreadLocal(pub *const c_void);

impl InstT for ThreadLocal {
    fn assert_well_formed(&self, _m: &dyn ModLikeT, _b: &dyn BlockLikeT, _iidx: InstIdx) {}

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, _opt: &mut T) {}

    fn cse_eq(&self, _opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::ThreadLocal(ThreadLocal(x)) = other
            && self.0 == *x
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::none(m)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut _iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, m: &M, _b: &B) -> String {
        format!(
            "threadlocal {}",
            m.addr_to_name(self.0.addr()).unwrap_or("<erased>")
        )
    }

    fn ty<'a>(&'a self, _m: &'a dyn ModLikeT) -> &'a Ty {
        &Ty::Ptr(0)
    }
}

/// Truncate with the same semantics as LLVM's `trunc`.
#[derive(Clone, Debug)]
pub(super) struct Trunc {
    pub tyidx: TyIdx,
    pub val: InstIdx,
    pub nuw: bool,
    pub nsw: bool,
}

impl InstT for Trunc {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Int(_),
            "%{iidx:?}: return type is not an integer type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer type"
        );

        assert!(
            b.inst_bitw(m, self.val) > m.ty(self.tyidx).bitw(),
            "%{iidx:?}: val bitw must be strictly greater than the return type bitw"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Trunc(Trunc {
            tyidx,
            val,
            nuw,
            nsw,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
            && self.nuw == *nuw
            && self.nsw == *nsw
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("trunc %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Unsigned integer division with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct UDiv {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
    pub exact: bool,
}

impl InstT for UDiv {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::UDiv(UDiv {
            tyidx,
            lhs,
            rhs,
            exact,
        }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
            && self.exact == *exact
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!(
            "udiv %{}, %{}",
            usize::from(self.lhs),
            usize::from(self.rhs)
        )
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Cast an unsigned integer to floating point with the same semantics as LLVM's `uitofp`.
#[derive(Clone, Debug)]
pub(super) struct UIToFP {
    pub tyidx: TyIdx,
    pub val: InstIdx,
    pub nneg: bool,
}

impl InstT for UIToFP {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Double | Ty::Float,
            "%{iidx:?}: return type is not a floating point type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::UIToFP(UIToFP { tyidx, val, nneg }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
            && self.nneg == *nneg
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("uitofp %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// `^` with normal LLVM semantics.
#[derive(Clone, Debug)]
pub(super) struct Xor {
    pub tyidx: TyIdx,
    /// What LLVM calls `op1`.
    pub lhs: InstIdx,
    /// What LLVM calls `op2`.
    pub rhs: InstIdx,
}

impl InstT for Xor {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert!(
            b.inst(self.lhs).ty(m) == b.inst(self.rhs).ty(m)
                && b.inst(self.rhs).ty(m) == m.ty(self.tyidx),
            "%{iidx:?}: inconsistent return / lhs / rhs types"
        );
    }

    /// Canonicalise to favour references to constants on the RHS.
    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.lhs = opt.equiv_iidx(self.lhs);
        self.rhs = opt.equiv_iidx(self.rhs);
        if matches!(opt.inst(self.lhs), Inst::Const(_))
            && !matches!(opt.inst(self.rhs), Inst::Const(_))
        {
            std::mem::swap(&mut self.lhs, &mut self.rhs);
        }
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::Xor(Xor { tyidx, lhs, rhs }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.lhs) == *lhs
            && opt.equiv_iidx(self.rhs) == *rhs
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::two(m, self.lhs, self.rhs)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.lhs = iidx_map(self.lhs);
        self.rhs = iidx_map(self.rhs);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("xor %{}, %{}", usize::from(self.lhs), usize::from(self.rhs))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// Zero extend with the same semantics as LLVM's `zext`.
#[derive(Clone, Debug)]
pub(super) struct ZExt {
    pub tyidx: TyIdx,
    pub val: InstIdx,
}

impl InstT for ZExt {
    fn assert_well_formed(&self, m: &dyn ModLikeT, b: &dyn BlockLikeT, iidx: InstIdx) {
        assert_matches!(
            m.ty(self.tyidx),
            Ty::Int(_),
            "%{iidx:?}: return type is not an integer type"
        );

        assert_matches!(
            b.inst(self.val).ty(m),
            Ty::Int(_),
            "%{iidx:?}: val is not an integer type"
        );

        assert!(
            b.inst_bitw(m, self.val) < m.ty(self.tyidx).bitw(),
            "%{iidx:?}: val bitw must be strictly less than the return type bitw"
        );
    }

    fn canonicalise<T: BlockLikeT + EquivIIdxT + ModLikeT>(&mut self, opt: &mut T) {
        self.val = opt.equiv_iidx(self.val);
    }

    fn cse_eq(&self, opt: &dyn EquivIIdxT, other: &Inst) -> bool {
        if let Inst::ZExt(ZExt { tyidx, val }) = other
            && self.tyidx == *tyidx
            && opt.equiv_iidx(self.val) == *val
        {
            true
        } else {
            false
        }
    }

    fn iter_iidxs<'a>(&'a self, m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        IterIidxsIterator::one(m, self.val)
    }

    #[cfg(test)]
    fn rewrite_iidxs<F>(&mut self, _m: &mut dyn ModLikeT, mut iidx_map: F)
    where
        F: FnMut(InstIdx) -> InstIdx,
    {
        self.val = iidx_map(self.val);
    }

    fn to_string<M: ModLikeT, B: BlockLikeT>(&self, _m: &M, _b: &B) -> String {
        format!("zext %{}", usize::from(self.val))
    }

    fn ty<'a>(&'a self, m: &'a dyn ModLikeT) -> &'a Ty {
        m.ty(self.tyidx)
    }
}

/// An iterator over a HIR instruction's [InstIdx]s.
pub(super) struct IterIidxsIterator<'a> {
    m: &'a dyn ModLikeT,
    kind: IterIidxsIteratorKind<'a>,
    /// The current offset of the iterator: this must be interpreted with respect to `Self::kind`.
    i: usize,
}

impl<'a> IterIidxsIterator<'a> {
    /// Create an [IterIidxsIterator] with an arbitrary [IterIidxsIteratorKind].
    fn with_kind(m: &'a dyn ModLikeT, kind: IterIidxsIteratorKind<'a>) -> IterIidxsIterator<'a> {
        IterIidxsIterator { m, kind, i: 0 }
    }

    /// Create an [IterIidxsIterator] with no [InstIdx]s.
    fn none(m: &'a dyn ModLikeT) -> IterIidxsIterator<'a> {
        Self::with_kind(m, IterIidxsIteratorKind::None)
    }

    /// Create an [IterIidxsIterator] with one [InstIdx]s.
    fn one(m: &'a dyn ModLikeT, iidx1: InstIdx) -> IterIidxsIterator<'a> {
        Self::with_kind(m, IterIidxsIteratorKind::One(iidx1))
    }

    /// Create an [IterIidxsIterator] with two [InstIdx]s.
    fn two(m: &'a dyn ModLikeT, iidx1: InstIdx, iidx2: InstIdx) -> IterIidxsIterator<'a> {
        Self::with_kind(m, IterIidxsIteratorKind::Two(iidx1, iidx2))
    }

    /// Create an [IterIidxsIterator] with three [InstIdx]s.
    fn three(
        m: &'a dyn ModLikeT,
        iidx1: InstIdx,
        iidx2: InstIdx,
        iidx3: InstIdx,
    ) -> IterIidxsIterator<'a> {
        Self::with_kind(m, IterIidxsIteratorKind::Three(iidx1, iidx2, iidx3))
    }
}

impl<'a> Iterator for IterIidxsIterator<'a> {
    type Item = InstIdx;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        match self.kind {
            IterIidxsIteratorKind::None => (),
            IterIidxsIteratorKind::One(iidx1) => {
                if self.i == 1 {
                    return Some(iidx1);
                }
            }
            IterIidxsIteratorKind::Two(iidx1, iidx2) => {
                if self.i == 1 {
                    return Some(iidx1);
                } else if self.i == 2 {
                    return Some(iidx2);
                }
            }
            IterIidxsIteratorKind::Three(iidx1, iidx2, iidx3) => {
                if self.i == 1 {
                    return Some(iidx1);
                } else if self.i == 2 {
                    return Some(iidx2);
                } else if self.i == 3 {
                    return Some(iidx3);
                }
            }
            IterIidxsIteratorKind::Call(Call {
                tgt,
                func_tyidx: _,
                args,
            }) => {
                if self.i == 1 {
                    return Some(*tgt);
                } else if self.i - 1 <= args.len() {
                    return Some(args[self.i - 2]);
                }
            }
            IterIidxsIteratorKind::Guard(Guard {
                expect: _,
                cond,
                geidx,
            }) => {
                if self.i == 1 {
                    return Some(*cond);
                } else {
                    let GuardExtra {
                        bid: _,
                        switch: _,
                        entry_vars,
                        exit_frames: _,
                    } = self.m.gextra(*geidx);
                    if self.i - 1 <= entry_vars.len() {
                        return Some(entry_vars[self.i - 2]);
                    }
                }
            }
            IterIidxsIteratorKind::Term(Term(vars)) => {
                if self.i <= vars.len() {
                    return Some(vars[self.i - 1]);
                }
            }
        }
        None
    }
}

/// The kind of an [IterIidxsIterator].
///
/// Note this `enum` is a trade-off. It would be more memory efficient to store variants for every
/// IR instruction kind. Most obviously doing so would bloat `IterIidxIterator::next`. Less
/// obviously, it would make it a little easier to introduce errors later: most IR elements have
/// 0..=3 [InstIdx]s, which is directly encoded in their `inst_iidxs` functions. In other words, if
///   someone edits one of those instructions, they (and the compiler) are more likely to realise
///   that their `inst_iidxs` function needs editing.
enum IterIidxsIteratorKind<'a> {
    None,
    One(InstIdx),
    Two(InstIdx, InstIdx),
    Three(InstIdx, InstIdx, InstIdx),
    Call(&'a Call),
    Guard(&'a Guard),
    Term(&'a Term),
}

/// Information about a guard necessary for deopt and side-tracing.
#[derive(Clone, Debug)]
pub(super) struct Frame {
    pub pc: aot_ir::InstId,
    pub pc_safepoint: &'static DeoptSafepoint,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::{
        hir_parser::str_to_mod,
        opt::{OptT, fullopt::FullOpt},
        regalloc::{TestRegIter, test::TestReg},
    };
    use strum::{Display, EnumCount};

    #[derive(Copy, Clone, Debug, Display, EnumCount, PartialEq)]
    enum DummyReg {
        R0,
    }

    impl RegT for DummyReg {
        type RegIdx = DummyRegIdx;
        const MAX_REGIDX: DummyRegIdx = DummyRegIdx::from_usize_unchecked(DummyReg::COUNT);

        fn undefined() -> Self {
            todo!()
        }

        fn from_regidx(_idx: Self::RegIdx) -> Self {
            todo!()
        }

        fn regidx(&self) -> Self::RegIdx {
            todo!()
        }

        fn is_caller_saved(&self) -> bool {
            todo!()
        }

        fn iter_test_regs() -> impl TestRegIter<Self> {
            DummyRegTestIter {}
        }

        fn from_str(_s: &str) -> Option<Self> {
            todo!();
        }
    }

    index_vec::define_index_type! {
        pub(crate) struct DummyRegIdx = u8;
    }

    struct DummyRegTestIter {}

    impl TestRegIter<DummyReg> for DummyRegTestIter {
        fn next_reg(&mut self, _: &Ty) -> Option<DummyReg> {
            Some(DummyReg::R0)
        }
    }

    #[test]
    fn inst_size() {
        // This is bigger than I suspect we're comfortable with, but optimising this is something
        // for the future when we better understand the real-world impact.
        assert_eq!(
            std::mem::size_of::<Inst>(),
            std::mem::size_of::<usize>() * 5
        );
    }

    #[test]
    fn iter_iidxs() {
        // The `None` case.
        let m = str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert!(
            block
                .inst(InstIdx::new(0))
                .iter_iidxs(&m)
                .collect::<Vec<_>>()
                .is_empty()
        );

        // The `One` case.
        let m = str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = zext %0
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert_eq!(
            block
                .inst(InstIdx::new(1))
                .iter_iidxs(&m)
                .collect::<Vec<_>>(),
            [InstIdx::new(0)]
        );

        // The `Two` case.
        let m = str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = add %0, %1
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert_eq!(
            block
                .inst(InstIdx::new(2))
                .iter_iidxs(&m)
                .collect::<Vec<_>>(),
            [InstIdx::new(0), InstIdx::new(1)]
        );

        // The `Three` case.
        let m = str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: ptr = arg [reg]
          %2: i64 = arg [reg]
          memcpy %0, %1, %2, true
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert_eq!(
            block
                .inst(InstIdx::new(3))
                .iter_iidxs(&m)
                .collect::<Vec<_>>(),
            [InstIdx::new(0), InstIdx::new(1), InstIdx::new(2)]
        );

        // The `Call` case.
        let m = str_to_mod::<DummyReg>(
            "
          extern f(i8)

          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          call f %0(%1)
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert_eq!(
            block
                .inst(InstIdx::new(2))
                .iter_iidxs(&m)
                .collect::<Vec<_>>(),
            [InstIdx::new(0), InstIdx::new(1),]
        );

        // The `Guard` case.
        let m = str_to_mod::<DummyReg>(
            "
          %0: i1 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = arg [reg]
          guard true, %0, [%1, %2]
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert_eq!(
            block
                .inst(InstIdx::new(3))
                .iter_iidxs(&m)
                .collect::<Vec<_>>(),
            [InstIdx::new(0), InstIdx::new(1), InstIdx::new(2)]
        );

        // The `Term` case.
        let m = str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          term [%0, %1]
        ",
        );
        let TraceEnd::Test { block, .. } = &m.trace_end else {
            panic!()
        };
        assert_eq!(
            block
                .inst(InstIdx::new(2))
                .iter_iidxs(&m)
                .collect::<Vec<_>>(),
            [InstIdx::new(0), InstIdx::new(1)]
        );
    }

    // Below are the well-formedness checks. We start with the generic well-formedness checks
    // before moving to the per-[Inst] checks.

    #[test]
    #[should_panic(expected = "%3: 'arg' instructions cannot appear after trace entry")]
    fn arg_cannot_appear_after_entry() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i16 = sext %0
          %3: i8 = arg [reg]
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%3: 'arg' instructions cannot appear after trace entry")]
    fn entry_instructions_are_arg_const() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i16 = sext %0
          %3: i8 = arg [reg]
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%0: forward reference to %1")]
    fn no_forward_references() {
        str_to_mod::<DummyReg>("%0: i8 = add %1, %2");
    }

    #[test]
    #[should_panic(expected = "Type 'Int(8)' appears more than once")]
    fn tys_appear_once() {
        let mut m = str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
        ",
        );
        m.tys.push(Ty::Int(8));
        m.assert_well_formed();
    }

    // The per-[Inst] checks.

    #[test]
    #[should_panic(expected = "%1: inconsistent bit widths")]
    fn abs_type_consistency() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = abs %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn add_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = add %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn add_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = add %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn and_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = and %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn and_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = and %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn ashr_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = ashr %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn ashr_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = ashr %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: call target is not a pointer")]
    fn call_must_to_be_pointer() {
        str_to_mod::<DummyReg>(
            "
          extern f();
          %0: i8 = arg [reg]
          %1: i16 = call f %0()
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: call has too few arguments")]
    fn call_too_few_args() {
        str_to_mod::<DummyReg>(
            "
          extern f(i32);
          %0: ptr = arg [reg]
          %1: i16 = call f %0()
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: call has too many arguments")]
    fn call_too_many_args() {
        str_to_mod::<DummyReg>(
            "
          extern f();
          %0: ptr = arg [reg]
          %1: i16 = call f %0(%0)
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: argument 0 has wrong type")]
    fn call_arg_has_wrong_type() {
        str_to_mod::<DummyReg>(
            "
          extern f(i32);
          %0: ptr = arg [reg]
          %1: i8 = 0
          call f %0(%1)
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: inconsistent bit widths")]
    fn ctpop_type_consistency() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = ctpop %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: pointer is not a ptr type")]
    fn dynptradd_not_ptr() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: ptr = dynptradd %0, %1, 1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: num_elems is not an integer type")]
    fn dynptradd_not_int() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: ptr = dynptradd %0, %0, 1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fadd_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = arg [reg]
          %2: float = fadd %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fadd_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: float = arg [reg]
          %2: double = fadd %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fdiv_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = arg [reg]
          %2: float = fdiv %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fdiv_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: float = arg [reg]
          %2: double = fdiv %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: inconsistent types for return type and val")]
    fn floor_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = floor %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: float / double required")]
    fn floor_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i32 = arg [reg]
          %1: i32 = floor %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fmul_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = arg [reg]
          %2: float = fmul %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fmul_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: float = arg [reg]
          %2: double = fmul %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: inconsistent return / val types")]
    fn fneg_type_consistency() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = fneg %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fsub_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = arg [reg]
          %2: float = fsub %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn fsub_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: float = arg [reg]
          %2: double = fsub %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: guard references a non-i1 for its condition")]
    fn guard_cond_must_be_i1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          guard true, %0, []
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent lhs / rhs types")]
    fn icmp_inconsistent_types() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i1 = icmp eq %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not an integer type")]
    fn inttoptr_must_take_int() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: ptr = inttoptr %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not a pointer")]
    fn inttoptr_must_return_pointer() {
        str_to_mod::<DummyReg>(
            "
          %0: i64 = arg [reg]
          %1: i64 = inttoptr %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: ptr is not a pointer")]
    fn load_must_take_pointer() {
        str_to_mod::<DummyReg>(
            "
          %0: i64 = arg [reg]
          %1: i8 = load %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn lshr_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = lshr %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn lshr_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = lshr %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%3: inconsistent dst / src types")]
    fn memcpy_type_consistency() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          %2: i64 = arg [reg]
          memcpy %0, %1, %2, true
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%3: len has wrong type")]
    fn memcpy_len_type() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: ptr = arg [reg]
          %2: i8 = arg [reg]
          memcpy %0, %1, %2, true
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%3: dst has wrong type")]
    fn memset_dst_type() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i32 = arg [reg]
          memset %0, %1, %2, true
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%3: val has wrong type")]
    fn memset_val_type() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: i16 = arg [reg]
          %2: i32 = arg [reg]
          memset %0, %1, %2, true
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%3: len has wrong type")]
    fn memset_len_type() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = arg [reg]
          memset %0, %1, %2, true
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn mul_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = mul %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn mul_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = mul %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn or_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = or %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn or_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = or %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: pointer is not a ptr type")]
    fn ptradd_not_ptr() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: ptr = ptradd %0, 1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not a pointer")]
    fn ptrtoint_must_take_pointer() {
        str_to_mod::<DummyReg>(
            "
          %0: i64 = arg [reg]
          %1: i64 = ptrtoint %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not an integer type")]
    fn ptrtoint_must_return_pointer() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: ptr = ptrtoint %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn sdiv_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = arg [reg]
          %2: float = sdiv %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn sdiv_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: float = arg [reg]
          %2: double = sdiv %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: select references a non-i1 for its condition")]
    fn select_cond_must_be_i1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i32 = arg [reg]
          %2: i32 = select %0, %1, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / truev / falsev types")]
    fn select_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i1 = arg [reg]
          %1: i32 = arg [reg]
          %2: i32 = select %0, %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not an integer type")]
    fn sext_must_take_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: i64 = sext %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not an integer type")]
    fn sext_must_return_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: i1 = arg [reg]
          %1: ptr = sext %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val bitw must be strictly less than the return type bitw")]
    fn sext_must_extend() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = sext %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn shl_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = shl %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn shl_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = shl %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not a floating point type")]
    fn sitofp_must_return_floating_point() {
        str_to_mod::<DummyReg>(
            "
          %0: i32 = arg [reg]
          %1: ptr = sitofp %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not an integer")]
    fn sitofp_input_must_take_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: double = sitofp %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn smax_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = smax %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn smax_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = smax %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn smin_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = smin %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn smin_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = smin %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn srem_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: double = arg [reg]
          %2: float = srem %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn srem_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: float = arg [reg]
          %1: float = arg [reg]
          %2: double = srem %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: ptr is not a pointer")]
    fn store_must_take_pointer() {
        str_to_mod::<DummyReg>(
            "
          %0: i64 = arg [reg]
          store %0, %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn sub_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = sub %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn sub_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = sub %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: term must be the last instruction in a trace")]
    fn term_must_come_last() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          term [%0]
          %2: i8 = add %0, %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: number of term vars does not match number of term VarLocs")]
    fn term_vars_must_match_exit_vlocs_in_number() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          term []
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: term var '%1' at position '0' does match type of %0")]
    fn term_vars_must_match_exit_vlocs_types() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i1 = arg [reg]
          term [%1, %1]
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not an integer type")]
    fn trunc_must_take_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: i64 = trunc %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not an integer type")]
    fn trunc_must_return_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: i1 = arg [reg]
          %1: ptr = trunc %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val bitw must be strictly greater than the return type bitw")]
    fn trunc_must_truncate() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = trunc %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn udiv_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = udiv %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn udiv_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = udiv %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not a floating point type")]
    fn uitofp_must_return_floating_point() {
        str_to_mod::<DummyReg>(
            "
          %0: i32 = arg [reg]
          %1: ptr = uitofp %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not an integer")]
    fn uitofp_input_must_take_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: double = uitofp %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn xor_type_consistency1() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i16 = arg [reg]
          %2: i16 = xor %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%2: inconsistent return / lhs / rhs types")]
    fn xor_type_consistency2() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i16 = xor %0, %1
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val is not an integer type")]
    fn zext_must_take_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: ptr = arg [reg]
          %1: i64 = zext %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: return type is not an integer type")]
    fn zext_must_return_integer() {
        str_to_mod::<DummyReg>(
            "
          %0: i1 = arg [reg]
          %1: ptr = zext %0
        ",
        );
    }

    #[test]
    #[should_panic(expected = "%1: val bitw must be strictly less than the return type bitw")]
    fn zext_must_extend() {
        str_to_mod::<DummyReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = zext %0
        ",
        );
    }

    #[test]
    fn cse_eq() {
        let m = str_to_mod::<TestReg>(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i64 = arg [reg]
          %3: i1 = icmp eq %0, %1
          guard true, %3, []
",
        );

        let mut opt = FullOpt::new_testing(m.guard_extras, m.tys);
        let TraceEnd::Test {
            entry_vlocs: _,
            block: Block { insts },
        } = m.trace_end
        else {
            panic!()
        };
        for inst in insts.into_iter() {
            if let Inst::Guard(_) = inst {
                opt.feed_void(inst).unwrap();
            } else {
                opt.feed(inst).unwrap();
            }
        }
        let tyidx = TyIdx::from_usize(0);
        let tyidx1 = TyIdx::from_usize(1);
        let val = InstIdx::from_usize(0);
        let val1 = InstIdx::from_usize(1);

        // FIXME: We need the series of tests below for every HIR element.
        let rhs = Inst::Abs(Abs {
            tyidx,
            val,
            int_min_poison: true,
        });
        assert!(
            Abs {
                tyidx,
                val,
                int_min_poison: true,
            }
            .cse_eq(&opt, &rhs)
        );
        assert!(
            !Abs {
                tyidx: tyidx1,
                val,
                int_min_poison: true,
            }
            .cse_eq(&opt, &rhs)
        );
        assert!(
            Abs {
                tyidx,
                val: val1,
                int_min_poison: true,
            }
            .cse_eq(&opt, &rhs)
        );
        assert!(
            !Abs {
                tyidx,
                val,
                int_min_poison: false,
            }
            .cse_eq(&opt, &rhs)
        );
    }
}
