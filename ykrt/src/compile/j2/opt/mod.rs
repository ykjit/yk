//! Optimisation.
//!
//! This module contains both the high-level outward-facing optimiser trait [OptT] and two
//! optimisers: [noopt] is the "don't optimise anything" "optimiser" mostly used by `YKD_OPT=0`;
//! and [fullopt] performs full optimisations. [fullopt] contains an inward-facing API that is
//! then used by optimisation passes.

use crate::compile::{CompilationError, j2::hir::*};
use index_vec::IndexVec;

mod cse;
pub(super) mod fullopt;
mod known_bits;
mod load_store;
pub(super) mod noopt;
mod strength_fold;

/// An outward-facing optimiser, used by [super::aot_to_hir]. By definition this operates on one
/// [Block] at a time, so it is both [ModLikeT] and [BlockLikeT].
pub(super) trait OptT: EquivIIdxT + ModLikeT + BlockLikeT {
    /// The block is now complete and the optimiser should turn it into a [Block] and a set of
    /// types (suitable for putting in a [Mod]).
    fn build(
        self: Box<Self>,
    ) -> Result<
        (
            Block,
            IndexVec<GuardExtraIdx, GuardExtra>,
            IndexVec<TyIdx, Ty>,
        ),
        CompilationError,
    >;

    #[allow(dead_code)]
    fn peel(self) -> (Block, Block);

    /// Feed a non-[Ty::Void] instruction into the optimiser and return an [InstIdx]. The returned
    /// [InstIdx] may refer to a previously inserted instruction, as an optimiser might prove that
    /// `inst` is unneeded. That previously inserted instruction may not even be of the same kind
    /// as `inst`!
    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError>;

    /// Feed a [Ty::Void], but not a [Guard], instruction into the optimiser, returning
    /// `Some([InstIdx])` if it was not removed. If `Some` is returned, the [InstIdx] may refer to
    /// a previously inserted instruction, as an optimiser might prove that `inst` is unneeded.
    /// That previously inserted instruction may not even be of the same kind as `inst`!
    ///
    /// Note: it is undefined behaviour to pass a [Guard] instruction to `feed_void`.
    /// [Self::feed_guard] must be used instead.
    fn feed_void(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError>;

    // Feed an argument into the trace. Arguments are either [Arg] or [Inst] instructions, and are
    // not optimised / deduplicated in any way. Calling this function after the first-non argument
    // has been passed will lead to undefined behaviour.
    fn feed_arg(&mut self, inst: Inst) -> Result<InstIdx, CompilationError>;

    /// Feed a [Guard], and its associated [GuardExtra] into the optimiser. The [Guard] must have
    /// its `geidx` value set to `GuardExtraIdx::MAX` or the optimiser will panic.
    ///
    /// Guards, by definition, are never returned as equivalent to a previous guard. If `Some` is
    /// returned, then a new guard has definitely been inserted into the instruction sequence.
    fn feed_guard(
        &mut self,
        inst: Guard,
        gextra: GuardExtra,
    ) -> Result<Option<InstIdx>, CompilationError>;

    /// Push a type [ty]. This type may be cached, and thus the [TyIdx] returned may not
    /// monotonically increase.
    fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError>;
}

/// The trait for objects (which may be blocks or optimisers or ...) which can return the current
/// known equivalent instruction for a given [InstIdx].
pub(super) trait EquivIIdxT {
    /// Return the instruction currently equivalent to `iidx`. By definition this will be `iidx` or
    /// a smaller value.
    ///
    /// Note that the value returned may vary as the optimiser receives more instructions. For
    /// example consider an input trace along the lines of:
    ///
    /// ```text
    /// %0: i32 = arg
    /// %1: i32 = arg
    /// %2: i32 = add %0, %1
    /// %3: i32 = 0
    /// %4: i1 = eq %0, %3
    /// guard true, %4, [...]
    /// %6: i32 = sub %0, %1
    /// ```
    ///
    /// From instructions 1..=5 `equiv_iidx(InstIdx::from(0))` will return `0`; from that point
    /// onwards it may (depending on the optimiser!) return `3` because the `guard` proves that it
    /// is equivalent.
    ///
    /// # Panics
    ///
    /// If `iidx` is greater than the number of instructions the optimiser currently holds.
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx;
}
