//! Trace optimisation.
//!
//! This is a forward-pass optimiser: as it advances, it builds up knowledge about past values that
//! can be taken advantage of by later values. In essence, as we go along we can view earlier
//! instructions as [equivalent](#equivalence) to later instructions.
//!
//! For example consider:
//!
//! ```text
//! %0: i8 = arg [reg]
//! %1: i8 = 9
//! %2: i1 = icmp eq, %0, %1
//! guard true, %2, []
//! %4: i8 = 4
//! %5: i8 = add %0, %4
//! blackbox %5
//! ```
//!
//! What do we know about `%2` as this trace is fed into the optimiser?
//!
//! * When feeding in `%0` and `%1` we do not know that `%2` will even exist: it is an error to ask
//!   questions about `%2` at this point.
//! * When feeding in the `icmp` we do not know if we will be able to prove that it is equivalent
//!   to a previous instruction. In other words we don't know if the `icmp` will become `%2` or
//!   will not be inserted.
//! * At the `guard` we know that `%2` is an `i1` whose values will be 0..255 (inc.).
//! * After the `guard` we know that `%0` must be exactly equal to `9` (otherwise the guard will
//!   have failed).
//!
//! After this point, we can then make use of our knowledge that `%0` and `%1` are equivalent:
//!
//! * We can now view the `add` instruction as equivalent to `add %1, %4`.
//! * Recognising that the `add` is thus adding two constants, we can constant fold the `add` to
//!   the constant 13.
//!
//! Thus after optimisation (including dead code elimination) the trace will look as follows:
//!
//! ```text
//! %0: i8 = arg [reg]
//! %1: i8 = 9
//! %2: i1 = icmp eq, %0, %1
//! guard true, %2, []
//! %5: i8 = 13
//! blackbox %5
//! ```
//!
//!
//! ## Equivalence
//!
//! The notion of instruction "equivalence" is both intuitive and easy to over-generalise.
//! Importantly: "equivalence" only makes sense during forward-pass optimisation. If at point X we
//! prove that instruction A is equivalent to instruction B, that knowledge is only correct for
//! instructions at positions > X. In other words, that knowledge is correctly iff one has
//! consecutively executed all instructions (notably guards!) up to, and including, X. After
//! optimisation is complete, and a HIR [Module] is created, we throw away all notions of
//! equivalence: they are by definition baked into the optimised trace.
//!
//! Equivalence is identified in two distinct ways:
//!
//! 1. When an instruction A is fed into the optimiser, it might be identified as equivalent to an
//!    existing instruction B, at which point A will be discarded, and the caller informed that B
//!    can be used instead of A.
//! 2. During optimisation, two earlier instructions might be identified as equivalent.
//!
//! In the case of (1), B is by definition the "winner": A is never inserted. In the case of (2),
//! the winner is the instruction that will be used henceforth. If A and B are identified as
//! equivalent, then: if one, but not both, of A or B is a constant, that instruction is the
//! winner; otherwise B will be the winner. This latter condition is not fundamental, but is
//! necessary to make writing tests plausible.
//!
//!
//! ## Structure of the optimiser
//!
//! The optimiser consists of an "outward facing to [aot_to_hir]" part ([FullOpt], implementing
//! [OptT]) and in "inward facing to passes" part ([PassOpt]). What links these two things is
//! [OptInternal]: the outer and inner parts both have access to this, passing ownership between
//! themselves as appropriate.
//!
//! The optimiser API is deliberately simple: an instruction goes into the optimisation chain and
//! passes operate on it. At the end it is either: committed to the trace; determined to be
//! equivalent to another instruction; or shown not to be needed at all.
//!
//!
//! ## Committing a pass's results
//!
//! As a pass operates, it may wish to add new instructions to the trace and/or identify new
//! instruction equivalences. To keep passes simple, nothing about a trace is mutated _during_ a
//! pass: instead a pass can identify changes which will be committed _after_ the pass has
//! completed. For clarity: if pass A identifies changes, those will be committed before pass B is
//! run.
//!
//!
//! ### Preinstructions
//!
//! Inserting instructions into the trace does not fit with the "optimise one instruction" model.
//! The [PassOpt] API thus has the concept of "preinstructions": that is, instructions which will
//! be committed to the trace before the current instruction. Preinstructions should be used
//! carefully:
//!
//! 1. They are only committed at the _end_ of the current pass. In other words, if a pass P calls
//!    [PassOpt::push_pre_inst] then those instructions are not accessible via [PassOpt::inst] for
//!    the duration of P. As soon as P has completed, the preinstructions will be committed, and
//!    subsequent passes will have access to them. Note: it is guaranteed that the [InstIdx]s
//!    returned by [PassOpt::push_pre_inst] will be valid after the preinstructions are
//!    subsequently committed.
//!
//! 2. They are committed unconditionally at the end of a pass, even if a subsequent pass proves
//!    that the instruction they were originally associated with is no longer needed.
//!
//! In practise, the main use of preinstructions is for constants: modulo (1), there are no real
//! issues for constants. Similarly, for other side-effect-free instructions, one may put some
//! pressure on dead-code elimination, but not on the eventual compiled trace. However, for
//! side-effectful instructions of any kind, one should be extremely cautious about committing them
//! as preinstructions.
//!
//!
//! ### New equivalences
//!
//! As a pass operates, it may identify that two previous instructions in the trace are now
//! equivalent to each other. [PassOpt::push_equiv] can be called as often as desired during a
//! pass, bearing in mind that these instruction equivalences will only be committed when the pass
//! completes.

use crate::compile::{
    CompilationError,
    j2::{
        effects::Effects,
        hir::*,
        opt::{
            EquivIIdxT, OptT, cse::CSE, known_bits::KnownBits, load_store::LoadStore,
            strength_fold::StrengthFold,
        },
    },
};
use index_vec::*;
use smallvec::SmallVec;
use std::{
    assert_matches,
    collections::HashMap,
    hash::{Hash, Hasher},
    mem,
};
use vob::Vob;

////////////////////////////////////////////////////////////////////////////////
// The external-facing part of the optimiser.

pub(in crate::compile::j2) struct FullOpt {
    /// The ordered set of optimisation passes that all instructions will be fed through.
    passes: [Box<dyn PassT>; 4],
    inner: OptInternal,
}

impl FullOpt {
    pub(in crate::compile::j2) fn new() -> Self {
        let mut tys = IndexVec::new();
        let mut ty_map = HashMap::new();
        let tyidx_void = tys.push(Ty::Void);
        ty_map.insert(Ty::Void, tyidx_void);
        let tyidx_ptr0 = tys.push(Ty::Ptr(0));
        ty_map.insert(Ty::Ptr(0), tyidx_ptr0);
        let tyidx_int1 = tys.push(Ty::Int(1));
        ty_map.insert(Ty::Int(1), tyidx_int1);
        Self {
            passes: [
                Box::new(KnownBits::new()),
                Box::new(StrengthFold::new()),
                Box::new(LoadStore::new()),
                Box::new(CSE::new()),
            ],
            inner: OptInternal {
                insts: IndexVec::new(),
                consts_map: HashMap::new(),
                guard_extras: IndexVec::new(),
                tys,
                tyidx_int1,
                tyidx_ptr0,
                tyidx_void,
                ty_map,
            },
        }
    }

    #[cfg(test)]
    pub(in crate::compile::j2) fn new_testing(tys: IndexVec<TyIdx, Ty>) -> Self {
        let ty_map = HashMap::from_iter(
            tys.iter()
                .enumerate()
                .map(|(x, y)| (y.to_owned(), TyIdx::from(x))),
        );
        let tyidx_ptr0 = *ty_map.get(&Ty::Ptr(0)).unwrap_or_else(|| panic!());
        let tyidx_void = *ty_map.get(&Ty::Void).unwrap_or_else(|| panic!());
        let tyidx_int1 = *ty_map.get(&Ty::Int(1)).unwrap_or_else(|| panic!());
        Self {
            passes: [
                Box::new(KnownBits::new()),
                Box::new(StrengthFold::new()),
                Box::new(LoadStore::new()),
                Box::new(CSE::new()),
            ],
            inner: OptInternal {
                insts: IndexVec::new(),
                consts_map: HashMap::new(),
                guard_extras: IndexVec::new(),
                tys,
                tyidx_int1,
                tyidx_ptr0,
                tyidx_void,
                ty_map,
            },
        }
    }

    /// Used by [Self::feed] and [Self::feed_void].
    fn feed_internal(
        &mut self,
        mut popt_inner: PassOptInner,
        mut inst: Inst,
    ) -> Result<Option<InstIdx>, CompilationError> {
        for i in 0..self.passes.len() {
            let mut opt = PassOpt {
                optinternal: &mut self.inner,
                inner: &mut popt_inner,
            };

            let fed = self.passes[i].feed(&mut opt, inst);

            for inst in popt_inner.pre_insts.drain(..) {
                self.commit_inst_dedup_opt(inst);
            }

            match fed {
                OptOutcome::NotNeeded => return Ok(None),
                OptOutcome::Rewritten(new_inst) => inst = new_inst,
                OptOutcome::Equiv(iidx) => return Ok(Some(iidx)),
            }
        }

        for (equiv1, equiv2) in popt_inner.new_equivs.drain(..) {
            let (equiv1, equiv2) = match (self.inst(equiv1), self.inst(equiv2)) {
                (Inst::Const(_), Inst::Const(_)) => {
                    continue;
                }
                (_, Inst::Const(_)) => (equiv1, equiv2),
                (_, _) => (equiv2, equiv1),
            };
            self.inner.insts.get_mut(equiv1).unwrap().equiv = equiv2;
            for pass in &mut self.passes {
                pass.equiv_committed(equiv1, equiv2);
            }
        }

        if let Inst::Guard(Guard { ref mut geidx, .. }) = inst {
            assert_eq!(*geidx, GuardExtraIdx::MAX);
            *geidx = self.inner.guard_extras.push(popt_inner.gextra.unwrap());
        } else {
            assert!(popt_inner.gextra.is_none());
        }

        Ok(Some(self.commit_inst_dedup_opt(inst)))
    }

    /// Commit `inst` to this trace, deduplicating constants if they already exist earlier in the
    /// trace.
    fn commit_inst_dedup_opt(&mut self, inst: Inst) -> InstIdx {
        if let Inst::Const(x) = &inst
            && let Some(x) = self.inner.consts_map.get(&HashableConst(x.to_owned()))
        {
            return *x;
        }
        self.commit_inst(inst)
    }

    /// Commit `inst` to this trace, without deduplicating constants.
    fn commit_inst(&mut self, inst: Inst) -> InstIdx {
        let opt = CommitInstOpt { inner: &self.inner };
        let iidx = self.inner.insts.len_idx();
        for pass in &mut self.passes {
            pass.inst_committed(&opt, iidx, &inst);
        }

        if let Inst::Const(x) = &inst {
            self.inner
                .consts_map
                .insert(HashableConst(x.to_owned()), iidx);
        }
        self.inner.insts.push(InstEquiv {
            inst,
            equiv: InstIdx::MAX,
        })
    }
}

impl ModLikeT for FullOpt {
    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        panic!("Not available in optimiser");
    }

    fn ty(&self, tyidx: TyIdx) -> &Ty {
        self.inner.ty(tyidx)
    }

    fn tyidx_int1(&self) -> TyIdx {
        self.inner.tyidx_int1
    }

    fn tyidx_ptr0(&self) -> TyIdx {
        self.inner.tyidx_ptr0
    }

    fn tyidx_void(&self) -> TyIdx {
        self.inner.tyidx_void
    }
}

impl BlockLikeT for FullOpt {
    fn inst(&self, idx: InstIdx) -> &Inst {
        &self.inner.insts[usize::from(idx)].inst
    }

    fn insts_len(&self) -> usize {
        self.inner.insts.len()
    }

    fn gextra(&self, geidx: GuardExtraIdx) -> &GuardExtra {
        &self.inner.guard_extras[geidx]
    }

    fn gextra_mut(&mut self, geidx: GuardExtraIdx) -> &mut GuardExtra {
        &mut self.inner.guard_extras[geidx]
    }
}

impl OptT for FullOpt {
    fn build(self: Box<Self>) -> Result<(Block, IndexVec<TyIdx, Ty>), CompilationError> {
        Ok((
            Block {
                insts: self
                    .inner
                    .insts
                    .into_iter()
                    .map(|x| x.inst)
                    .collect::<IndexVec<_, _>>(),
                guard_extras: self.inner.guard_extras,
            },
            self.inner.tys,
        ))
    }

    fn build_with_peel(
        mut self: Box<Self>,
    ) -> Result<(Block, Option<Block>, IndexVec<TyIdx, Ty>), CompilationError> {
        // First of all create the entry iteration `Block`. This is useful when we're creating the
        // peel `Block` below.
        let entry = Block {
            insts: mem::take(&mut self.inner.insts)
                .into_iter()
                .map(|x| x.inst)
                .collect::<IndexVec<_, _>>(),
            guard_extras: mem::take(&mut self.inner.guard_extras),
        };
        self.inner.consts_map.clear();
        assert!(self.inner.guard_extras.is_empty());

        // Dead code analysis: we don't want to waste energy feeding things into the peel that
        // can't possibly be used.
        let mut is_used = Vob::from_elem(false, entry.insts.len());
        for (iidx, inst) in entry.insts_iter(..).rev() {
            if is_used[usize::from(iidx)] || inst.write_effects().interferes(Effects::all()) {
                is_used.set(usize::from(iidx), true);
                for op_iidx in inst.iter_iidxs(&entry) {
                    is_used.set(usize::from(op_iidx), true);
                }
            }
        }

        // For each terminal exit variable in the entry block, create an arg/const in the peel
        // block.
        let mut map = index_vec![InstIdx::MAX; entry.insts.len()];
        for entry_iidx in entry.term_vars() {
            let peel_iidx = self.inner.insts.len_idx();
            // FIXME: The next line won't work when we do loop-invariant code motion.
            map[peel_iidx] = peel_iidx;
            map[*entry_iidx] = peel_iidx;
            match entry.inst(*entry_iidx) {
                Inst::Const(x) => {
                    self.inner.insts.push(InstEquiv {
                        inst: x.clone().into(),
                        equiv: InstIdx::MAX,
                    });
                    self.inner
                        .consts_map
                        .insert(HashableConst(x.to_owned()), peel_iidx);
                }
                _ => {
                    let tyidx = entry.inst(*entry_iidx).tyidx(&*self);
                    self.inner.insts.push(InstEquiv {
                        inst: Arg { tyidx }.into(),
                        equiv: InstIdx::MAX,
                    });
                }
            }
        }

        // Set the optimisation passes up for peeling.
        let mut popt_inner = PassOptInner::new();
        let mut popt = PassOpt {
            optinternal: &mut self.inner,
            inner: &mut popt_inner,
        };
        for pass in &mut self.passes {
            pass.prepare_for_peel(&mut popt, &entry, &map);
        }
        assert!(popt_inner.gextra.is_none());
        assert!(popt_inner.new_equivs.is_empty());
        for inst in popt_inner.pre_insts.drain(..) {
            self.commit_inst(inst);
        }

        // Feed each instruction from `entry` (except the arguments!) into the peel.
        for (iidx, inst) in entry
            .insts_iter(..)
            .skip(entry.term_vars().len()) // Skip `Arg` instructions
            .filter(|(iidx, _)| is_used[usize::from(*iidx)])
        {
            let mut inst = inst.clone();
            if let Inst::Guard(mut x) = inst {
                let old_gextra = &entry.guard_extras[x.geidx];
                let gextra = GuardExtra {
                    bid: old_gextra.bid,
                    switch: old_gextra.switch.clone(),
                    deopt_vars: old_gextra
                        .deopt_vars
                        .iter()
                        .map(|x| map[*x])
                        .collect::<Vec<_>>(),
                    deopt_frames: old_gextra.deopt_frames.clone(),
                };
                x.cond = map[x.cond];
                x.geidx = GuardExtraIdx::MAX;
                // FIXME: Peeled guards can find contradictions because not all loops are
                // control-flow stable.
                self.feed_guard(x, gextra)?;
            } else if inst.tyidx(&*self) == self.inner.tyidx_void {
                inst.rewrite_iidxs(&mut *self, |x| map[x]);
                self.feed_void(inst)?;
            } else {
                inst.rewrite_iidxs(&mut *self, |x| map[x]);
                map[iidx] = self.feed(inst)?;
            }
        }

        let peel = Block {
            insts: mem::take(&mut self.inner.insts)
                .into_iter()
                .map(|x| x.inst)
                .collect::<IndexVec<_, _>>(),
            guard_extras: mem::take(&mut self.inner.guard_extras),
        };

        Ok((entry, Some(peel), self.inner.tys))
    }

    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_ne!(inst.tyidx(self), self.tyidx_void());
        self.feed_internal(PassOptInner::new(), inst)
            .map(|x| x.unwrap())
    }

    fn feed_void(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(inst.tyidx(self), self.tyidx_void());
        assert!(!matches!(inst, Inst::Guard(_)));
        self.feed_internal(PassOptInner::new(), inst)
    }

    fn feed_arg(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_matches!(inst, Inst::Arg(_) | Inst::Const(_));
        Ok(self.commit_inst(inst))
    }

    fn feed_guard(
        &mut self,
        inst: Guard,
        gextra: GuardExtra,
    ) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(inst.geidx, GuardExtraIdx::MAX);
        self.feed_internal(PassOptInner::with_gextra(gextra), inst.into())
    }

    fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        self.inner.push_ty(ty)
    }
}

impl EquivIIdxT for FullOpt {
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx {
        self.inner.equiv_iidx(iidx)
    }
}

////////////////////////////////////////////////////////////////////////////////
// The shared part of the optimiser.

struct OptInternal {
    insts: IndexVec<InstIdx, InstEquiv>,
    /// A map allowing us to deduplicate constants. Note the use of [HashableConst]: constant
    /// deduplication is "best effort" because of the difficulties imposed by floating point
    /// numbers.
    consts_map: HashMap<HashableConst, InstIdx>,
    guard_extras: IndexVec<GuardExtraIdx, GuardExtra>,
    tys: IndexVec<TyIdx, Ty>,
    /// The [TyIdx] for [Ty::Int(1)].
    tyidx_int1: TyIdx,
    /// The [TyIdx] for [Ty::Ptr(0)].
    tyidx_ptr0: TyIdx,
    /// The [TyIdx] for [Ty::Void].
    tyidx_void: TyIdx,
    /// A map allowing us to deduplicate types. This guarantees that a given [Ty] appears exactly
    /// once in a module.
    ty_map: HashMap<Ty, TyIdx>,
}

impl OptInternal {
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx {
        let mut search = iidx;
        loop {
            let equiv = self.insts[search].equiv;
            if equiv == InstIdx::MAX {
                // FIXME: Reinstate this.
                // if search != iidx {
                //     self.insts[iidx].equiv = search;
                // }
                return search;
            }
            search = equiv;
        }
    }

    fn inst(&self, iidx: InstIdx) -> &Inst {
        &self.insts[iidx].inst
    }

    fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        if let Some(tyidx) = self.ty_map.get(&ty) {
            return Ok(*tyidx);
        }
        let tyidx = self.tys.push(ty.clone());
        self.ty_map.insert(ty, tyidx);
        Ok(tyidx)
    }

    fn ty(&self, tyidx: TyIdx) -> &Ty {
        &self.tys[tyidx]
    }
}

/// A wrapper around [Const]s that allows them to be hashed for deduplication purposes. This struct
/// is conservative as per [super::super::hir]'s documentation: it treats floating point numbers as
/// bit patterns. That means, for example, that it will not identify `+0.0` and `-0.0` as
/// equivalent. This struct should therefore be used with caution!
struct HashableConst(Const);

impl Hash for HashableConst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.tyidx.hash(state);
        match &self.0.kind {
            ConstKind::Double(x) => x.to_bits().hash(state),
            ConstKind::Float(x) => x.to_bits().hash(state),
            ConstKind::Int(x) => x.hash(state),
            ConstKind::Ptr(x) => x.hash(state),
        }
    }
}

impl PartialEq for HashableConst {
    fn eq(&self, other: &Self) -> bool {
        self.0.tyidx == other.0.tyidx
            && match (&self.0.kind, &other.0.kind) {
                (ConstKind::Double(x), ConstKind::Double(y)) => x.to_bits() == y.to_bits(),
                (ConstKind::Float(x), ConstKind::Float(y)) => x.to_bits() == y.to_bits(),
                (ConstKind::Int(x), ConstKind::Int(y)) => x == y,
                (ConstKind::Ptr(x), ConstKind::Ptr(y)) => x == y,
                (_, _) => false,
            }
    }
}

impl Eq for HashableConst {}

/// What an optimisation has managed to make of a given input [Inst].
#[derive(Debug)]
pub enum OptOutcome {
    NotNeeded,
    /// The input [Inst] has been rewritten to a new [Inst].
    Rewritten(Inst),
    /// The input [Inst] is equivalent to [InstIdx].
    Equiv(InstIdx),
}

/// A wrapper around an [Inst] and our current knowledge of another instruction it is equivalent
/// to.
#[derive(Debug)]
struct InstEquiv {
    /// The [Inst] at a given [InstIdx].
    inst: Inst,
    /// As the optimiser has advanced, we might have been able to prove that `inst` is equivalent
    /// to a later instruction: if so, `equiv` will give the index of that instruction; if not, it
    /// will be set [InstIdx::MAX].
    equiv: InstIdx,
}

////////////////////////////////////////////////////////////////////////////////
// The internal-facing part of the optimiser.

/// The interface that passes must implement.
pub(super) trait PassT {
    /// Feed [inst] instruction into the optimiser.
    fn feed(&mut self, opt: &mut PassOpt, inst: Inst) -> OptOutcome;

    /// After an instruction has been committed to the trace -- i.e. there is
    /// no possibility that an optimisation pass will remove it -- this
    /// function will be called on all passes.
    fn inst_committed(&mut self, ci: &CommitInstOpt, iidx: InstIdx, inst: &Inst);

    /// `equiv1` and `equiv2` have been identified as equivalent and henceforth `equiv1` will be
    /// rewritten to `equiv2`.
    fn equiv_committed(&mut self, equiv1: InstIdx, equiv2: InstIdx);

    /// Prepare for peeling on `entry`: the pass can copy information across from the first
    /// iteration to the peel. `map` contains a map from instruction indices so far known in the
    /// peel to the instruction index in the first iteration.
    fn prepare_for_peel(
        &mut self,
        opt: &mut PassOpt,
        entry: &Block,
        map: &IndexVec<InstIdx, InstIdx>,
    );
}

/// The object passed to [PassT::feed] so that they can interact with the optimiser.
pub struct PassOpt<'a> {
    optinternal: &'a mut OptInternal,
    inner: &'a mut PassOptInner,
}

impl PassOpt<'_> {
    /// If `iidx` references a constant, return an owned version of the accompany [ConstKind], or
    /// `None` otherwise.
    pub(super) fn as_constkind(&self, iidx: InstIdx) -> Option<ConstKind> {
        match self.inst(iidx) {
            Inst::Const(Const { kind, .. }) => Some(kind.clone()),
            _ => None,
        }
    }

    /// Push `ty`: this is safe for passes to call at any point.
    pub(super) fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        self.optinternal.push_ty(ty)
    }

    /// Add the "pre-instruction" `preinst`: when the current pass has completed, `inst` will be
    /// committed to the trace, and immediately available to subsequent passes. The [InstIdx] that
    /// is returned will be `preinst`'s [InstIdx] _after_ it is committed. In other words, during
    /// the current pass, [InstIdx] is invalid, and attempting any operations with it will lead to
    /// undefined behaviour.
    pub(super) fn push_pre_inst(&mut self, preinst: Inst) -> InstIdx {
        if let Inst::Const(c) = &preinst
            && let Some(x) = self
                .optinternal
                .consts_map
                .get(&HashableConst(c.to_owned()))
        {
            return *x;
        }
        let iidx = InstIdx::from_usize(self.optinternal.insts.len() + self.inner.pre_insts.len());
        self.inner.pre_insts.push(preinst);
        iidx
    }

    /// When the current pass has completed, `equiv1` will be considered equivalent to `equiv2`.
    /// One of these will necessarily be picked as the "winner" in the sense that, after this pass,
    /// `equiv1` will always be rewritten to `equiv2` or vice versa. If one, but both, of the
    /// instructions is a constant, it is guaranteed to be the winner; otherwise `equiv2` will be
    /// picked as the winner. The latter is not fundamental, but is necessary to make writing tests
    /// plausible.
    ///
    /// Note: it is the caller's job to ensure that `iidx` and `equiv_to` have already been
    /// transformed for equivalence.
    pub(super) fn push_equiv(&mut self, equiv1: InstIdx, equiv2: InstIdx) {
        self.inner.new_equivs.push((equiv1, equiv2));
    }
}

impl BlockLikeT for PassOpt<'_> {
    fn inst(&self, iidx: InstIdx) -> &Inst {
        self.optinternal.inst(iidx)
    }

    fn insts_len(&self) -> usize {
        self.optinternal.insts.len()
    }

    fn gextra(&self, geidx: GuardExtraIdx) -> &GuardExtra {
        if geidx == GuardExtraIdx::MAX {
            self.inner.gextra.as_ref().unwrap()
        } else {
            &self.optinternal.guard_extras[geidx]
        }
    }

    fn gextra_mut(&mut self, geidx: GuardExtraIdx) -> &mut GuardExtra {
        if geidx == GuardExtraIdx::MAX {
            self.inner.gextra.as_mut().unwrap()
        } else {
            &mut self.optinternal.guard_extras[geidx]
        }
    }
}

impl ModLikeT for PassOpt<'_> {
    fn ty(&self, tyidx: TyIdx) -> &Ty {
        self.optinternal.ty(tyidx)
    }

    fn tyidx_int1(&self) -> TyIdx {
        self.optinternal.tyidx_int1
    }

    fn tyidx_ptr0(&self) -> TyIdx {
        self.optinternal.tyidx_ptr0
    }

    fn tyidx_void(&self) -> TyIdx {
        self.optinternal.tyidx_void
    }

    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        todo!()
    }
}

impl EquivIIdxT for PassOpt<'_> {
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx {
        self.optinternal.equiv_iidx(iidx)
    }
}

struct PassOptInner {
    gextra: Option<GuardExtra>,
    new_equivs: SmallVec<[(InstIdx, InstIdx); 1]>,
    pre_insts: SmallVec<[Inst; 1]>,
}

impl PassOptInner {
    fn new() -> Self {
        Self {
            gextra: None,
            new_equivs: SmallVec::new(),
            pre_insts: SmallVec::new(),
        }
    }

    fn with_gextra(gextra: GuardExtra) -> Self {
        Self {
            gextra: Some(gextra),
            new_equivs: SmallVec::new(),
            pre_insts: SmallVec::new(),
        }
    }
}

/// The object passed to [PassT::inst_committed] so that they can interact with the optimiser.
pub struct CommitInstOpt<'a> {
    inner: &'a OptInternal,
}

impl BlockLikeT for CommitInstOpt<'_> {
    fn inst(&self, iidx: InstIdx) -> &Inst {
        self.inner.inst(iidx)
    }

    fn insts_len(&self) -> usize {
        self.inner.insts.len()
    }

    fn gextra(&self, _geidx: GuardExtraIdx) -> &GuardExtra {
        todo!();
    }

    fn gextra_mut(&mut self, _geidx: GuardExtraIdx) -> &mut GuardExtra {
        todo!();
    }
}

impl ModLikeT for CommitInstOpt<'_> {
    fn ty(&self, tyidx: TyIdx) -> &Ty {
        self.inner.ty(tyidx)
    }

    fn tyidx_int1(&self) -> TyIdx {
        self.inner.tyidx_int1
    }

    fn tyidx_ptr0(&self) -> TyIdx {
        self.inner.tyidx_ptr0
    }

    fn tyidx_void(&self) -> TyIdx {
        self.inner.tyidx_void
    }

    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        todo!()
    }
}

impl EquivIIdxT for CommitInstOpt<'_> {
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx {
        self.inner.equiv_iidx(iidx)
    }
}

#[cfg(test)]
pub(in crate::compile::j2) mod test {
    use super::*;
    use crate::compile::j2::{hir_parser::str_to_mod, regalloc::RegT, regalloc::test::TestReg};
    use fm::FMBuilder;
    use index_vec::IndexVec;
    use lazy_static::lazy_static;
    use regex::Regex;

    lazy_static! {
        static ref PTN_RE: Regex = Regex::new(r"\{\{.+?\}\}").unwrap();
        static ref PTN_RE_IGNORE: Regex = Regex::new(r"\{\{_}\}").unwrap();
        static ref TEXT_RE: Regex = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
    }

    pub(in crate::compile::j2) fn str_to_peel_mod<Reg: RegT>(mod_s: &str) -> Mod<Reg> {
        let m = str_to_mod::<Reg>(mod_s);
        let TraceEnd::Test {
            entry_vlocs,
            block: Block {
                insts,
                guard_extras,
            },
        } = &m.trace_end
        else {
            panic!()
        };
        let mut fopt = Box::new(FullOpt::new());
        fopt.inner.guard_extras = guard_extras.clone();
        fopt.inner.tys = m.tys.clone();

        // We need to maintain a manual map of iidxs the user has written in their test to the
        // current state of the actual optimiser. Consider:
        //
        // ```
        // %0: i8 = arg [reg]
        // %1: i8 = 0
        // %2: i8 = add %0, %1
        // %3: blackbox %2
        // ```
        //
        // The `add` is a no-op, and won't be inserted into the optimisation's ongoing module at
        // all, so when we get to `blackbox %2` there is no `%2` to reference: we'd get an
        // out-of-bounds error! We thus need to rewrite this to `blackbox %0` _before_ feeding the
        // instruction to the optimiser.
        let mut opt_map = IndexVec::with_capacity(insts.len());
        let mut insts_iter = insts.into_iter();
        for _ in 0..entry_vlocs.len() {
            opt_map.push(opt_map.len_idx());
            fopt.feed_arg(insts_iter.next().unwrap().clone()).unwrap();
        }
        for inst in insts_iter {
            let mut inst = inst.clone();
            inst.rewrite_iidxs(&mut *fopt, |x| opt_map[x]);
            if let Inst::Guard(mut x @ Guard { geidx, .. }) = inst {
                x.geidx = GuardExtraIdx::MAX;
                fopt.feed_guard(x, fopt.inner.guard_extras[geidx].clone())
                    .unwrap();
                opt_map.push(opt_map.len_idx());
            } else if let Inst::Arg(_) = inst {
                unreachable!();
            } else if *m.ty(inst.tyidx(&m)) == Ty::Void {
                if let Some(x) = fopt.feed_void(inst).unwrap() {
                    opt_map.push(x);
                } else {
                    opt_map.push(opt_map.len_idx());
                }
            } else {
                opt_map.push(fopt.feed(inst).unwrap());
            }
        }
        let tyidx_int1 = fopt.inner.tyidx_int1;
        let tyidx_ptr0 = fopt.inner.tyidx_ptr0;
        let tyidx_void = fopt.inner.tyidx_void;
        let (entry, peel, tys) = fopt.build_with_peel().unwrap();
        Mod {
            trid: m.trid,
            trace_start: TraceStart::Test,
            trace_end: TraceEnd::TestPeel {
                entry_vlocs: entry_vlocs.to_owned(),
                entry,
                peel: peel.unwrap(),
            },
            tys,
            tyidx_int1,
            tyidx_ptr0,
            tyidx_void,
            addr_name_map: None,
            smaps: m.smaps,
        }
    }

    pub(in crate::compile::j2::opt) fn full_opt_test(mod_s: &str, ptn: &str) {
        let m = str_to_peel_mod::<TestReg>(mod_s);
        let s = m.to_string();
        let fmb = FMBuilder::new(ptn)
            .unwrap()
            .name_matcher_ignore(PTN_RE_IGNORE.clone(), TEXT_RE.clone())
            .name_matcher(PTN_RE.clone(), TEXT_RE.clone())
            .build()
            .unwrap();
        if let Err(e) = fmb.matches(&s) {
            eprintln!("{e}");
            panic!();
        }
    }

    /// Take an input module string `mod_s` and run it through user-defined passes in the function
    /// `feed_f`. After each call of `feed_f`, `inst_committed` and/or `equiv_committed` will be
    /// called as appropriate.
    pub(in crate::compile::j2) fn user_defined_opt_test<F, G, H>(
        mod_s: &str,
        feed_f: F,
        inst_committed_f: G,
        equiv_committed_f: H,
        ptn: &str,
    ) where
        for<'a> F: Fn(&'a mut PassOpt, Inst) -> OptOutcome,
        for<'a> G: Fn(&'a CommitInstOpt, InstIdx, &Inst),
        for<'a> H: Fn(InstIdx, InstIdx),
    {
        let m = str_to_mod::<TestReg>(mod_s);
        let mut fopt = Box::new(FullOpt::new());
        let TraceEnd::Test {
            entry_vlocs,
            block: Block {
                insts,
                guard_extras,
            },
        } = m.trace_end
        else {
            panic!()
        };
        fopt.inner.guard_extras = guard_extras;
        fopt.inner.tys = m.tys;
        for (tyidx, ty) in fopt.inner.tys.iter_enumerated() {
            fopt.inner.ty_map.insert(ty.clone(), tyidx);
        }
        // We need to maintain a manual map of iidxs the user has written in their test to the
        // current state of the actual optimiser. See the comment in [full_opt_test].
        let mut opt_map = IndexVec::with_capacity(insts.len());
        for mut inst in insts.into_iter() {
            inst.rewrite_iidxs(&mut *fopt, |x| opt_map[x]);
            let mut popt_inner = PassOptInner::new();
            let mut opt = PassOpt {
                optinternal: &mut fopt.inner,
                inner: &mut popt_inner,
            };
            let fed = feed_f(&mut opt, inst);

            for inst in popt_inner.pre_insts.drain(..) {
                let iidx = fopt.inner.insts.len_idx();
                let opt = CommitInstOpt { inner: &fopt.inner };
                inst_committed_f(&opt, iidx, &inst);
                fopt.inner.insts.push(InstEquiv {
                    inst,
                    equiv: InstIdx::MAX,
                });
            }

            for (equiv1, equiv2) in popt_inner.new_equivs.drain(..) {
                let (equiv1, equiv2) = match (fopt.inst(equiv1), fopt.inst(equiv2)) {
                    (_, Inst::Const(_)) => (equiv1, equiv2),
                    (_, _) => (equiv2, equiv1),
                };
                fopt.inner.insts.get_mut(equiv1).unwrap().equiv = equiv2;
                equiv_committed_f(equiv1, equiv2);
            }

            match fed {
                OptOutcome::NotNeeded => {
                    opt_map.push(InstIdx::MAX);
                    continue;
                }
                OptOutcome::Rewritten(new_inst) => {
                    inst = new_inst;
                }
                OptOutcome::Equiv(iidx) => {
                    opt_map.push(iidx);
                    continue;
                }
            }

            let iidx = fopt.inner.insts.len_idx();
            opt_map.push(iidx);
            let opt = CommitInstOpt { inner: &fopt.inner };
            inst_committed_f(&opt, iidx, &inst);
            fopt.inner.insts.push(InstEquiv {
                inst,
                equiv: InstIdx::MAX,
            });
        }
        let tyidx_int1 = fopt.inner.tyidx_int1;
        let tyidx_ptr0 = fopt.inner.tyidx_ptr0;
        let tyidx_void = fopt.inner.tyidx_void;
        let (block, tys) = fopt.build().unwrap();
        let m = Mod {
            trid: m.trid,
            trace_start: TraceStart::Test,
            trace_end: TraceEnd::Test { entry_vlocs, block },
            tys,
            tyidx_int1,
            tyidx_ptr0,
            tyidx_void,
            addr_name_map: None,
            smaps: m.smaps,
        };
        let s = m.to_string();
        let fmb = FMBuilder::new(ptn)
            .unwrap()
            .name_matcher_ignore(PTN_RE_IGNORE.clone(), TEXT_RE.clone())
            .name_matcher(PTN_RE.clone(), TEXT_RE.clone())
            .build()
            .unwrap();
        if let Err(e) = fmb.matches(&s) {
            eprintln!("{e}");
            panic!();
        }
    }

    #[test]
    fn equivalency() {
        // This test is a somewhat weak test -- but better than nothing! -- that instruction
        // equivalency works during optimisation stages. This partly tests functions in this module
        // but also relies on `strength_fold` doing the right thing.

        fn test_sf(mod_s: &str, ptn: &str) {
            user_defined_opt_test(
                mod_s,
                |opt, mut inst| {
                    inst.canonicalise(opt);
                    StrengthFold::new().feed(opt, inst)
                },
                |_, _, _| (),
                |_, _| (),
                ptn,
            );
        }

        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 4
          %3: i1 = icmp eq %0, %2
          %4: i8 = 4
          %5: i1 = icmp eq %0, %4
          guard true, %3, []
          guard true, %5, []
          blackbox %0
          blackbox %4
          term [%0]
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 4
          %3: i1 = icmp eq %0, %2
          %4: i8 = 4
          %5: i1 = icmp eq %0, %4
          guard true, %3, []
          guard true, %5, []
          blackbox %4
          blackbox %4
          term [%4]
          ...
        ",
        );

        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = 1
          %3: i8 = 4
          %4: i1 = icmp eq %0, %3
          %5: i1 = icmp eq %0, %1
          guard true, %4, []
          guard true, %5, []
          blackbox %0
          blackbox %1
          term [%0, %1]
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = 1
          %3: i8 = 4
          %4: i1 = icmp eq %0, %3
          %5: i1 = icmp eq %0, %1
          guard true, %4, []
          guard true, %5, []
          blackbox %3
          blackbox %3
          term [%3, %3]
          ...
        ",
        );
    }

    #[test]
    fn peeling() {
        // Simple constant propagation and strength/fold in the peel.
        full_opt_test(
            "
          %0: i8 = arg [reg]
          %1: i8 = 4
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          term [%0]
        ",
            "
          %0: i8 = arg
          %1: i8 = 4
          %2: i1 = icmp eq %0, %1
          %3: i8 = 4
          guard true, %2, []
          term [%1]
          ; peel
          %0: i8 = 4
          %1: i1 = 1
          term [%0]
        ",
        );

        // Simple load/store optimisation across the peel.
        full_opt_test(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = load %0
          %3: i8 = 4
          %4: i1 = icmp eq %2, %3
          guard true, %4, []
          term [%0, %2]
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          %2: i8 = load %0
          %3: i8 = 4
          %4: i1 = icmp eq %2, %3
          %5: i8 = 4
          guard true, %4, []
          term [%0, %3]
          ; peel
          %0: ptr = arg
          %1: i8 = 4
          %2: i8 = 4
          %3: i1 = 1
          term [%0, %1]
        ",
        );
    }
}
