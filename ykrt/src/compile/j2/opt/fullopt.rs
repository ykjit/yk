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
//!
//! ## Structure of the optimiser
//!
//! The optimiser consists of an "outward facing to [aot_to_hir]" part ([FullOpt], implementing
//! [OptT]) and in "inward facing to passes" part ([PassOpt]). What links these two things is
//! [OptInternal]: the outer and inner parts both have access to this, passing ownership between
//! themselves as appropriate.
//!
//!
//! ## Preinstructions
//!
//! The optimiser API is deliberately simple: an instruction goes into the optimisation chain and
//! passes operate on it. At the end it is either: committed to the trace; determined to be
//! equivalent to another instruction; or shown not to be needed at all.
//!
//! However, sometimes when transforming an instruction one needs to ensure that other instructions
//! (typically constants) are present in the trace. This does not fit with the "optimise one
//! instruction" model. The [PassOpt] API thus has the concept of "preinstructions": that is,
//! instructions which will be committed to the trace before the current instruction.
//! Preinstructions should be used carefully:
//!
//! 1. They are only committed at the _end_ of the current pass. In other words, if a pass P calls
//!    [PassOpt::push_pre_inst] then those instructions are not accessible via [PassOpt::inst] for
//!    the duration of P. As soon as P has completed, the preinstructions will be committed, and
//!    subsequent passes will have access to them. Note: it is guaranteed that the [InstIdx]s
//!    returned by [PassOPt::push_pre_inst] will be valid after the preinstructions are
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

use crate::compile::{
    CompilationError,
    j2::{
        hir::*,
        opt::{EquivIIdxT, OptT, cse::CSE, strength_fold::StrengthFold},
    },
};
use index_vec::*;
use smallvec::SmallVec;
use std::collections::HashMap;

////////////////////////////////////////////////////////////////////////////////
// The external-facing part of the optimiser.

pub(in crate::compile::j2) struct FullOpt {
    /// The ordered set of optimisation passes that all instructions will be fed through.
    passes: [Box<dyn PassT>; 2],
    inner: OptInternal,
}

impl FullOpt {
    pub(in crate::compile::j2) fn new() -> Self {
        Self {
            passes: [Box::new(StrengthFold::new()), Box::new(CSE::new())],
            inner: OptInternal {
                insts: IndexVec::new(),
                tys: IndexVec::new(),
                ty_map: HashMap::new(),
            },
        }
    }

    /// Used by [Self::feed] and [Self::feed_void].
    fn feed_internal(&mut self, mut inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        for i in 0..self.passes.len() {
            let mut opt = PassOpt {
                inner: &mut self.inner,
                pre_insts: SmallVec::new(),
            };

            match self.passes[i].feed(&mut opt, inst) {
                OptOutcome::NotNeeded => return Ok(None),
                OptOutcome::Rewritten(new_inst) => inst = new_inst,
                OptOutcome::Equiv(iidx) => return Ok(Some(iidx)),
            }

            for inst in opt.pre_insts {
                self.commit_inst(inst);
            }
        }
        Ok(Some(self.commit_inst(inst)))
    }

    /// Commit `inst` to this trace.
    fn commit_inst(&mut self, inst: Inst) -> InstIdx {
        let opt = CommitInstOpt { inner: &self.inner };
        let iidx = self.inner.insts.len_idx();
        for pass in &mut self.passes {
            pass.inst_committed(&opt, iidx, &inst);
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
}

impl BlockLikeT for FullOpt {
    fn inst(&self, idx: InstIdx) -> &Inst {
        &self.inner.insts[usize::from(idx)].inst
    }
}

impl OptT for FullOpt {
    fn build(self: Box<Self>) -> (Block, IndexVec<TyIdx, Ty>) {
        (
            Block {
                insts: self
                    .inner
                    .insts
                    .into_iter()
                    .map(|x| x.inst)
                    .collect::<IndexVec<_, _>>(),
            },
            self.inner.tys,
        )
    }

    fn peel(self) -> (Block, Block) {
        todo!()
    }

    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_ne!(*inst.ty(self), Ty::Void);
        self.feed_internal(inst).map(|x| x.unwrap())
    }

    fn feed_void(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(*inst.ty(self), Ty::Void);
        self.feed_internal(inst)
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
    tys: IndexVec<TyIdx, Ty>,
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

/// What an optimisation has managed to make of a given input [Inst].
#[derive(Debug)]
pub(super) enum OptOutcome {
    NotNeeded,
    /// The input [Inst] has been rewritten to a new [Inst].
    Rewritten(Inst),
    /// The input [Inst] is equivalent to [InstIdx].
    Equiv(InstIdx),
}

/// A wrapper around an [Inst] and our current knowledge of another instruction it is equivalent
/// to.
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
}

/// The object passed to [PassT::feed] so that they can interact with the optimiser.
pub(super) struct PassOpt<'a> {
    inner: &'a mut OptInternal,
    pre_insts: SmallVec<[Inst; 1]>,
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
        self.inner.push_ty(ty)
    }

    /// Add the "pre-instruction" `preinst`: when the current pass has completed, `inst` will be
    /// committed to the trace, and immediately available to subsequent passes. The [InstIdx] that
    /// is returned will be `preinst`'s [InstIdx] _after_ it is committed. In other words, during
    /// the current pass, [InstIdx] is invalid, and attempting any operations with it will lead to
    /// undefined behaviour.
    pub(super) fn push_pre_inst(&mut self, preinst: Inst) -> InstIdx {
        let iidx = InstIdx::from_usize(self.inner.insts.len() + self.pre_insts.len());
        self.pre_insts.push(preinst);
        iidx
    }

    /// Henceforth consider `iidx` to be equivalent to `equiv_to` (and/or vice versa). Note: it is
    /// the caller's job to ensure that `iidx` and `equiv_to` have already been transformed for
    /// equivalence.
    pub(super) fn set_equiv(&mut self, iidx: InstIdx, equiv_to: InstIdx) {
        assert_eq!(iidx, self.equiv_iidx(iidx));
        assert_eq!(equiv_to, self.equiv_iidx(equiv_to));
        match (self.inst(iidx), self.inst(equiv_to)) {
            (Inst::Const(_), Inst::Const(_)) => (),
            (_, Inst::Const(_)) => self.inner.insts.get_mut(iidx).unwrap().equiv = equiv_to,
            (_, _) => self.inner.insts.get_mut(equiv_to).unwrap().equiv = iidx,
        }
    }
}

impl BlockLikeT for PassOpt<'_> {
    fn inst(&self, iidx: InstIdx) -> &Inst {
        self.inner.inst(iidx)
    }
}

impl ModLikeT for PassOpt<'_> {
    fn ty(&self, tyidx: TyIdx) -> &Ty {
        self.inner.ty(tyidx)
    }

    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        todo!()
    }
}

impl EquivIIdxT for PassOpt<'_> {
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx {
        self.inner.equiv_iidx(iidx)
    }
}

/// The object passed to [PassT::inst_committed] so that they can interact with the optimiser.
pub(super) struct CommitInstOpt<'a> {
    inner: &'a OptInternal,
}

impl BlockLikeT for CommitInstOpt<'_> {
    fn inst(&self, iidx: InstIdx) -> &Inst {
        self.inner.inst(iidx)
    }
}

impl ModLikeT for CommitInstOpt<'_> {
    fn ty(&self, tyidx: TyIdx) -> &Ty {
        self.inner.ty(tyidx)
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
pub(in crate::compile::j2::opt) mod test {
    use super::*;
    use crate::compile::j2::{hir_parser::str_to_mod, regalloc::test::TestReg};
    use fm::FMBuilder;
    use index_vec::IndexVec;
    use lazy_static::lazy_static;
    use regex::Regex;

    lazy_static! {
        static ref PTN_RE: Regex = Regex::new(r"\{\{.+?\}\}").unwrap();
        static ref PTN_RE_IGNORE: Regex = Regex::new(r"\{\{_}\}").unwrap();
        static ref TEXT_RE: Regex = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
    }

    pub(in crate::compile::j2::opt) fn opt_and_test<F, G>(
        mod_s: &str,
        feed_f: F,
        committed_f: G,
        ptn: &str,
    ) where
        for<'a> F: Fn(&'a mut PassOpt, Inst) -> OptOutcome,
        for<'a> G: Fn(&'a CommitInstOpt, InstIdx, &Inst),
    {
        let m = str_to_mod::<TestReg>(mod_s);
        let mut fopt = Box::new(FullOpt::new());
        fopt.inner.tys = m.tys;
        let ModKind::Test {
            entry_vlocs,
            block: Block { insts },
        } = m.kind
        else {
            panic!()
        };
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
        for mut inst in insts.into_iter() {
            inst.rewrite_iidxs(|x| opt_map[x]);
            let mut opt = PassOpt {
                inner: &mut fopt.inner,
                pre_insts: SmallVec::new(),
            };
            match feed_f(&mut opt, inst) {
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

            for inst in opt.pre_insts {
                let iidx = fopt.inner.insts.len_idx();
                let opt = CommitInstOpt { inner: &fopt.inner };
                committed_f(&opt, iidx, &inst);
                fopt.inner.insts.push(InstEquiv {
                    inst,
                    equiv: InstIdx::MAX,
                });
            }

            let iidx = fopt.inner.insts.len_idx();
            opt_map.push(iidx);
            let opt = CommitInstOpt { inner: &fopt.inner };
            committed_f(&opt, iidx, &inst);
            fopt.inner.insts.push(InstEquiv {
                inst,
                equiv: InstIdx::MAX,
            });
        }
        let (block, tys) = fopt.build();
        let m = Mod {
            trid: m.trid,
            kind: ModKind::Test { entry_vlocs, block },
            tys,
            guard_restores: IndexVec::new(),
            addr_name_map: None,
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
            opt_and_test(
                mod_s,
                |opt, mut inst| {
                    inst.canonicalise(opt);
                    StrengthFold::new().feed(opt, inst)
                },
                |_, _, _| (),
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
          exit [%0]
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
          blackbox %2
          blackbox %4
          exit [%2]
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
          exit [%0, %1]
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
          exit [%3, %3]
        ",
        );
    }
}
