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
//! ## How passes should use the optimiser
//!
//! Passes will be passed an already-rewritten [Inst]. When looking up other instructions, one
//! should use [Opt::inst_rewrite] to obtain "old" [Inst]s: this automatically rewrites those older
//! instructions given the current state of knowledge.

use crate::compile::{
    CompilationError,
    j2::{
        hir::*,
        opt::{OptT, strength_fold::strength_fold},
    },
};
use index_vec::*;

pub(in crate::compile::j2) struct Opt {
    insts: IndexVec<InstIdx, InstEquiv>,
    tys: IndexVec<TyIdx, Ty>,
}

impl Opt {
    pub(in crate::compile::j2) fn new() -> Self {
        Self {
            insts: IndexVec::new(),
            tys: IndexVec::new(),
        }
    }

    /// Used by [Self::feed] and [Self::feed_void].
    fn feed_internal(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        let inst = self.rewrite(inst);
        match strength_fold(self, inst) {
            OptOutcome::NotNeeded => Ok(None),
            OptOutcome::Rewritten(inst) => Ok(Some(self.push_inst(inst))),
            OptOutcome::Equiv(iidx) => Ok(Some(iidx)),
        }
    }

    /// Push `inst` into this optimisation module.
    fn push_inst(&mut self, inst: Inst) -> InstIdx {
        self.insts.push(InstEquiv {
            inst,
            equiv: InstIdx::MAX,
        })
    }

    /// Produce a rewritten version of `iidx`: this is a convenience function over [Self::rewrite].
    pub(super) fn inst_rewrite(&mut self, iidx: InstIdx) -> Inst {
        self.rewrite(self.inst(iidx).clone())
    }

    /// Rewrite `inst` to reflect knowledge the optimiser has built up (e.g. ranges) and then
    /// canonicalise. In general, passes should not be using this function directly: they should be
    /// passing instruction indexes to [Self::inst_rewrite].
    pub(super) fn rewrite(&mut self, inst: Inst) -> Inst {
        let mut inst = inst.map(|iidx| self.equiv_iidx(iidx));
        inst.canonicalise(self);
        inst
    }

    /// Henceforth consider `iidx` to be equivalent to `equiv_to` (and/or vice versa). Note: it is
    /// the caller's job to ensure that `iidx` and `equiv_to` have already been transformed for
    /// equivalence.
    pub(super) fn set_equiv(&mut self, iidx: InstIdx, equiv_to: InstIdx) {
        assert_eq!(iidx, self.equiv_iidx(iidx));
        assert_eq!(equiv_to, self.equiv_iidx(equiv_to));
        match (self.inst(iidx), self.inst(equiv_to)) {
            (Inst::Const(_), Inst::Const(_)) => (),
            (_, Inst::Const(_)) => self.insts.get_mut(iidx).unwrap().equiv = equiv_to,
            (_, _) => self.insts.get_mut(equiv_to).unwrap().equiv = iidx,
        }
    }
}

impl ModLikeT for Opt {
    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        panic!("Not available in optimiser");
    }

    fn ty(&self, tyidx: TyIdx) -> &Ty {
        &self.tys[tyidx]
    }
}

impl BlockLikeT for Opt {
    fn inst(&self, idx: InstIdx) -> &Inst {
        &self.insts[usize::from(idx)].inst
    }
}

impl OptT for Opt {
    fn build(self: Box<Self>) -> (Block, IndexVec<TyIdx, Ty>) {
        (
            Block {
                insts: self
                    .insts
                    .into_iter()
                    .map(|x| x.inst)
                    .collect::<IndexVec<_, _>>(),
            },
            self.tys,
        )
    }

    fn peel(self) -> (Block, Block) {
        todo!()
    }

    fn equiv_iidx(&mut self, iidx: InstIdx) -> InstIdx {
        let mut search = iidx;
        loop {
            let equiv = self.insts[search].equiv;
            if equiv == InstIdx::MAX {
                if search != iidx {
                    self.insts[iidx].equiv = search;
                }
                return search;
            }
            search = equiv;
        }
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
        Ok(self.tys.push(ty))
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

    pub(in crate::compile::j2::opt) fn opt_and_test<F>(mod_s: &str, opt_f: F, ptn: &str)
    where
        F: Fn(&mut Opt, Inst) -> OptOutcome,
    {
        let m = str_to_mod::<TestReg>(mod_s);
        let mut opt = Box::new(Opt::new());
        opt.tys = m.tys;
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
        for inst in insts.into_iter() {
            let inst = inst.map(|x| opt_map[x]);
            match opt_f(&mut opt, inst) {
                OptOutcome::NotNeeded => (),
                OptOutcome::Rewritten(inst) => {
                    opt_map.push(opt.push_inst(inst));
                }
                OptOutcome::Equiv(iidx) => {
                    opt_map.push(iidx);
                }
            }
        }
        let (block, tys) = opt.build();
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
                |opt, inst| {
                    let inst = opt.rewrite(inst);
                    strength_fold(opt, inst)
                },
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
