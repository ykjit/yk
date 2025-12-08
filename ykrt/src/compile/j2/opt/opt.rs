//! A draft trace optimiser. This is not tested, and is merely intended to help us understand what
//! APIs and so on we need to write a better trace optimiser. This will probably need to be
//! somewhat rethought when peeling is implemented.

use crate::compile::{
    CompilationError,
    j2::{
        hir::*,
        opt::{OptT, strength_fold::strength_fold},
    },
};
use index_vec::*;

pub(in crate::compile::j2) struct Opt {
    instkits: IndexVec<InstIdx, InstKit>,
    tys: IndexVec<TyIdx, Ty>,
}

impl Opt {
    pub(in crate::compile::j2) fn new() -> Self {
        Self {
            instkits: IndexVec::new(),
            tys: IndexVec::new(),
        }
    }

    /// Push `inst` into this optimisation module.
    pub(super) fn push_inst(&mut self, inst: Inst) -> InstIdx {
        self.instkits.push(InstKit {
            inst,
            range: Range::Unknown,
        })
    }

    /// Produce a rewritten version of `iidx`: this is a convenience function over [Self::rewrite].
    pub(super) fn inst_rewrite(&self, iidx: InstIdx) -> Inst {
        self.rewrite(self.inst(iidx).clone())
    }

    /// Rewrite `inst` to reflect knowledge the optimiser has built up (e.g. ranges) and then
    /// canonicalise.
    pub(super) fn rewrite(&self, inst: Inst) -> Inst {
        inst.map_iidxs(|iidx| self.map_iidx(iidx))
            .canonicalise(self, self)
    }

    /// Set `iidx`'s range to `range`.
    pub(super) fn set_range(&mut self, iidx: InstIdx, range: Range) {
        let instkit = self.instkits.get_mut(iidx).unwrap();
        match instkit.range {
            Range::Unknown => instkit.range = range,
            Range::Equivalent(cur_iidx) => match range {
                Range::Unknown => todo!(),
                Range::Equivalent(new_iidx) => {
                    if cur_iidx != new_iidx {
                        if let Inst::Const(_) = self.inst(cur_iidx) {
                            self.instkits.get_mut(new_iidx).unwrap().range =
                                Range::Equivalent(cur_iidx);
                        } else if let Inst::Const(_) = self.inst(new_iidx) {
                            self.instkits.get_mut(cur_iidx).unwrap().range =
                                Range::Equivalent(new_iidx);
                            self.instkits.get_mut(iidx).unwrap().range =
                                Range::Equivalent(new_iidx);
                        }
                    }
                }
            },
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
        &self.instkits[usize::from(idx)].inst
    }
}

impl OptT for Opt {
    fn build(self: Box<Self>) -> (Block, IndexVec<TyIdx, Ty>) {
        (
            Block {
                insts: self
                    .instkits
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

    fn map_iidx(&self, iidx: InstIdx) -> InstIdx {
        let mut search = iidx;
        loop {
            match self.instkits[search].range {
                Range::Unknown => return search,
                Range::Equivalent(other) => search = other,
            }
        }
    }

    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_ne!(*inst.ty(self), Ty::Void);
        let inst = self.rewrite(inst);
        match strength_fold(self, inst) {
            OptOutcome::NotNeeded => panic!(),
            OptOutcome::Rewritten(inst) => Ok(self.push_inst(inst)),
            OptOutcome::ReducedTo(iidx) => Ok(iidx),
        }
    }

    fn feed_void(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(*inst.ty(self), Ty::Void);
        let inst = self.rewrite(inst);
        match strength_fold(self, inst) {
            OptOutcome::NotNeeded => Ok(None),
            OptOutcome::Rewritten(inst) => Ok(Some(self.push_inst(inst))),
            OptOutcome::ReducedTo(iidx) => Ok(Some(iidx)),
        }
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
    ReducedTo(InstIdx),
}

/// What we know about a given [InstIdx].
struct InstKit {
    /// The [Inst] at a given [InstIdx].
    inst: Inst,
    /// The current range of values know for [InstIdx].
    range: Range,
}

pub(super) enum Range {
    Unknown,
    #[allow(unused)]
    Equivalent(InstIdx),
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
        let mut opt_map = IndexVec::with_capacity(insts.len());
        for inst in insts.into_iter() {
            let inst = inst.map_iidxs(|x| opt_map[x]);
            match opt_f(&mut opt, inst) {
                OptOutcome::NotNeeded => (),
                OptOutcome::Rewritten(inst) => {
                    opt_map.push(opt.push_inst(inst));
                }
                OptOutcome::ReducedTo(iidx) => {
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
                |opt, inst| strength_fold(opt, opt.rewrite(inst)),
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
          blackbox %4
          blackbox %4
          exit [%4]
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
