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
        match self.instkits[iidx].range {
            Range::Unknown => iidx,
            Range::Equivalent(other) => other,
        }
    }

    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        let inst = inst
            .map_iidxs(|iidx| self.map_iidx(iidx))
            .canonicalise(self, self);

        match strength_fold(self, inst) {
            OptOutcome::Rewritten(inst) | OptOutcome::Unchanged(inst) => Ok(self.push_inst(inst)),
            OptOutcome::ReducedTo(iidx) => Ok(iidx),
        }
    }

    fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        Ok(self.tys.push(ty))
    }
}

/// What an optimisation has managed to make of a given input [Inst].
pub(super) enum OptOutcome {
    /// The input [Inst] has been rewritten to a new [Inst].
    Rewritten(Inst),
    /// The input [Inst] is equivalent to [InstIdx].
    ReducedTo(InstIdx),
    /// The input [Inst] could not be optimised: it must be returned here so that it can be passed
    /// on to the next phase.
    Unchanged(Inst),
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
                OptOutcome::Rewritten(inst) | OptOutcome::Unchanged(inst) => {
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
}
