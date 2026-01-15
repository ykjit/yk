//! A non-optimising "optimiser".
//!
//! This will be used when `YKD_OPT=0` and, perhaps, for some testing purposes.

use crate::compile::{
    CompilationError,
    j2::{
        hir::*,
        opt::{EquivIIdxT, OptT},
    },
};
use index_vec::*;
use std::{assert_matches::assert_matches, collections::HashMap};

pub(in crate::compile::j2) struct NoOpt {
    insts: IndexVec<InstIdx, Inst>,
    guard_extras: IndexVec<GuardExtraIdx, GuardExtra>,
    tys: IndexVec<TyIdx, Ty>,
    /// ty_map is used to ensure that only distinct [Ty]s lead to new [TyIdx]s.
    ty_map: HashMap<Ty, TyIdx>,
}

impl NoOpt {
    pub(in crate::compile::j2) fn new() -> Self {
        Self {
            insts: IndexVec::new(),
            guard_extras: IndexVec::new(),
            tys: IndexVec::new(),
            ty_map: HashMap::new(),
        }
    }
}

impl ModLikeT for NoOpt {
    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        panic!("Not available in optimiser");
    }

    fn gextra(&self, _geidx: GuardExtraIdx) -> &GuardExtra {
        todo!();
    }

    fn gextra_mut(&mut self, _geidx: GuardExtraIdx) -> &mut GuardExtra {
        todo!();
    }

    fn ty(&self, tyidx: TyIdx) -> &Ty {
        &self.tys[tyidx]
    }
}

impl BlockLikeT for NoOpt {
    fn inst(&self, idx: InstIdx) -> &Inst {
        &self.insts[usize::from(idx)]
    }
}

impl OptT for NoOpt {
    fn build(
        mut self: Box<Self>,
    ) -> Result<
        (
            Block,
            IndexVec<GuardExtraIdx, GuardExtra>,
            IndexVec<GuardBlockIdx, Block>,
            IndexVec<TyIdx, Ty>,
        ),
        CompilationError,
    > {
        let mut guards = IndexVec::with_capacity(self.guard_extras.len());
        // Because we update the `GuardExtra` at the end of each iteration, the borrow checker
        // won't let us iterate over the `guard_extras` directly _and_ call `self.inst`, so we have
        // to iterate over the indices.
        for i in 0..self.guard_extras.len() {
            let mut ginsts = IndexVec::with_capacity(self.guard_extras[i].exit_vars.len());
            for iidx in &self.guard_extras[i].exit_vars {
                match self.inst(*iidx) {
                    Inst::Const(x) => {
                        ginsts.push(x.clone().into());
                    }
                    Inst::Guard(_) => panic!(),
                    Inst::Term(_) => panic!(),
                    x => {
                        ginsts.push(Inst::Arg(Arg {
                            tyidx: *self.ty_map.get(x.ty(&*self)).unwrap(),
                        }));
                    }
                }
            }
            ginsts.push(Inst::Term(Term(
                (0..self.guard_extras[i].exit_vars.len())
                    .map(InstIdx::from)
                    .collect::<Vec<_>>(),
            )));
            self.guard_extras[i].gbidx = Some(guards.push(Block { insts: ginsts }));
        }
        Ok((
            Block { insts: self.insts },
            self.guard_extras,
            guards,
            self.tys,
        ))
    }

    fn peel(self) -> (Block, Block) {
        todo!()
    }

    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_ne!(*inst.ty(self), Ty::Void);
        assert!(!matches!(inst, Inst::Guard(_)));
        Ok(self.insts.push(inst))
    }

    fn feed_void(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(*inst.ty(self), Ty::Void);
        assert!(!matches!(inst, Inst::Guard(_)));
        Ok(Some(self.insts.push(inst)))
    }

    fn feed_arg(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_matches!(inst, Inst::Arg(_) | Inst::Const(_));
        assert!(!matches!(inst, Inst::Guard(_)));
        self.feed(inst)
    }

    fn feed_guard(
        &mut self,
        mut inst: Guard,
        gextra: GuardExtra,
    ) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(inst.geidx, GuardExtraIdx::MAX);
        inst.geidx = self.guard_extras.push(gextra);
        Ok(Some(self.insts.push(inst.into())))
    }

    fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        if let Some(tyidx) = self.ty_map.get(&ty) {
            return Ok(*tyidx);
        }
        let tyidx = self.tys.push(ty.clone());
        self.ty_map.insert(ty, tyidx);
        Ok(tyidx)
    }
}

impl EquivIIdxT for NoOpt {
    fn equiv_iidx(&self, iidx: InstIdx) -> InstIdx {
        iidx
    }
}
