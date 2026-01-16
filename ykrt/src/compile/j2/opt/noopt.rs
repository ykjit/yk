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
        self: Box<Self>,
    ) -> (
        Block,
        IndexVec<GuardExtraIdx, GuardExtra>,
        IndexVec<TyIdx, Ty>,
    ) {
        (Block { insts: self.insts }, self.guard_extras, self.tys)
    }

    fn peel(self) -> (Block, Block) {
        todo!()
    }

    fn feed(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_ne!(*inst.ty(self), Ty::Void);
        Ok(self.insts.push(inst))
    }

    fn feed_void(&mut self, inst: Inst) -> Result<Option<InstIdx>, CompilationError> {
        assert_eq!(*inst.ty(self), Ty::Void);
        assert!(!matches!(inst, Inst::Guard(_)));
        Ok(Some(self.insts.push(inst)))
    }

    fn feed_arg(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        assert_matches!(inst, Inst::Arg(_) | Inst::Const(_));
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
