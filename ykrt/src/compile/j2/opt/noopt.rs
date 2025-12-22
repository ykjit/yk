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
use std::collections::HashMap;

pub(in crate::compile::j2) struct NoOpt {
    insts: IndexVec<InstIdx, Inst>,
    tys: IndexVec<TyIdx, Ty>,
    /// ty_map is used to ensure that only distinct [Ty]s lead to new [TyIdx]s.
    ty_map: HashMap<Ty, TyIdx>,
}

impl NoOpt {
    pub(in crate::compile::j2) fn new() -> Self {
        Self {
            insts: IndexVec::new(),
            tys: IndexVec::new(),
            ty_map: HashMap::new(),
        }
    }
}

impl ModLikeT for NoOpt {
    fn addr_to_name(&self, _addr: usize) -> Option<&str> {
        panic!("Not available in optimiser");
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
    fn build(self: Box<Self>) -> (Block, IndexVec<TyIdx, Ty>) {
        (Block { insts: self.insts }, self.tys)
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
        Ok(Some(self.insts.push(inst)))
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
