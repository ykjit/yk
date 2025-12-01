//! A draft trace optimiser. This is not tested, and is merely intended to help us understand what
//! APIs and so on we need to write a better trace optimiser. This will probably need to be
//! somewhat rethought when peeling is implemented.

use crate::compile::{
    CompilationError,
    j2::{hir::*, opt::OptT},
    jitc_yk::arbbitint::ArbBitInt,
};
use index_vec::*;

pub(in crate::compile::j2) struct Opt {
    insts: IndexVec<InstIdx, Inst>,
    ranges: IndexVec<InstIdx, Range>,
    tys: IndexVec<TyIdx, Ty>,
}

impl Opt {
    pub(in crate::compile::j2) fn new() -> Self {
        Self {
            insts: IndexVec::new(),
            ranges: IndexVec::new(),
            tys: IndexVec::new(),
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
        &self.insts[usize::from(idx)]
    }
}

impl OptT for Opt {
    fn build(self: Box<Self>) -> (Block, IndexVec<TyIdx, Ty>) {
        (Block { insts: self.insts }, self.tys)
    }

    fn peel(self) -> (Block, Block) {
        todo!()
    }

    fn map_iidx(&self, iidx: InstIdx) -> InstIdx {
        match self.ranges[iidx] {
            Range::Unknown => iidx,
            Range::Equivalent(other) => other,
        }
    }

    /// This can be called recursively.
    fn push_inst(&mut self, inst: Inst) -> Result<InstIdx, CompilationError> {
        let inst = inst
            .map_iidxs(|iidx| self.map_iidx(iidx))
            .canonicalise(self, self);

        match &inst {
            Inst::And(And { tyidx, lhs, rhs }) => {
                if let (
                    Inst::Const(Const { kind: lhs_kind, .. }),
                    Inst::Const(Const { kind: rhs_kind, .. }),
                ) = (&self.insts[*lhs], &self.insts[*rhs])
                {
                    let (ConstKind::Int(lhs_c), ConstKind::Int(rhs_c)) = (lhs_kind, rhs_kind)
                    else {
                        panic!()
                    };
                    self.ranges.push(Range::Unknown);
                    return Ok(self.insts.push(Inst::Const(Const {
                        tyidx: *tyidx,
                        kind: ConstKind::Int(lhs_c.bitand(rhs_c)),
                    })));
                }
            }
            Inst::DynPtrAdd(DynPtrAdd {
                ptr,
                num_elems,
                elem_size,
            }) => {
                if let Inst::Const(Const { tyidx: _, kind }) = &self.insts[*num_elems] {
                    let ConstKind::Int(nelems) = kind else {
                        panic!()
                    };
                    let v = nelems.to_sign_ext_isize().unwrap();
                    let off =
                        i32::try_from(v.checked_mul(isize::try_from(*elem_size).unwrap()).unwrap())
                            .unwrap();
                    self.ranges.push(Range::Unknown);
                    if off == 0 {
                        return Ok(*ptr);
                    } else {
                        return self.push_inst(Inst::PtrAdd(PtrAdd {
                            ptr: *ptr,
                            off,
                            in_bounds: false,
                            nusw: false,
                            nuw: false,
                        }));
                    }
                }
            }
            Inst::Guard(Guard {
                expect,
                cond,
                bid: _,
                entry_vars: _,
                gridx: _,
                switch: _,
            }) => {
                if let Inst::Const(_) = &self.insts[*cond] {
                    return Ok(*cond);
                }
                if *expect
                    && let Inst::ICmp(ICmp {
                        lhs,
                        rhs,
                        pred: IPred::Eq,
                        ..
                    }) = &self.insts[*cond]
                    && let Inst::Const(_) = &self.insts[*rhs]
                {
                    match &self.ranges[*lhs] {
                        Range::Unknown => self.ranges[*lhs] = Range::Equivalent(*rhs),
                        Range::Equivalent(x) => {
                            if x != rhs {
                                todo!("{x:?} {rhs:?}");
                            }
                        }
                    }
                }
            }
            Inst::ICmp(ICmp {
                pred: IPred::Eq,
                lhs,
                rhs,
                samesign: _,
            }) => {
                if let Inst::Const(Const { kind: lhs_kind, .. }) = &self.insts[*lhs]
                    && let Inst::Const(Const { kind: rhs_kind, .. }) = &self.insts[*rhs]
                    && lhs_kind == rhs_kind
                {
                    let dst_tyidx = self.push_ty(Ty::Int(1))?;
                    self.ranges.push(Range::Unknown);
                    return Ok(self.insts.push(Inst::Const(Const {
                        tyidx: dst_tyidx,
                        kind: ConstKind::Int(ArbBitInt::from_u64(1, 1)),
                    })));
                }
            }
            Inst::LShr(LShr {
                tyidx,
                lhs,
                rhs,
                exact: _,
            }) => {
                if let (
                    Inst::Const(Const { kind: lhs_kind, .. }),
                    Inst::Const(Const { kind: rhs_kind, .. }),
                ) = (&self.insts[*lhs], &self.insts[*rhs])
                {
                    let (ConstKind::Int(lhs_c), ConstKind::Int(rhs_c)) = (lhs_kind, rhs_kind)
                    else {
                        panic!()
                    };
                    // If checked_shr fails, we've encountered LLVM poison and can
                    // choose any value.
                    let lshr = lhs_c
                        .checked_lshr(rhs_c.to_zero_ext_u32().unwrap())
                        .unwrap_or_else(|| ArbBitInt::all_bits_set(lhs_c.bitw()));
                    self.ranges.push(Range::Unknown);
                    return Ok(self.insts.push(Inst::Const(Const {
                        tyidx: *tyidx,
                        kind: ConstKind::Int(lshr),
                    })));
                }
            }
            Inst::PtrAdd(PtrAdd {
                ptr,
                off,
                in_bounds,
                nusw,
                nuw,
            }) => {
                // LLVM semantics require pointer arithmetic to wrap as though they were "pointer
                // index typed".
                assert!(!in_bounds && !nusw && !nuw);
                if let Inst::Const(Const { tyidx, kind }) = &self.insts[*ptr] {
                    let ConstKind::Ptr(addr) = kind else { panic!() };
                    self.ranges.push(Range::Unknown);
                    return Ok(self.insts.push(Inst::Const(Const {
                        tyidx: *tyidx,
                        kind: ConstKind::Ptr(
                            addr.wrapping_add_signed(isize::try_from(*off).unwrap()),
                        ),
                    })));
                }
            }
            Inst::SExt(SExt { tyidx, val }) => {
                if let Inst::Const(Const {
                    tyidx: _src_tyidx,
                    kind,
                }) = &self.insts[*val]
                {
                    match kind {
                        ConstKind::Double(_) | ConstKind::Float(_) => unreachable!(),
                        ConstKind::Int(src_val) => {
                            let dst_bitw = self.tys[*tyidx].bitw();
                            let dst_val = src_val.sign_extend(dst_bitw);
                            let dst_tyidx = self.push_ty(Ty::Int(dst_bitw))?;
                            self.ranges.push(Range::Unknown);
                            return Ok(self.insts.push(Inst::Const(Const {
                                tyidx: dst_tyidx,
                                kind: ConstKind::Int(dst_val),
                            })));
                        }
                        ConstKind::Ptr(_) => todo!(),
                    }
                }
            }
            Inst::ZExt(ZExt { tyidx, val }) => {
                if let Inst::Const(Const {
                    tyidx: _src_tyidx,
                    kind,
                }) = &self.insts[*val]
                {
                    match kind {
                        ConstKind::Double(_) | ConstKind::Float(_) => unreachable!(),
                        ConstKind::Int(src_val) => {
                            let dst_bitw = self.tys[*tyidx].bitw();
                            let dst_val = src_val.zero_extend(dst_bitw);
                            let dst_tyidx = self.push_ty(Ty::Int(dst_bitw))?;
                            self.ranges.push(Range::Unknown);
                            return Ok(self.insts.push(Inst::Const(Const {
                                tyidx: dst_tyidx,
                                kind: ConstKind::Int(dst_val),
                            })));
                        }
                        ConstKind::Ptr(_) => todo!(),
                    }
                }
            }
            _ => (),
        }

        self.ranges.push(Range::Unknown);
        Ok(self.insts.push(inst))
    }

    fn push_ty(&mut self, ty: Ty) -> Result<TyIdx, CompilationError> {
        Ok(self.tys.push(ty))
    }
}

enum Range {
    Unknown,
    Equivalent(InstIdx),
}
