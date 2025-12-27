//! Strength reduction and constant folding

use crate::compile::{
    j2::{
        hir::*,
        opt::{
            EquivIIdxT,
            fullopt::{CommitInstOpt, OptOutcome, PassOpt, PassT},
        },
    },
    jitc_yk::arbbitint::ArbBitInt,
};

pub(super) struct StrengthFold;

impl StrengthFold {
    pub(super) fn new() -> Self {
        Self
    }
}

impl PassT for StrengthFold {
    fn feed(&mut self, opt: &mut PassOpt, inst: Inst) -> OptOutcome {
        match inst {
            Inst::Abs(x) => opt_abs(opt, x),
            Inst::AShr(x) => opt_ashr(opt, x),
            Inst::Add(x) => opt_add(opt, x),
            Inst::And(x) => opt_and(opt, x),
            Inst::CtPop(x) => opt_ctpop(opt, x),
            Inst::DynPtrAdd(x) => opt_dynptradd(opt, x),
            Inst::FAdd(x) => opt_fadd(opt, x),
            Inst::FDiv(x) => opt_fdiv(opt, x),
            Inst::FMul(x) => opt_fmul(opt, x),
            Inst::FSub(x) => opt_fsub(opt, x),
            Inst::Guard(x) => opt_guard(opt, x),
            Inst::ICmp(x) => opt_icmp(opt, x),
            Inst::IntToPtr(x) => opt_inttoptr(opt, x),
            Inst::LShr(x) => opt_lshr(opt, x),
            Inst::MemCpy(x) => opt_memcpy(opt, x),
            Inst::Mul(x) => opt_mul(opt, x),
            Inst::Or(x) => opt_or(opt, x),
            Inst::PtrAdd(x) => opt_ptradd(opt, x),
            Inst::PtrToInt(x) => opt_ptrtoint(opt, x),
            Inst::Select(x) => opt_select(opt, x),
            Inst::SExt(x) => opt_sext(opt, x),
            Inst::Shl(x) => opt_shl(opt, x),
            Inst::Sub(x) => opt_sub(opt, x),
            Inst::Trunc(x) => opt_trunc(opt, x),
            Inst::UDiv(x) => opt_udiv(opt, x),
            Inst::Xor(x) => opt_xor(opt, x),
            Inst::ZExt(x) => opt_zext(opt, x),
            _ => OptOutcome::Rewritten(inst),
        }
    }

    fn inst_committed(&mut self, _opt: &CommitInstOpt, _iidx: InstIdx, _inst: &Inst) {}
}

fn opt_abs(opt: &mut PassOpt, mut inst: Abs) -> OptOutcome {
    inst.canonicalise(opt);
    let Abs {
        tyidx,
        val,
        int_min_poison,
    } = inst;
    assert!(int_min_poison);
    if let Some(ConstKind::Int(val_c)) = opt.as_constkind(val) {
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(val_c.bitabs()),
        }));
    }
    OptOutcome::Rewritten(inst.into())
}

fn opt_add(opt: &mut PassOpt, mut inst: Add) -> OptOutcome {
    inst.canonicalise(opt);
    let Add {
        tyidx,
        lhs,
        rhs,
        nuw,
        nsw,
    } = inst;
    assert!(!nuw && !nsw);

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 + c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(lhs_c.wrapping_add(&rhs_c)),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x + 0` to `x`.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_and(opt: &mut PassOpt, mut inst: And) -> OptOutcome {
    inst.canonicalise(opt);
    let And { tyidx, lhs, rhs } = inst;
    if lhs == rhs {
        // Reduce x & x with x.
        return OptOutcome::Equiv(lhs);
    }

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 & c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(lhs_c.bitand(&rhs_c)),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x & 0` to `0`.
                return OptOutcome::Equiv(rhs);
            }
            if rhs_c == ArbBitInt::all_bits_set(rhs_c.bitw()) {
                // Reduce `x & y` to `x` if `y` is a constant that has all
                // the necessary bits set for this integer type. For an i1, for
                // example, `x & 1` can be replaced with `x`.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_ashr(opt: &mut PassOpt, mut inst: AShr) -> OptOutcome {
    inst.canonicalise(opt);
    let AShr {
        tyidx,
        lhs,
        rhs,
        exact,
    } = inst;
    opt_ashr_lshr(opt, inst.into(), tyidx, lhs, rhs, exact, |lhs_c, rhs_c| {
        lhs_c.checked_ashr(rhs_c.to_zero_ext_u32().unwrap())
    })
}

/// Optimise the common parts of `ashr` and `lshr`, outsourcing the difference between the two to
/// `f`.
fn opt_ashr_lshr<F>(
    opt: &mut PassOpt,
    inst: Inst,
    tyidx: TyIdx,
    lhs: InstIdx,
    rhs: InstIdx,
    exact: bool,
    f: F,
) -> OptOutcome
where
    F: FnOnce(&ArbBitInt, &ArbBitInt) -> Option<ArbBitInt>,
{
    assert!(!exact);

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `lhs_c >> rhs_c`.
            let c = f(&lhs_c, &rhs_c).unwrap_or_else(|| ArbBitInt::all_bits_set(lhs_c.bitw()));
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(c),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x >> 0` to `x`.
                return OptOutcome::Equiv(lhs);
            }
        }
        (Some(ConstKind::Int(lhs_c)), _) => {
            if lhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `0 >> x` to `0`.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst)
}

fn opt_ctpop(opt: &mut PassOpt, mut inst: CtPop) -> OptOutcome {
    inst.canonicalise(opt);
    let CtPop { tyidx, val } = inst;
    if let Some(ConstKind::Int(c)) = opt.as_constkind(val) {
        // LLVM's ctpop has a polymorphic return type: since the maximum number of bits we can
        // represent in LLVM IR is 2^23, and `count_ones` returns a `u32`, using
        // `ArbBitInt::from_u64` is always safe.
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(c.bitw(), u64::from(c.count_ones()))),
        }));
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_dynptradd(opt: &mut PassOpt, mut inst: DynPtrAdd) -> OptOutcome {
    inst.canonicalise(opt);
    let DynPtrAdd {
        ptr,
        num_elems,
        elem_size,
    } = inst;
    if let Some(ConstKind::Int(c)) = opt.as_constkind(num_elems) {
        // LLVM IR semantics are such that GEP indices are sign-extended or truncated to the
        // "pointer index size" (which for address space zero is a pointer-sized integer). First
        // make sure we will be operating on that type.
        let v = c.to_sign_ext_isize().unwrap();
        // In LLVM slient two's compliment wrapping is permitted, but in Rust a `unchecked_mul()`
        // that wraps is UB. It seems unlikely that the overflow case will actually happen, so we
        // can cross that bridge if we come to it.
        let off = v.checked_mul(isize::try_from(elem_size).unwrap()).unwrap();
        let off = i32::try_from(off).unwrap();
        if off == 0 {
            return OptOutcome::Equiv(ptr);
        } else {
            // We've optimised to a `ptradd`, so run it through that pass, which may be able to
            // optimise it further.
            return opt_ptradd(
                opt,
                PtrAdd {
                    ptr,
                    off,
                    in_bounds: false,
                    nusw: false,
                    nuw: false,
                },
            );
        }
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_fadd(opt: &mut PassOpt, mut inst: FAdd) -> OptOutcome {
    inst.canonicalise(opt);
    let FAdd { tyidx, lhs, rhs } = inst;

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Float(lhs_c)), Some(ConstKind::Float(rhs_c))) => {
            // Constant fold `c1f + c2f`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Float(lhs_c + rhs_c),
            }));
        }
        (Some(ConstKind::Double(lhs_c)), Some(ConstKind::Double(rhs_c))) => {
            // Constant fold `c1d + c2d`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Double(lhs_c + rhs_c),
            }));
        }
        // Note: rhs = 0.0 is not safe to eliminate in general, due to IEE754.
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_fdiv(opt: &mut PassOpt, mut inst: FDiv) -> OptOutcome {
    inst.canonicalise(opt);
    let FDiv { tyidx, lhs, rhs } = inst;

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Float(lhs_c)), Some(ConstKind::Float(rhs_c))) => {
            // Constant fold `c1f / c2f`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Float(lhs_c / rhs_c),
            }));
        }
        (Some(ConstKind::Double(lhs_c)), Some(ConstKind::Double(rhs_c))) => {
            // Constant fold `c1d / c2d`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Double(lhs_c / rhs_c),
            }));
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_fmul(opt: &mut PassOpt, mut inst: FMul) -> OptOutcome {
    inst.canonicalise(opt);
    let FMul { tyidx, lhs, rhs } = inst;

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Float(lhs_c)), Some(ConstKind::Float(rhs_c))) => {
            // Constant fold `c1f * c2f`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Float(lhs_c * rhs_c),
            }));
        }
        (Some(ConstKind::Double(lhs_c)), Some(ConstKind::Double(rhs_c))) => {
            // Constant fold `c1d * c2d`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Double(lhs_c * rhs_c),
            }));
        }
        // Note: rhs = 0.0 is not safe to eliminate in general, due to IEE754.
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_fsub(opt: &mut PassOpt, mut inst: FSub) -> OptOutcome {
    inst.canonicalise(opt);
    let FSub { tyidx, lhs, rhs } = inst;

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Float(lhs_c)), Some(ConstKind::Float(rhs_c))) => {
            // Constant fold `c1f - c2f`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Float(lhs_c - rhs_c),
            }));
        }
        (Some(ConstKind::Double(lhs_c)), Some(ConstKind::Double(rhs_c))) => {
            // Constant fold `c1d - c2d`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Double(lhs_c - rhs_c),
            }));
        }
        // Note: rhs = 0.0 is not safe to eliminate in general, due to IEE754.
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_guard(opt: &mut PassOpt, mut inst @ Guard { expect, cond, .. }: Guard) -> OptOutcome {
    // Since guards tend to have lots of operands, we avoid `canonicalising` unless we really need
    // to. This needs to be done carefully, because after we've called `set_equiv` below,
    // recanonicalising the guard would change the `entry_vars` in a semantically incorrect way.

    let cond = opt.equiv_iidx(cond);
    if let Inst::Const(_) = opt.inst(cond) {
        // A guard that references a constant is, by definition, not needed and
        // doesn't affect future analyses.
        return OptOutcome::NotNeeded;
    }

    if expect
        && let cond_inst @ Inst::ICmp(ICmp {
            pred: IPred::Eq, ..
        }) = opt.inst(cond)
    {
        let mut cond_inst = cond_inst.to_owned();
        cond_inst.canonicalise(opt);
        let Inst::ICmp(ICmp {
            pred: IPred::Eq,
            lhs,
            rhs,
            samesign,
        }) = cond_inst
        else {
            panic!()
        };
        assert!(!samesign);
        if lhs == rhs {
            return OptOutcome::NotNeeded;
        }
        inst.canonicalise(opt);
        opt.set_equiv(lhs, rhs);
    } else {
        inst.canonicalise(opt);
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_icmp(opt: &mut PassOpt, mut inst: ICmp) -> OptOutcome {
    inst.canonicalise(opt);
    let ICmp {
        pred,
        lhs,
        rhs,
        samesign,
    } = inst;
    assert!(!samesign);
    if let (Some(lhs_c), Some(rhs_c)) = (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        // Constant fold.
        let v = match (lhs_c, rhs_c) {
            (ConstKind::Int(lhs_c), ConstKind::Int(rhs_c)) => match pred {
                IPred::Eq => lhs_c == rhs_c,
                IPred::Ne => lhs_c != rhs_c,
                IPred::Ugt => lhs_c.to_zero_ext_u64() > rhs_c.to_zero_ext_u64(),
                IPred::Uge => lhs_c.to_zero_ext_u64() >= rhs_c.to_zero_ext_u64(),
                IPred::Ult => lhs_c.to_zero_ext_u64() < rhs_c.to_zero_ext_u64(),
                IPred::Ule => lhs_c.to_zero_ext_u64() <= rhs_c.to_zero_ext_u64(),
                IPred::Sgt => lhs_c.to_sign_ext_i64() > rhs_c.to_sign_ext_i64(),
                IPred::Sge => lhs_c.to_sign_ext_i64() >= rhs_c.to_sign_ext_i64(),
                IPred::Slt => lhs_c.to_sign_ext_i64() < rhs_c.to_sign_ext_i64(),
                IPred::Sle => lhs_c.to_sign_ext_i64() <= rhs_c.to_sign_ext_i64(),
            },
            (ConstKind::Ptr(lhs_c), ConstKind::Ptr(rhs_c)) => match pred {
                IPred::Eq => lhs_c == rhs_c,
                IPred::Ne => lhs_c != rhs_c,
                IPred::Ugt => lhs_c > rhs_c,
                IPred::Uge => lhs_c >= rhs_c,
                IPred::Ult => lhs_c < rhs_c,
                IPred::Ule => lhs_c <= rhs_c,
                IPred::Sgt | IPred::Sge | IPred::Slt | IPred::Sle => unreachable!(),
            },
            _ => unreachable!(),
        };
        let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(1, v as u64)),
        }));
    } else if let IPred::Eq | IPred::Uge | IPred::Ule | IPred::Sge | IPred::Sle = pred
        && lhs == rhs
    {
        // If the predicate includes equality then `%x eq %x` is trivially true.
        let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(1, 1)),
        }));
    } else if let IPred::Ne | IPred::Ugt | IPred::Ult | IPred::Sgt | IPred::Slt = pred
        && lhs == rhs
    {
        // If the predicate includes inequality then `%x ne %x` is trivially false.
        let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(1, 0)),
        }));
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_inttoptr(opt: &mut PassOpt, mut inst: IntToPtr) -> OptOutcome {
    inst.canonicalise(opt);
    let IntToPtr { val, .. } = inst;
    if let Some(ConstKind::Int(c)) = opt.as_constkind(val) {
        if c.bitw() <= u32::try_from(std::mem::size_of::<usize>() * 8).unwrap() {
            let tyidx = opt.push_ty(Ty::Ptr(0)).unwrap();
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Ptr(c.to_zero_ext_usize().unwrap()),
            }));
        } else {
            todo!();
        }
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_lshr(opt: &mut PassOpt, mut inst: LShr) -> OptOutcome {
    inst.canonicalise(opt);
    let LShr {
        tyidx,
        lhs,
        rhs,
        exact,
    } = inst;
    opt_ashr_lshr(opt, inst.into(), tyidx, lhs, rhs, exact, |lhs_c, rhs_c| {
        lhs_c.checked_lshr(rhs_c.to_zero_ext_u32().unwrap())
    })
}

fn opt_memcpy(opt: &mut PassOpt, mut inst: MemCpy) -> OptOutcome {
    inst.canonicalise(opt);
    let MemCpy {
        dst,
        src,
        len,
        volatile: _,
    } = inst;
    match (opt.as_constkind(dst), opt.as_constkind(src)) {
        (Some(ConstKind::Ptr(lhs_c)), Some(ConstKind::Ptr(rhs_c))) if lhs_c == rhs_c => {
            // memcpy where lhs_ptr == rhs_ptr. Not needed.
            // This is technically UB, but GCC 15 and Clang 21
            // both choose to optimise away the memcpy.
            // Thus, we shall do the same.
            return OptOutcome::NotNeeded;
        }
        _ => (),
    }

    if let Some(ConstKind::Int(len_c)) = opt.as_constkind(len)
        && let Some(len_as_int) = len_c.to_zero_ext_u64()
        && len_as_int == 0
    {
        // memcpy of zero bytes. Not needed.
        return OptOutcome::NotNeeded;
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_mul(opt: &mut PassOpt, mut inst: Mul) -> OptOutcome {
    inst.canonicalise(opt);
    let Mul {
        tyidx,
        lhs,
        rhs,
        nuw,
        nsw,
    } = inst;
    assert!(!nuw && !nsw);
    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 * c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(lhs_c.wrapping_mul(&rhs_c)),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            match rhs_c.to_zero_ext_u64() {
                Some(0) => {
                    // Reduce `x * 0` to `0`.
                    return OptOutcome::Equiv(rhs);
                }
                Some(1) => {
                    // Reduce `x * 1` to `x`.
                    return OptOutcome::Equiv(lhs);
                }
                Some(x) if x.is_power_of_two() => {
                    // Replace `x * y` with `x << ...`.
                    let c_iidx = opt.push_pre_inst(Inst::Const(Const {
                        tyidx,
                        kind: ConstKind::Int(ArbBitInt::from_u64(
                            rhs_c.bitw(),
                            u64::from(x.ilog2()),
                        )),
                    }));
                    return OptOutcome::Rewritten(
                        Shl {
                            tyidx,
                            lhs,
                            rhs: c_iidx,
                            nuw: false,
                            nsw: false,
                        }
                        .into(),
                    );
                }
                _ => (),
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_ptradd(opt: &mut PassOpt, mut inst: PtrAdd) -> OptOutcome {
    // LLVM semantics require pointer arithmetic to wrap as though they were "pointer index typed"
    // (a pointer-sized integer, for addrspace 0, the only address space we support right now).
    let mut off: isize = 0;
    loop {
        inst.canonicalise(opt);
        let PtrAdd {
            ptr,
            off: inst_off,
            in_bounds,
            nusw,
            nuw,
        } = inst;
        assert!(!in_bounds && !nusw && !nuw);

        off = off.checked_add(isize::try_from(inst_off).unwrap()).unwrap();
        if let Some(ConstKind::Ptr(addr)) = opt.as_constkind(ptr) {
            // Constant fold `ptr + off`.
            let tyidx = opt.push_ty(Ty::Ptr(0)).unwrap();
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Ptr(addr.wrapping_add_signed(off)),
            }));
        } else if let Inst::PtrAdd(x) = opt.inst(ptr) {
            inst = x.to_owned();
        } else if off == 0 {
            // Reduce `ptr + 0` to `x`.
            return OptOutcome::Equiv(ptr);
        } else {
            let inst = PtrAdd {
                ptr,
                off: i32::try_from(off).unwrap(),
                in_bounds,
                nusw,
                nuw,
            };
            return OptOutcome::Rewritten(inst.into());
        }
    }
}

fn opt_or(opt: &mut PassOpt, mut inst: Or) -> OptOutcome {
    inst.canonicalise(opt);
    let Or {
        tyidx,
        lhs,
        rhs,
        disjoint,
    } = inst;
    assert!(!disjoint);
    if lhs == rhs {
        // Reduce x | x to x.
        return OptOutcome::Equiv(lhs);
    }
    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 | c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(lhs_c.bitor(&rhs_c)),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x | 0` to `x`.
                return OptOutcome::Equiv(lhs);
            }
            if rhs_c == ArbBitInt::all_bits_set(rhs_c.bitw()) {
                // Reduce `x | y` to `y` if `y` is a constant that has all
                // the necessary bits set for this integer type.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_ptrtoint(opt: &mut PassOpt, mut inst: PtrToInt) -> OptOutcome {
    inst.canonicalise(opt);
    let PtrToInt { tyidx, val } = inst;
    if let Some(ConstKind::Ptr(addr)) = opt.as_constkind(val) {
        let dst_bitw = opt.ty(tyidx).bitw();
        let dst_tyidx = opt.push_ty(Ty::Int(dst_bitw)).unwrap();
        if dst_bitw <= u32::try_from(std::mem::size_of::<usize>() * 8).unwrap() {
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx: dst_tyidx,
                kind: ConstKind::Int(ArbBitInt::from_usize(addr).truncate(dst_bitw)),
            }));
        } else {
            todo!();
        }
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_select(opt: &mut PassOpt, mut inst: Select) -> OptOutcome {
    inst.canonicalise(opt);
    let Select {
        cond,
        truev,
        falsev,
        ..
    } = inst;
    if truev == falsev {
        // If truev and falsev map to the same value, we can reduce to either side.
        return OptOutcome::Equiv(truev);
    }

    if let Some(ConstKind::Int(cond_c)) = opt.as_constkind(cond) {
        match cond_c.to_zero_ext_u8().unwrap() {
            0 => return OptOutcome::Equiv(falsev),
            1 => return OptOutcome::Equiv(truev),
            _ => unreachable!(),
        }
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_sext(opt: &mut PassOpt, mut inst: SExt) -> OptOutcome {
    inst.canonicalise(opt);
    let SExt { tyidx, val } = inst;
    match opt.as_constkind(val) {
        Some(ConstKind::Double(_) | ConstKind::Float(_)) => unreachable!(),
        Some(ConstKind::Int(src_val)) => {
            let dst_bitw = opt.ty(tyidx).bitw();
            let dst_tyidx = opt.push_ty(Ty::Int(dst_bitw)).unwrap();
            OptOutcome::Rewritten(Inst::Const(Const {
                tyidx: dst_tyidx,
                kind: ConstKind::Int(src_val.sign_extend(dst_bitw)),
            }))
        }
        Some(ConstKind::Ptr(_)) => todo!(),
        None => OptOutcome::Rewritten(inst.into()),
    }
}

fn opt_shl(opt: &mut PassOpt, mut inst: Shl) -> OptOutcome {
    inst.canonicalise(opt);
    let Shl {
        tyidx,
        lhs,
        rhs,
        nuw,
        nsw,
    } = inst;
    assert!(!nuw && !nsw);
    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `lhs_c << rhs_c`.
            let c = lhs_c
                .checked_shl(rhs_c.to_zero_ext_u32().unwrap())
                .unwrap_or_else(|| ArbBitInt::all_bits_set(lhs_c.bitw()));
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(c),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x << 0` to `x`.
                return OptOutcome::Equiv(lhs);
            }
        }
        (Some(ConstKind::Int(lhs_c)), _) => {
            if lhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `0 << x` to `0`.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_sub(opt: &mut PassOpt, mut inst: Sub) -> OptOutcome {
    inst.canonicalise(opt);
    let Sub {
        tyidx,
        lhs,
        rhs,
        nuw,
        nsw,
    } = inst;
    assert!(!nuw && !nsw);
    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 - c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(lhs_c.wrapping_sub(&rhs_c)),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x - 0` to `x`.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_trunc(opt: &mut PassOpt, mut inst: Trunc) -> OptOutcome {
    inst.canonicalise(opt);
    let Trunc {
        tyidx,
        val,
        nuw,
        nsw,
    } = inst;
    assert!(!nuw && !nsw);
    if let Some(ConstKind::Int(c)) = opt.as_constkind(val) {
        let dst_bitw = opt.ty(tyidx).bitw();
        let dst_tyidx = opt.push_ty(Ty::Int(dst_bitw)).unwrap();
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx: dst_tyidx,
            kind: ConstKind::Int(c.truncate(dst_bitw)),
        }));
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_udiv(opt: &mut PassOpt, mut inst: UDiv) -> OptOutcome {
    inst.canonicalise(opt);
    let UDiv {
        tyidx,
        lhs,
        rhs,
        exact,
    } = inst;
    assert!(!exact);
    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 / c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(
                    lhs_c
                        .checked_udiv(&rhs_c)
                        .unwrap_or_else(|| ArbBitInt::all_bits_set(rhs_c.bitw())),
                ),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            match rhs_c.to_zero_ext_u64() {
                Some(1) => {
                    // Reduce `x / 1` to `x`.
                    return OptOutcome::Equiv(lhs);
                }
                Some(x) if x.is_power_of_two() => {
                    // Replace `x * y` with `x >> ...`.
                    let c_iidx = opt.push_pre_inst(Inst::Const(Const {
                        tyidx,
                        kind: ConstKind::Int(ArbBitInt::from_u64(
                            rhs_c.bitw(),
                            u64::from(x.ilog2()),
                        )),
                    }));
                    return OptOutcome::Rewritten(
                        LShr {
                            tyidx,
                            lhs,
                            rhs: c_iidx,
                            exact: false,
                        }
                        .into(),
                    );
                }
                _ => (),
            }
        }
        (_, _) => (),
    }
    OptOutcome::Rewritten(inst.into())
}

fn opt_xor(opt: &mut PassOpt, mut inst: Xor) -> OptOutcome {
    inst.canonicalise(opt);
    let Xor { tyidx, lhs, rhs } = inst;
    let bitw = opt.ty(tyidx).bitw();
    if lhs == rhs {
        // Reduce x ^ x to 0.
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(bitw, 0)),
        }));
    }

    match (opt.as_constkind(lhs), opt.as_constkind(rhs)) {
        (Some(ConstKind::Int(lhs_c)), Some(ConstKind::Int(rhs_c))) => {
            // Constant fold `c1 ^ c2`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Int(lhs_c.bitxor(&rhs_c)),
            }));
        }
        (_, Some(ConstKind::Int(rhs_c))) => {
            if rhs_c.to_zero_ext_u8() == Some(0) {
                // Reduce `x ^ 0` to `x`.
                return OptOutcome::Equiv(lhs);
            }
        }
        _ => (),
    }
    OptOutcome::Rewritten(inst.into())
}

fn opt_zext(opt: &mut PassOpt, mut inst: ZExt) -> OptOutcome {
    inst.canonicalise(opt);
    let ZExt { tyidx, val } = inst;
    match opt.as_constkind(val) {
        Some(ConstKind::Double(_) | ConstKind::Float(_)) => unreachable!(),
        Some(ConstKind::Int(src_val)) => {
            let dst_bitw = opt.ty(tyidx).bitw();
            let dst_tyidx = opt.push_ty(Ty::Int(dst_bitw)).unwrap();
            OptOutcome::Rewritten(Inst::Const(Const {
                tyidx: dst_tyidx,
                kind: ConstKind::Int(src_val.zero_extend(dst_bitw)),
            }))
        }
        Some(ConstKind::Ptr(_)) => todo!(),
        None => OptOutcome::Rewritten(inst.into()),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::fullopt::test::opt_and_test;

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

    #[test]
    fn opt_abs() {
        // Simple constant folding e.g abs(-1) == 1
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = abs %0, int_min_poison
          blackbox %1
        ",
            "
          ...
          %1: i8 = 2
          blackbox %1
        ",
        );

        test_sf(
            "
          %0: i8 = -2
          %1: i8 = abs %0, int_min_poison
          blackbox %1
        ",
            "
          ...
          %1: i8 = 2
          blackbox %1
        ",
        );
    }

    #[test]
    fn opt_add() {
        // Simple constant folding e.g `1 + 2`.
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 3
          %2: i8 = add %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 5
          blackbox %2
        ",
        );

        test_sf(
            "
          %0: i8 = 0
          %1: i8 = 255
          %2: i8 = add %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 255
          blackbox %2
        ",
        );

        // Strength reduction of `x + 0`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 0
          %2: i8 = add %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 0
          exit [%0]
        ",
        );
    }

    #[test]
    fn opt_and() {
        // x & x == x
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = and %0, %0
          exit [%1]
        ",
            "
          ...
          %0: i8 = arg
          exit [%0]
        ",
        );

        // Simple constant folding e.g `1 & 2`.
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 3
          %2: i8 = and %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 2
          blackbox %2
        ",
        );

        test_sf(
            "
          %0: i8 = 0
          %1: i8 = 255
          %2: i8 = and %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 0
          blackbox %2
        ",
        );

        // Strength reduction of `y & 0`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 0
          %2: i8 = and %0, %1
          exit [%2]
        ",
            "
          ...
          %1: i8 = 0
          exit [%1]
        ",
        );

        // Strength reduction of `y & 0b1111111` (i.e. all bits set in the appropriate type).
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 255
          %2: i8 = and %0, %1
          exit [%1]
        ",
            "
          ...
          %1: i8 = 255
          exit [%1]
        ",
        );
    }

    #[test]
    fn opt_ashr() {
        // Constant folding. We deliberately use an example where ashr/lshr give different
        // results
        test_sf(
            "
          %0: i8 = -2
          %1: i8 = 1
          %2: i8 = ashr %0, %1
          blackbox %2
        ",
            "
          %0: i8 = 254
          %1: i8 = 1
          %2: i8 = 255
          blackbox %2
        ",
        );

        // `x >> 0` reduces to `x`
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: i8 = ashr %0, %1
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = 0
          blackbox %0
        ",
        );

        // `0 >> x` reduces to `0`
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: i8 = ashr %1, %0
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = 0
          blackbox %1
        ",
        );
    }

    #[test]
    fn opt_ctpop() {
        // Constant fold the number of set bits
        test_sf(
            "
          %0: i64 = 0x1234
          %1: i64 = ctpop %0
          blackbox %1
        ",
            "
          %0: i64 = 4660
          %1: i64 = 5
          blackbox %1
        ",
        );
    }

    #[test]
    fn opt_dynptradd() {
        // Constant fold the number of elements
        test_sf(
            "
          %0: ptr = 0x1234
          %1: i32 = 10
          %2: i8 = dynptradd %0, %1, 4
          blackbox %2
        ",
            "
          %0: ptr = 0x1234
          %1: i32 = 10
          %2: ptr = 0x125C
          blackbox %2
        ",
        );

        // Constant fold the number of elements and optimise the intermediate ptradd
        test_sf(
            "
          %0: ptr = arg [reg]
          %1: ptr = ptradd %0, 4
          %2: i32 = 10
          %3: i8 = dynptradd %1, %2, 4
          blackbox %3
        ",
            "
          %0: ptr = arg
          %1: ptr = ptradd %0, 4
          %2: i32 = 10
          %3: ptr = ptradd %0, 44
          blackbox %3
        ",
        );
    }

    #[test]
    fn opt_fadd() {
        // Constant fold lhs float and rhs float
        test_sf(
            "
          %0: float = 1.01float
          %1: float = 2.01float
          %2: float = fadd %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = 3.02
          blackbox %2
        ",
        );

        // Constant fold lhs double and rhs double
        test_sf(
            "
          %0: double = 1.02double
          %1: double = 2.03double
          %2: double = fadd %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = 3.05
          blackbox %2
        ",
        );
    }

    #[test]
    fn opt_fdiv() {
        // Constant fold lhs float and rhs float
        test_sf(
            "
          %0: float = 2.02float
          %1: float = 1.01float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = 2
          blackbox %2
        ",
        );

        // Constant fold lhs double and rhs double
        test_sf(
            "
          %0: double = 2.02double
          %1: double = 1.01double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = 2
          blackbox %2
        ",
        );

        // Constant fold lhs=non-zero, rhs=0.0f
        test_sf(
            "
          %0: float = 2.02float
          %1: float = 0.0float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = inf
          blackbox %2
        ",
        );

        // Constant fold lhs=non-zero, rhs=0.0d
        test_sf(
            "
          %0: double = 2.02double
          %1: double = 0.0double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = inf
          blackbox %2
        ",
        );

        // Constant fold lhs=non-zero, rhs=-0.0f
        test_sf(
            "
          %0: float = 2.02float
          %1: float = -0.0float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = -inf
          blackbox %2
        ",
        );

        // Constant fold lhs=non-zero, rhs=-0.0d
        test_sf(
            "
          %0: double = 2.02double
          %1: double = -0.0double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = -inf
          blackbox %2
        ",
        );

        // Constant fold lhs=0.0f, rhs=non-zero
        test_sf(
            "
          %0: float = 0.0float
          %1: float = 2.0float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = 0
          blackbox %2
        ",
        );

        // Constant fold lhs=0.0d, rhs=non-zero
        test_sf(
            "
          %0: double = 0.0double
          %1: double = 2.0double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = 0
          blackbox %2
        ",
        );

        // Constant fold lhs=-0.0f, rhs=non-zero
        test_sf(
            "
          %0: float = -0.0float
          %1: float = 2.0float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = -0
          blackbox %2
        ",
        );

        // Constant fold lhs=-0.0d, rhs=non-zero
        test_sf(
            "
          %0: double = -0.0double
          %1: double = 2.0double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = -0
          blackbox %2
        ",
        );

        // Constant fold lhs=0.0f, rhs=0.0f
        test_sf(
            "
          %0: float = 0.0float
          %1: float = 0.0float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = NaN
          blackbox %2
        ",
        );

        // Constant fold lhs=0.0d, rhs=0.0d
        test_sf(
            "
          %0: double = 0.0double
          %1: double = 0.0double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = NaN
          blackbox %2
        ",
        );

        // Constant fold lhs=0.0f, rhs=-0.0f
        test_sf(
            "
          %0: float = 0.0float
          %1: float = -0.0float
          %2: float = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = NaN
          blackbox %2
        ",
        );

        // Constant fold lhs=0.0d, rhs=-0.0d
        test_sf(
            "
          %0: double = 0.0double
          %1: double = -0.0double
          %2: double = fdiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = NaN
          blackbox %2
        ",
        );
    }

    #[test]
    fn opt_fmul() {
        // Constant fold lhs float and rhs float
        test_sf(
            "
          %0: float = 2.02float
          %1: float = 1.01float
          %2: float = fmul %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = 2.0402
          blackbox %2
        ",
        );

        // Constant fold lhs double and rhs double
        test_sf(
            "
          %0: double = 2.02double
          %1: double = 1.01double
          %2: double = fmul %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = 2.0402
          blackbox %2
        ",
        );
    }

    #[test]
    fn opt_fsub() {
        // Constant fold lhs float and rhs float
        test_sf(
            "
          %0: float = 2.02float
          %1: float = 1.01float
          %2: float = fsub %0, %1
          blackbox %2
        ",
            "
          ...
          %2: float = 1.01
          blackbox %2
        ",
        );

        // Constant fold lhs double and rhs double
        test_sf(
            "
          %0: double = 2.02double
          %1: double = 1.01double
          %2: double = fsub %0, %1
          blackbox %2
        ",
            "
          ...
          %2: double = 1.01
          blackbox %2
        ",
        );
    }

    #[test]
    fn opt_guard() {
        // Guard referencing a constant
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i1 = 1
          guard true, %1, []
          exit [%0]
        ",
            "
          %0: i8 = arg
          %1: i1 = 1
          exit [%0]
        ",
        );

        // Guard setting a range on IPred::Eq
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 4
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          exit [%0]
        ",
            "
          %0: i8 = arg
          %1: i8 = 4
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          exit [%1]
        ",
        );

        // Guard not setting a range on IPred::Ne
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 4
          %2: i1 = icmp ne %0, %1
          guard true, %2, []
          exit [%0]
        ",
            "
          %0: i8 = arg
          %1: i8 = 4
          %2: i1 = icmp ne %0, %1
          guard true, %2, []
          exit [%0]
        ",
        );

        // Equivalency determined by guards
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

        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          exit [%0, %1]
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          exit [%0, %0]
        ",
        );

        // Removing duplicate guards after equivalency
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = 1
          %3: i8 = 4
          %4: i1 = icmp eq %0, %3
          %5: i1 = icmp eq %0, %1
          %6: i1 = icmp eq %1, %3
          guard true, %4, []
          guard true, %5, []
          guard true, %6, []
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
          %6: i1 = icmp eq %1, %3
          guard true, %4, []
          guard true, %5, []
          blackbox %3
          blackbox %3
          exit [%3, %3]
        ",
        );
    }

    #[test]
    fn opt_icmp() {
        // Ints

        // Simple constant folding.

        // eq
        test_sf(
            "
          %0: i8 = 5
          %1: i8 = 5
          %2: i1 = icmp eq %0, %1
          blackbox %2
          %4: i8 = 6
          %5: i1 = icmp eq %0, %4
          blackbox %5
        ",
            "
          %0: i8 = 5
          %1: i8 = 5
          %2: i1 = 1
          blackbox %2
          %4: i8 = 6
          %5: i1 = 0
          blackbox %5
        ",
        );

        // ne
        test_sf(
            "
          %0: i8 = 5
          %1: i8 = 5
          %2: i1 = icmp ne %0, %1
          blackbox %2
          %4: i8 = 6
          %5: i1 = icmp ne %0, %4
          blackbox %5
        ",
            "
          %0: i8 = 5
          %1: i8 = 5
          %2: i1 = 0
          blackbox %2
          %4: i8 = 6
          %5: i1 = 1
          blackbox %5
        ",
        );

        // ugt
        test_sf(
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = icmp ugt %0, %1
          blackbox %2
          %4: i8 = 5
          %5: i1 = icmp ugt %0, %4
          blackbox %5
          %7: i8 = 6
          %8: i1 = icmp ugt %0, %7
          blackbox %8
        ",
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = 1
          blackbox %2
          %4: i8 = 5
          %5: i1 = 0
          blackbox %5
          %7: i8 = 6
          %8: i1 = 0
          blackbox %8
        ",
        );

        // uge
        test_sf(
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = icmp uge %0, %1
          blackbox %2
          %4: i8 = 5
          %5: i1 = icmp uge %0, %4
          blackbox %5
          %7: i8 = 6
          %8: i1 = icmp uge %0, %7
          blackbox %8
        ",
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = 1
          blackbox %2
          %4: i8 = 5
          %5: i1 = 1
          blackbox %5
          %7: i8 = 6
          %8: i1 = 0
          blackbox %8
        ",
        );

        // ult
        test_sf(
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = icmp ult %0, %1
          blackbox %2
          %4: i8 = 5
          %5: i1 = icmp ult %0, %4
          blackbox %5
          %7: i8 = 6
          %8: i1 = icmp ult %0, %7
          blackbox %8
        ",
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = 0
          blackbox %2
          %4: i8 = 5
          %5: i1 = 0
          blackbox %5
          %7: i8 = 6
          %8: i1 = 1
          blackbox %8
        ",
        );

        // ule
        test_sf(
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = icmp ule %0, %1
          blackbox %2
          %4: i8 = 5
          %5: i1 = icmp ule %0, %4
          blackbox %5
          %7: i8 = 6
          %8: i1 = icmp ule %0, %7
          blackbox %8
        ",
            "
          %0: i8 = 5
          %1: i8 = 4
          %2: i1 = 0
          blackbox %2
          %4: i8 = 5
          %5: i1 = 1
          blackbox %5
          %7: i8 = 6
          %8: i1 = 1
          blackbox %8
        ",
        );

        // sgt
        test_sf(
            "
          %0: i8 = -5
          %1: i8 = -6
          %2: i1 = icmp sgt %0, %1
          blackbox %2
          %4: i8 = -5
          %5: i1 = icmp sgt %0, %4
          blackbox %5
          %7: i8 = -4
          %8: i1 = icmp sgt %0, %7
          blackbox %8
          %10: i8 = 0
          %11: i1 = icmp sgt %0, %10
          blackbox %11
        ",
            "
          %0: i8 = 251
          %1: i8 = 250
          %2: i1 = 1
          blackbox %2
          %4: i8 = 251
          %5: i1 = 0
          blackbox %5
          %7: i8 = 252
          %8: i1 = 0
          blackbox %8
          %10: i8 = 0
          %11: i1 = 0
          blackbox %11
        ",
        );

        // sge
        test_sf(
            "
          %0: i8 = -5
          %1: i8 = -6
          %2: i1 = icmp sge %0, %1
          blackbox %2
          %4: i8 = -5
          %5: i1 = icmp sge %0, %4
          blackbox %5
          %7: i8 = -4
          %8: i1 = icmp sge %0, %7
          blackbox %8
          %10: i8 = 0
          %11: i1 = icmp sge %0, %10
          blackbox %11
        ",
            "
          %0: i8 = 251
          %1: i8 = 250
          %2: i1 = 1
          blackbox %2
          %4: i8 = 251
          %5: i1 = 1
          blackbox %5
          %7: i8 = 252
          %8: i1 = 0
          blackbox %8
          %10: i8 = 0
          %11: i1 = 0
          blackbox %11
        ",
        );

        // slt
        test_sf(
            "
          %0: i8 = -5
          %1: i8 = -6
          %2: i1 = icmp slt %0, %1
          blackbox %2
          %4: i8 = -5
          %5: i1 = icmp slt %0, %4
          blackbox %5
          %7: i8 = -4
          %8: i1 = icmp slt %0, %7
          blackbox %8
          %10: i8 = 0
          %11: i1 = icmp slt %0, %10
          blackbox %11
        ",
            "
          %0: i8 = 251
          %1: i8 = 250
          %2: i1 = 0
          blackbox %2
          %4: i8 = 251
          %5: i1 = 0
          blackbox %5
          %7: i8 = 252
          %8: i1 = 1
          blackbox %8
          %10: i8 = 0
          %11: i1 = 1
          blackbox %11
        ",
        );

        // sle
        test_sf(
            "
          %0: i8 = -5
          %1: i8 = -6
          %2: i1 = icmp sle %0, %1
          blackbox %2
          %4: i8 = -5
          %5: i1 = icmp sle %0, %4
          blackbox %5
          %7: i8 = -4
          %8: i1 = icmp sle %0, %7
          blackbox %8
          %10: i8 = 0
          %11: i1 = icmp sle %0, %10
          blackbox %11
        ",
            "
          %0: i8 = 251
          %1: i8 = 250
          %2: i1 = 0
          blackbox %2
          %4: i8 = 251
          %5: i1 = 1
          blackbox %5
          %7: i8 = 252
          %8: i1 = 1
          blackbox %8
          %10: i8 = 0
          %11: i1 = 1
          blackbox %11
        ",
        );

        // Pointers

        // eq
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1234
          %2: i1 = icmp eq %0, %1
          blackbox %2
          %4: ptr = 0xABCD
          %5: i1 = icmp eq %0, %4
          blackbox %5
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1234
          %2: i1 = 1
          blackbox %2
          %4: ptr = 0xABCD
          %5: i1 = 0
          blackbox %5
        ",
        );

        // ne
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1234
          %2: i1 = icmp ne %0, %1
          blackbox %2
          %4: ptr = 0xABCD
          %5: i1 = icmp ne %0, %4
          blackbox %5
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1234
          %2: i1 = 0
          blackbox %2
          %4: ptr = 0xABCD
          %5: i1 = 1
          blackbox %5
        ",
        );

        // ugt
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = icmp ugt %0, %1
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = icmp ugt %0, %4
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = icmp ugt %0, %7
          blackbox %8
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = 1
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = 0
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = 0
          blackbox %8
        ",
        );

        // uge
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = icmp uge %0, %1
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = icmp uge %0, %4
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = icmp uge %0, %7
          blackbox %8
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = 1
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = 1
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = 0
          blackbox %8
        ",
        );

        // ult
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = icmp ult %0, %1
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = icmp ult %0, %4
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = icmp ult %0, %7
          blackbox %8
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = 0
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = 0
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = 1
          blackbox %8
        ",
        );

        // ule
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = icmp ule %0, %1
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = icmp ule %0, %4
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = icmp ule %0, %7
          blackbox %8
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1233
          %2: i1 = 0
          blackbox %2
          %4: ptr = 0x1234
          %5: i1 = 1
          blackbox %5
          %7: ptr = 0x1235
          %8: i1 = 1
          blackbox %8
        ",
        );

        // Equality/inequality comparisons of the same instruction are true/false
        // respectively.

        // eq
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp eq %0, %0
          blackbox %2
          %4: i1 = icmp eq %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 1
          blackbox %2
          %4: i1 = icmp eq %0, %1
          blackbox %4
        ",
        );

        // ne
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp ne %0, %0
          blackbox %2
          %4: i1 = icmp ne %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp ne %0, %1
          blackbox %4
        ",
        );

        // ugt
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp ugt %0, %0
          blackbox %2
          %4: i1 = icmp ugt %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp ugt %0, %1
          blackbox %4
        ",
        );

        // uge
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp uge %0, %0
          blackbox %2
          %4: i1 = icmp uge %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 1
          blackbox %2
          %4: i1 = icmp uge %0, %1
          blackbox %4
        ",
        );

        // ult
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp ult %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp ult %0, %1
          blackbox %4
        ",
        );

        // ule
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp ule %0, %0
          blackbox %2
          %4: i1 = icmp ule %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 1
          blackbox %2
          %4: i1 = icmp ule %0, %1
          blackbox %4
        ",
        );

        // sgt
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp sgt %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp sgt %0, %1
          blackbox %4
        ",
        );

        // sge
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp sge %0, %0
          blackbox %2
          %4: i1 = icmp sge %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 1
          blackbox %2
          %4: i1 = icmp sge %0, %1
          blackbox %4
        ",
        );

        // slt
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp slt %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 0
          blackbox %2
          %4: i1 = icmp slt %0, %1
          blackbox %4
        ",
        );

        // sge
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp sge %0, %0
          blackbox %2
          %4: i1 = icmp sge %0, %1
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = 1
          blackbox %2
          %4: i1 = icmp sge %0, %1
          blackbox %4
        ",
        );
    }

    #[test]
    fn opt_inttoptr() {
        test_sf(
            "
          %0: i8 = 129
          %1: ptr = inttoptr %0
          blackbox %1
          %3: i64 = 0xABCD1234
          %4: ptr = inttoptr %3
          blackbox %4
        ",
            "
          %0: i8 = 129
          %1: ptr = 0x81
          blackbox %1
          %3: i64 = 2882343476
          %4: ptr = 0xABCD1234
          blackbox %4
        ",
        );
    }

    #[test]
    fn opt_lshr() {
        // Constant folding. We deliberately use an example where ashr/lshr give different
        // results
        test_sf(
            "
          %0: i8 = -2
          %1: i8 = 1
          %2: i8 = lshr %0, %1
          blackbox %2
        ",
            "
          %0: i8 = 254
          %1: i8 = 1
          %2: i8 = 127
          blackbox %2
        ",
        );

        // `x >> 0` reduces to `x`
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: i8 = lshr %0, %1
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = 0
          blackbox %0
        ",
        );

        // `0 >> x` reduces to `0`
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: i8 = lshr %1, %0
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = 0
          blackbox %1
        ",
        );
    }

    #[test]
    fn opt_memcpy() {
        // memcpy where src==dst is eliminated.
        test_sf(
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1234
          %2: i64 = 4
          memcpy %0, %1, %2, false
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1234
          %2: i64 = 4
        ",
        );

        // memcpy where src!=dst is not eliminated.
        test_sf(
            "
          %0: ptr = 0x4321
          %1: ptr = 0x1234
          %2: i64 = 4
          memcpy %0, %1, %2, false
        ",
            "
          %0: ptr = 0x4321
          %1: ptr = 0x1234
          %2: i64 = 4
          memcpy %0, %1, %2, false
        ",
        );

        // memcpy where len == 0 is eliminated.
        test_sf(
            "
          %0: ptr = 0x4321
          %1: ptr = 0x1234
          %2: i64 = 0
          memcpy %0, %1, %2, false
        ",
            "
          %0: ptr = 0x4321
          %1: ptr = 0x1234
          %2: i64 = 0
        ",
        );
    }

    #[test]
    fn opt_mul() {
        // Simple constant folding e.g `1 * 2`.
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 3
          %2: i8 = mul %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 6
          blackbox %2
        ",
        );

        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 200
          %2: i8 = mul %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 144
          blackbox %2
        ",
        );

        // Strength reduction of `x * 0`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 0
          %2: i8 = mul %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 0
          exit [%1]
        ",
        );

        // Strength reduction of `x * 1`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 1
          %2: i8 = mul %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 1
          exit [%0]
        ",
        );

        // Strength reduction of `x * y` if y is a power of 2.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 4
          %2: i8 = mul %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 4
          %2: i8 = 2
          %3: i8 = shl %0, %2
          exit [%3]
        ",
        );
    }

    #[test]
    fn opt_or() {
        // x | x == x
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = or %0, %0
          exit [%1]
        ",
            "
          ...
          %0: i8 = arg
          exit [%0]
        ",
        );

        // Constant folding
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 8
          %2: i8 = or %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 10
          blackbox %2
        ",
        );

        // Strength reduction of `y | 0`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 0
          %2: i8 = or %0, %1
          exit [%2]
        ",
            "
          ...
          %1: i8 = 0
          exit [%0]
        ",
        );

        // Strength reduction of `y & 0b1111111` (i.e. all bits set in the appropriate type).
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 255
          %2: i8 = or %0, %1
          exit [%1]
        ",
            "
          ...
          %1: i8 = 255
          exit [%1]
        ",
        );
    }

    #[test]
    fn opt_ptradd() {
        // Constant folding
        test_sf(
            "
          %0: ptr = 0x1234
          %1: i8 = ptradd %0, 4
          blackbox %1
        ",
            "
          %0: ptr = 0x1234
          %1: ptr = 0x1238
          blackbox %1
        ",
        );

        // ptr + 0 == ptr
        test_sf(
            "
          %0: ptr = arg [reg]
          %1: i8 = ptradd %0, 0
          blackbox %1
        ",
            "
          %0: ptr = arg
          blackbox %0
        ",
        );

        // Collapsing `ptradd` chains.

        test_sf(
            "
          %0: ptr = arg [reg]
          %1: i8 = ptradd %0, 4
          %2: i8 = ptradd %1, 4
          blackbox %2
        ",
            "
          %0: ptr = arg
          %1: ptr = ptradd %0, 4
          %2: ptr = ptradd %0, 8
          blackbox %2
        ",
        );

        // ptr + 0 == ptr
        test_sf(
            "
          %0: ptr = arg [reg]
          %1: i8 = ptradd %0, 4
          blackbox %0
        ",
            "
          %0: ptr = arg
          %1: ptr = ptradd %0, 4
          blackbox %0
        ",
        );
    }

    #[test]
    fn opt_ptrtoint() {
        test_sf(
            "
          %0: ptr = 0x1234
          %1: i8 = ptrtoint %0
          blackbox %1
          %3: i16 = ptrtoint %0
          blackbox %3
        ",
            "
          %0: ptr = 0x1234
          %1: i8 = 52
          blackbox %1
          %3: i16 = 4660
          blackbox %3
        ",
        );
    }

    #[test]
    fn opt_select() {
        // Constant false
        test_sf(
            "
          %0: i1 = 0
          %1: i8 = 2
          %2: i8 = 3
          %3: i8 = select %0, %1, %2
          blackbox %3
        ",
            "
          %0: i1 = 0
          %1: i8 = 2
          %2: i8 = 3
          blackbox %2
        ",
        );

        // Constant true
        test_sf(
            "
          %0: i1 = 1
          %1: i8 = 2
          %2: i8 = 3
          %3: i8 = select %0, %1, %2
          blackbox %3
        ",
            "
          %0: i1 = 1
          %1: i8 = 2
          %2: i8 = 3
          blackbox %1
        ",
        );

        // Equal truev/falsev
        test_sf(
            "
          %0: i1 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = select %0, %1, %1
          blackbox %2
        ",
            "
          %0: i1 = arg
          %1: i8 = arg
          blackbox %1
        ",
        );
    }

    #[test]
    fn opt_sext() {
        test_sf(
            "
          %0: i8 = 3
          %1: i16 = sext %0
          blackbox %1
          %3: i8 = 255
          %4: i16 = sext %3
          blackbox %4
        ",
            "
          %0: i8 = 3
          %1: i16 = 3
          blackbox %1
          %3: i8 = 255
          %4: i16 = 65535
          blackbox %4
        ",
        );
    }

    #[test]
    fn opt_shl() {
        // Constant folding
        test_sf(
            "
          %0: i8 = 7
          %1: i8 = 3
          %2: i8 = shl %0, %1
          blackbox %2
        ",
            "
          %0: i8 = 7
          %1: i8 = 3
          %2: i8 = 56
          blackbox %2
        ",
        );

        // `x << 0` reduces to `x`
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: i8 = shl %0, %1
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = 0
          blackbox %0
        ",
        );

        // `0 << x` reduces to `0`
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = 0
          %2: i8 = shl %1, %0
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = 0
          blackbox %1
        ",
        );
    }

    #[test]
    fn opt_sub() {
        // Simple constant folding e.g `1 - 2`.
        test_sf(
            "
          %0: i8 = 3
          %1: i8 = 2
          %2: i8 = sub %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 1
          blackbox %2
        ",
        );

        // Test constant folding wraps.
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 4
          %2: i8 = sub %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 254
          blackbox %2
        ",
        );

        // Strength reduction of `x - 0`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 0
          %2: i8 = sub %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 0
          exit [%0]
        ",
        );
    }

    #[test]
    fn opt_trunc() {
        test_sf(
            "
          %0: i16 = 0xFFFF
          %1: i8 = trunc %0
          blackbox %1
          %3: i16 = 254
          %4: i8 = trunc %3
          blackbox %4
        ",
            "
          %0: i16 = 65535
          %1: i8 = 255
          blackbox %1
          %3: i16 = 254
          %4: i8 = 254
          blackbox %4
        ",
        );
    }

    #[test]
    fn opt_udiv() {
        // Simple constant folding e.g `1 / 2`.
        test_sf(
            "
          %0: i8 = 6
          %1: i8 = 2
          %2: i8 = udiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 3
          blackbox %2
        ",
        );

        test_sf(
            "
          %0: i8 = 255
          %1: i8 = 2
          %2: i8 = udiv %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 127
          blackbox %2
        ",
        );

        // Strength reduction of `x / 1`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 1
          %2: i8 = udiv %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 1
          exit [%0]
        ",
        );

        // Strength reduction of `x / y` if y is a power of 2.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 4
          %2: i8 = udiv %0, %1
          exit [%2]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 4
          %2: i8 = 2
          %3: i8 = lshr %0, %2
          exit [%3]
        ",
        );
    }

    #[test]
    fn opt_xor() {
        // x ^ x == 0
        test_sf(
            "
          %0: i8 = arg [reg]
          %1: i8 = xor %0, %0
          exit [%1]
        ",
            "
          ...
          %0: i8 = arg
          %1: i8 = 0
          exit [%1]
        ",
        );

        // Constant folding
        test_sf(
            "
          %0: i8 = 2
          %1: i8 = 3
          %2: i8 = xor %0, %1
          blackbox %2
        ",
            "
          ...
          %2: i8 = 1
          blackbox %2
        ",
        );

        // Strength reduction of `y ^ 0`.
        test_sf(
            "
          %0: i8 = arg [ reg ]
          %1: i8 = 0
          %2: i8 = xor %0, %1
          exit [%2]
        ",
            "
          ...
          %1: i8 = 0
          exit [%0]
        ",
        );
    }

    #[test]
    fn opt_zext() {
        test_sf(
            "
          %0: i8 = 3
          %1: i16 = zext %0
          blackbox %1
          %3: i8 = 255
          %4: i16 = zext %3
          blackbox %4
        ",
            "
          %0: i8 = 3
          %1: i16 = 3
          blackbox %1
          %3: i8 = 255
          %4: i16 = 255
          blackbox %4
        ",
        );
    }
}
