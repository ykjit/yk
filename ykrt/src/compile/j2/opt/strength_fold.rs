//! Strength reduction and constant folding

use crate::compile::{
    j2::{
        hir::*,
        opt::{
            OptT,
            opt::{Opt, OptOutcome, Range},
        },
    },
    jitc_yk::arbbitint::ArbBitInt,
};

pub(super) fn strength_fold(opt: &mut Opt, inst: Inst) -> OptOutcome {
    match inst {
        Inst::Add(x) => opt_add(opt, x),
        Inst::And(x) => opt_and(opt, x),
        Inst::Guard(x) => opt_guard(opt, x),
        Inst::ICmp(x) => opt_icmp(opt, x),
        Inst::PtrAdd(x) => opt_ptradd(opt, x),
        Inst::Sub(x) => opt_sub(opt, x),
        _ => OptOutcome::Rewritten(inst),
    }
}

fn opt_add(
    opt: &mut Opt,
    inst @ Add {
        tyidx,
        lhs,
        rhs,
        nuw,
        nsw,
    }: Add,
) -> OptOutcome {
    assert!(!nuw && !nsw);
    if let (
        Inst::Const(Const {
            kind: ConstKind::Int(lhs_c),
            ..
        }),
        Inst::Const(Const {
            kind: ConstKind::Int(rhs_c),
            ..
        }),
    ) = (opt.inst_rewrite(lhs), opt.inst_rewrite(rhs))
    {
        // Constant fold `c1 + c2`.
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(lhs_c.wrapping_add(&rhs_c)),
        }));
    } else if let Inst::Const(Const {
        kind: ConstKind::Int(rhs_c),
        ..
    }) = opt.inst_rewrite(rhs)
        && rhs_c.to_zero_ext_u8() == Some(0)
    {
        // Reduce `x + 0` to `x`.
        return OptOutcome::ReducedTo(lhs);
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_and(opt: &mut Opt, inst @ And { tyidx, lhs, rhs }: And) -> OptOutcome {
    if lhs == rhs {
        // Reduce x & x with x.
        return OptOutcome::ReducedTo(lhs);
    } else if let (
        Inst::Const(Const {
            kind: ConstKind::Int(lhs_c),
            ..
        }),
        Inst::Const(Const {
            kind: ConstKind::Int(rhs_c),
            ..
        }),
    ) = (opt.inst_rewrite(lhs), opt.inst_rewrite(rhs))
    {
        // Constant fold `c1 & c2`.
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(lhs_c.bitand(&rhs_c)),
        }));
    } else if let Inst::Const(Const {
        kind: ConstKind::Int(rhs_c),
        ..
    }) = opt.inst_rewrite(rhs)
    {
        if rhs_c.to_zero_ext_u8() == Some(0) {
            // Reduce `x & 0` to `0`.
            return OptOutcome::ReducedTo(rhs);
        }
        if rhs_c == ArbBitInt::all_bits_set(rhs_c.bitw()) {
            // Reduce `x & y` to `x` if `y` is a constant that has all
            // the necessary bits set for this integer type. For an i1, for
            // example, `x & 1` can be replaced with `x`.
            return OptOutcome::ReducedTo(lhs);
        }
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_guard(opt: &mut Opt, inst @ Guard { expect, cond, .. }: Guard) -> OptOutcome {
    let cond_inst = opt.inst_rewrite(cond);
    if let Inst::Const(_) = cond_inst {
        // A guard that references a constant is, by definition, not needed and
        // doesn't affect future analyses.
        return OptOutcome::NotNeeded;
    } else if expect {
        if let Inst::ICmp(ICmp {
            pred: IPred::Eq,
            lhs,
            rhs,
            samesign,
        }) = cond_inst
            && lhs == rhs
        {
            assert!(!samesign);
            return OptOutcome::NotNeeded;
        }
        if let Inst::ICmp(ICmp {
            pred: IPred::Eq,
            lhs,
            rhs,
            samesign,
        }) = cond_inst
            && let Inst::Const(_) = opt.inst_rewrite(rhs)
        {
            assert!(!samesign);
            opt.set_range(lhs, Range::Equivalent(rhs));
        }
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_icmp(
    opt: &mut Opt,
    inst @ ICmp {
        pred,
        lhs,
        rhs,
        samesign,
    }: ICmp,
) -> OptOutcome {
    assert!(!samesign);
    if let (
        Inst::Const(Const {
            kind: ConstKind::Int(lhs_c),
            ..
        }),
        Inst::Const(Const {
            kind: ConstKind::Int(rhs_c),
            ..
        }),
    ) = (opt.inst_rewrite(lhs), opt.inst_rewrite(rhs))
    {
        // Constant fold.
        let v = match pred {
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
        };
        let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(1, v as u64)),
        }));
    } else if let IPred::Eq | IPred::Uge | IPred::Ule | IPred::Sge | IPred::Sle = pred
        && opt.map_iidx(lhs) == opt.map_iidx(rhs)
    {
        // If the predicate includes equality then `%x eq %x` is trivially true.
        let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(ArbBitInt::from_u64(1, 1)),
        }));
    } else if let IPred::Ne | IPred::Ugt | IPred::Ult | IPred::Sgt | IPred::Slt = pred
        && opt.map_iidx(lhs) == opt.map_iidx(rhs)
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

fn opt_ptradd(opt: &mut Opt, mut inst: PtrAdd) -> OptOutcome {
    // LLVM semantics require pointer arithmetic to wrap as though they were "pointer index typed"
    // (a pointer-sized integer, for addrspace 0, the only address space we support right now).
    let mut off: isize = 0;
    loop {
        let PtrAdd {
            ptr,
            off: inst_off,
            in_bounds,
            nusw,
            nuw,
        } = inst;
        assert!(!in_bounds && !nusw && !nuw);

        off = off.checked_add(isize::try_from(inst_off).unwrap()).unwrap();
        let ptr_inst = opt.inst_rewrite(ptr);
        if let Inst::Const(Const {
            tyidx,
            kind: ConstKind::Ptr(addr),
        }) = ptr_inst
        {
            // Constant fold `ptr + off`.
            return OptOutcome::Rewritten(Inst::Const(Const {
                tyidx,
                kind: ConstKind::Ptr(addr.wrapping_add_signed(off)),
            }));
        } else if let Inst::PtrAdd(x) = ptr_inst {
            inst = x;
        } else if off == 0 {
            // Reduce `ptr + 0` to `x`.
            return OptOutcome::ReducedTo(ptr);
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

fn opt_sub(
    opt: &mut Opt,
    inst @ Sub {
        tyidx,
        lhs,
        rhs,
        nuw,
        nsw,
    }: Sub,
) -> OptOutcome {
    assert!(!nuw && !nsw);
    if let (
        Inst::Const(Const {
            kind: ConstKind::Int(lhs_c),
            ..
        }),
        Inst::Const(Const {
            kind: ConstKind::Int(rhs_c),
            ..
        }),
    ) = (opt.inst_rewrite(lhs), opt.inst_rewrite(rhs))
    {
        // Constant fold `c1 - c2`.
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(lhs_c.wrapping_sub(&rhs_c)),
        }));
    } else if let Inst::Const(Const {
        kind: ConstKind::Int(rhs_c),
        ..
    }) = opt.inst_rewrite(rhs)
        && rhs_c.to_zero_ext_u8() == Some(0)
    {
        // Reduce `x - 0` to `x`.
        return OptOutcome::ReducedTo(lhs);
    }

    OptOutcome::Rewritten(inst.into())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::opt::test::opt_and_test;

    fn test_sf(mod_s: &str, ptn: &str) {
        opt_and_test(
            mod_s,
            |opt, inst| strength_fold(opt, opt.rewrite(inst)),
            ptn,
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
}
