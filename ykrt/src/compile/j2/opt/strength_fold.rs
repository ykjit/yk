//! Strength reduction and constant folding

use crate::compile::{
    j2::{
        hir::*,
        opt::opt::{Opt, OptOutcome, Range},
    },
    jitc_yk::arbbitint::ArbBitInt,
};

pub(super) fn strength_fold(opt: &mut Opt, inst: Inst) -> OptOutcome {
    match inst {
        Inst::Add(x) => opt_add(opt, x),
        Inst::And(x) => opt_and(opt, x),
        Inst::Guard(x) => opt_guard(opt, x),
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
