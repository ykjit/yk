//! Strength reduction and constant folding

use crate::compile::{
    j2::{
        hir::*,
        opt::opt::{Opt, OptOutcome},
    },
    jitc_yk::arbbitint::ArbBitInt,
};

pub(super) fn strength_fold(opt: &mut Opt, inst: Inst) -> OptOutcome {
    match inst {
        Inst::And(x) => opt_and(opt, x),
        _ => OptOutcome::Unchanged(inst),
    }
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
    ) = (opt.inst(lhs), &opt.inst(rhs))
    {
        // Constant fold `c1 & c2`.
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(lhs_c.bitand(rhs_c)),
        }));
    } else if let Inst::Const(Const {
        kind: ConstKind::Int(rhs_c),
        ..
    }) = opt.inst(rhs)
    {
        if rhs_c.to_zero_ext_u8() == Some(0) {
            // Reduce `x & 0` to `0`.
            return OptOutcome::ReducedTo(rhs);
        }
        if rhs_c == &ArbBitInt::all_bits_set(rhs_c.bitw()) {
            // Reduce `x & y` to `x` if `y` is a constant that has all
            // the necessary bits set for this integer type. For an i1, for
            // example, `x & 1` can be replaced with `x`.
            return OptOutcome::ReducedTo(lhs);
        }
    }

    OptOutcome::Unchanged(inst.into())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::opt::test::opt_and_test;

    fn test_sf(mod_s: &str, ptn: &str) {
        opt_and_test(mod_s, strength_fold, ptn);
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
}
