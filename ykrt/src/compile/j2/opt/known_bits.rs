//! Known bits analysis.
//!
//! Data-flow analysis to gain information about bits. This is heavily influenced by the
//! [PyPy blog post](https://pypy.org/posts/2024/08/toy-knownbits.html)

use crate::compile::{
    j2::{
        hir::{And, Const, ConstKind, Inst, InstIdx, InstT, Or, Ty},
        opt::{
            BlockLikeT,
            fullopt::{CommitInstOpt, OptOutcome, PassOpt, PassT},
        },
    },
    jitc_yk::arbbitint::ArbBitInt,
};
use index_vec::IndexVec;

/// Known-bits analysis.
pub(super) struct KnownBits {
    /// Maps an SSA value to its corresponding known bits.  The value is None by default when
    /// unpopulated. When querying for the value, a None value is returned as a `KnownBitValue`
    /// with all bits set to unknown.
    known_bits: IndexVec<InstIdx, Option<KnownBitValue>>,
    /// The KnownBitValue of the current instruction being processed. This is only committed at
    /// the end of the instruction's analysis.
    pending_commit: Option<KnownBitValue>,
}

impl PassT for KnownBits {
    fn feed(&mut self, opt: &mut PassOpt, inst: Inst) -> OptOutcome {
        self.pending_commit = None;
        match inst {
            Inst::And(x) => self.opt_and(opt, x),
            Inst::Const(x) => self.opt_const(x),
            Inst::Or(x) => self.opt_or(opt, x),
            _ => OptOutcome::Rewritten(inst),
        }
    }

    fn inst_committed(&mut self, _opt: &CommitInstOpt, iidx: InstIdx, _inst: &Inst) {
        assert_eq!(iidx.index(), self.known_bits.len());
        self.known_bits.push(self.pending_commit.clone());
    }
}

impl KnownBits {
    /// Create an empty known bits analysis object.
    pub(super) fn new() -> Self {
        KnownBits {
            known_bits: IndexVec::new(),
            pending_commit: None,
        }
    }

    /// Returns what we know about the bits of `iidx`.
    fn as_knownbits(&self, opt: &PassOpt, iidx: InstIdx) -> Option<KnownBitValue> {
        match opt.inst(iidx).ty(opt) {
            Ty::Func(_) => None,
            Ty::Void => None,
            ty => Some(
                self.known_bits[iidx]
                    .clone()
                    .unwrap_or_else(|| KnownBitValue::unknown(ty.bitw())),
            ),
        }
    }

    fn set_pending(&mut self, bits: KnownBitValue) {
        self.pending_commit = Some(bits);
    }

    fn opt_and(&mut self, opt: &mut PassOpt, mut inst: And) -> OptOutcome {
        inst.canonicalise(opt);
        let And { tyidx, lhs, rhs } = inst;
        if let Some(lhs_b) = self.as_knownbits(opt, lhs)
            && let Some(rhs_b) = self.as_knownbits(opt, rhs)
        {
            let res = lhs_b.bitand(&rhs_b);
            self.set_pending(res.clone());

            // If we know the output's bits, emit that.
            if res.all_known() {
                return OptOutcome::Rewritten(Inst::Const(Const {
                    tyidx,
                    kind: ConstKind::Int(res.as_arbbitint()),
                }));
            }

            // The `and` operation adds new information in the form of set zero. E.g. `unknown &
            // (~1)` zeroes the least significant bit. If the result has no new set zeroes,
            // that means this op is useless.
            if rhs_b.all_known()
                && rhs_b
                    .zeroes()
                    .bitand(&lhs_b.known_ones().bitor(&lhs_b.unknowns))
                    .count_ones()
                    == 0
            {
                return OptOutcome::Equiv(lhs);
            }
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_const(&mut self, inst: Const) -> OptOutcome {
        let Const { tyidx: _, kind } = &inst;
        if let ConstKind::Int(kind) = kind {
            self.set_pending(KnownBitValue::from_const(kind.clone()))
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_or(&mut self, opt: &PassOpt, mut inst: Or) -> OptOutcome {
        inst.canonicalise(opt);
        let Or {
            tyidx,
            lhs,
            rhs,
            disjoint: _,
        } = inst;
        if let Some(lhs_b) = self.as_knownbits(opt, lhs)
            && let Some(rhs_b) = self.as_knownbits(opt, rhs)
        {
            let res = lhs_b.bitor(&rhs_b);
            self.set_pending(res.clone());

            // If we know the output's bits, emit that.
            if res.all_known() {
                return OptOutcome::Rewritten(Inst::Const(Const {
                    tyidx,
                    kind: ConstKind::Int(res.as_arbbitint()),
                }));
            }

            // The `or` operation adds new information in the form of set ones. E.g. `unknown | (1)`
            // sets the least significant bit to one. If the result has no new set ones, that means
            // this op is useless.
            if rhs_b.all_known()
                && rhs_b
                    .known_ones()
                    .bitand(&lhs_b.zeroes().bitor(&lhs_b.unknowns))
                    .count_ones()
                    == 0
            {
                return OptOutcome::Equiv(lhs);
            }
        }
        OptOutcome::Rewritten(inst.into())
    }
}

/// Known bits for a single value.
///
/// In short:
/// | one | unknown | knownbit |
/// |-----|---------|----------|
/// | 0   | 1       | ?        |
/// | 0   | 0       | 0        |
/// | 1   | 0       | 1        |
/// | 1   | 1       | illegal  |
///
/// To ensure monotonicity,transitions from `?` to` 0` or `1` are valid, but not the other way
/// around. `illegal` occurs when both `0` and `1` are set and known, which is impossible in a
/// valid program. `illegal` indicates a likely bug in the optimizer/IR.
#[derive(Clone)]
struct KnownBitValue {
    ones: ArbBitInt,
    unknowns: ArbBitInt,
}

impl KnownBitValue {
    /// Constructs a KnownBitValue from a constant.
    fn from_const(num: ArbBitInt) -> Self {
        let bitw = num.bitw();
        KnownBitValue {
            ones: num,
            unknowns: ArbBitInt::from_u64(bitw, 0),
        }
    }

    /// Constructs an unknown KnownBitValue.
    pub fn unknown(bitw: u32) -> Self {
        KnownBitValue {
            ones: ArbBitInt::from_u64(bitw, 0),
            unknowns: ArbBitInt::from_u64(bitw, u64::MAX),
        }
    }

    /// If all bits are known, return the constant value.
    ///
    /// # Panics
    ///
    /// If the bits are not all known.
    fn as_arbbitint(&self) -> ArbBitInt {
        assert!(self.all_known());
        self.ones.clone()
    }

    /// Returns true if all bits are known.
    fn all_known(&self) -> bool {
        self.unknowns.count_ones() == 0
    }

    /// Return an integer containing all the bits that are known.
    fn knowns(&self) -> ArbBitInt {
        self.unknowns.bitneg()
    }

    /// Returns an integer with all the known zeroes flipped to ones.
    fn zeroes(&self) -> ArbBitInt {
        self.knowns().bitand(&self.ones.bitneg())
    }

    /// Returns an integer with all the known ones.
    fn known_ones(&self) -> ArbBitInt {
        self.knowns().bitand(&self.ones)
    }

    fn bitand(&self, other: &KnownBitValue) -> KnownBitValue {
        let set_ones = self.ones.bitand(&other.ones);
        let set_zeroes = self.zeroes().bitor(&other.zeroes());
        let unknowns = self
            .unknowns
            .bitor(&other.unknowns)
            .bitand(&set_zeroes.bitneg());
        KnownBitValue {
            ones: set_ones,
            unknowns,
        }
    }

    fn bitor(&self, other: &KnownBitValue) -> KnownBitValue {
        let set_ones = self.ones.bitor(&other.ones);
        let unknowns = self
            .unknowns
            .bitor(&other.unknowns)
            .bitand(&set_ones.bitneg());
        KnownBitValue {
            ones: set_ones,
            unknowns,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::{fullopt::test::opt_and_test, strength_fold::StrengthFold};
    use std::{cell::RefCell, rc::Rc};

    fn test_known_bits(mod_s: &str, ptn: &str) {
        let known_bits = Rc::new(RefCell::new(KnownBits::new()));
        let strength_fold = Rc::new(RefCell::new(StrengthFold::new()));
        opt_and_test(
            mod_s,
            |opt, mut inst| {
                inst.canonicalise(opt);
                match known_bits.borrow_mut().feed(opt, inst) {
                    OptOutcome::Rewritten(new_inst) => {
                        strength_fold.borrow_mut().feed(opt, new_inst)
                    }
                    x => x,
                }
            },
            |opt, iidx, inst| known_bits.borrow_mut().inst_committed(opt, iidx, inst),
            ptn,
        );
    }

    #[test]
    fn opt_and() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 3
          %3: i8 = and %0, %1
          %4: i8 = and %3, %2
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 3
          %3: i8 = and %0, %1
          blackbox %3
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 3
          %2: i8 = 1
          %3: i8 = and %0, %1
          %4: i8 = and %3, %2
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = 3
          %2: i8 = 1
          %3: i8 = and %0, %1
          %4: i8 = and %3, %2
          blackbox %4
        ",
        );
    }

    #[test]
    fn opt_or() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          %6: i8 = or %5, %3
          blackbox %6
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          blackbox %5
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          blackbox %5
        ",
        );
    }

    #[test]
    fn opt_constant() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 3
          %2: i8 = or %0, %1
          %3: i8 = 1
          %4: i8 = and %2, %3
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = 3
          %2: i8 = or %0, %1
          %3: i8 = 1
          %4: i8 = 1
          blackbox %4
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 3
          %2: i8 = and %0, %1
          %3: i8 = 3
          %4: i8 = or %2, %3
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = 3
          %2: i8 = and %0, %1
          %3: i8 = 3
          %4: i8 = 3
          blackbox %4
        ",
        );
    }
}
