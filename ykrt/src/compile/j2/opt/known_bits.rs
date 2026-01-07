//! Known bits analysis.
//!
//! Data-flow analysis to gain information about bits. This is heavily influenced by the
//! [PyPy blog post](https://pypy.org/posts/2024/08/toy-knownbits.html)

use crate::compile::{
    j2::{
        hir::*,
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
            Inst::AShr(x) => self.opt_ashr(opt, x),
            Inst::And(x) => self.opt_and(opt, x),
            Inst::Const(x) => self.opt_const(x),
            Inst::ICmp(x) => self.opt_icmp(opt, x),
            Inst::LShr(x) => self.opt_lshr(opt, x),
            Inst::Or(x) => self.opt_or(opt, x),
            Inst::SExt(x) => self.opt_sext(opt, x),
            Inst::Shl(x) => self.opt_shl(opt, x),
            Inst::ZExt(x) => self.opt_zext(opt, x),
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

    fn opt_ashr(&mut self, opt: &PassOpt, inst: AShr) -> OptOutcome {
        let AShr {
            tyidx: _,
            lhs,
            rhs,
            exact: _,
        } = inst;
        if let Some(lhs_b) = self.as_knownbits(opt, lhs)
            && let Some(ConstKind::Int(rhs_c)) = opt.as_constkind(rhs)
            && let Some(rhs_int) = rhs_c.to_zero_ext_u32()
            && let Some(res) = lhs_b.checked_ashr(rhs_int)
        {
            self.set_pending(res.clone());
        }
        OptOutcome::Rewritten(inst.into())
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

    fn opt_icmp(&mut self, opt: &mut PassOpt, mut inst: ICmp) -> OptOutcome {
        inst.canonicalise(opt);
        let ICmp {
            pred,
            lhs,
            rhs,
            samesign: _samesign,
        } = inst;
        if let Some(lhs_b) = self.as_knownbits(opt, lhs)
            && let Some(rhs_b) = self.as_knownbits(opt, rhs)
        {
            let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
            match pred {
                IPred::Eq if lhs_b.definitely_ne(&rhs_b) => {
                    OptOutcome::Rewritten(Inst::Const(Const {
                        tyidx,
                        kind: ConstKind::Int(ArbBitInt::from_u64(1, 0)),
                    }))
                }
                IPred::Ne if lhs_b.definitely_ne(&rhs_b) => {
                    OptOutcome::Rewritten(Inst::Const(Const {
                        tyidx,
                        kind: ConstKind::Int(ArbBitInt::from_u64(1, 1)),
                    }))
                }
                _ => OptOutcome::Rewritten(inst.into()),
            }
        } else {
            OptOutcome::Rewritten(inst.into())
        }
    }

    fn opt_lshr(&mut self, opt: &PassOpt, inst: LShr) -> OptOutcome {
        let LShr {
            tyidx: _,
            lhs,
            rhs,
            exact: _,
        } = inst;
        if let Some(lhs_b) = self.as_knownbits(opt, lhs)
            && let Some(ConstKind::Int(rhs_c)) = opt.as_constkind(rhs)
            && let Some(rhs_int) = rhs_c.to_zero_ext_u32()
            && let Some(res) = lhs_b.checked_lshr(rhs_int)
        {
            self.set_pending(res.clone());
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

    fn opt_sext(&mut self, opt: &PassOpt, inst: SExt) -> OptOutcome {
        let SExt { tyidx, val } = inst;
        if let Some(val_b) = self.as_knownbits(opt, val) {
            let dst_bitw = opt.ty(tyidx).bitw();
            let res = val_b.sign_extend(dst_bitw);
            self.set_pending(res.clone());
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_shl(&mut self, opt: &PassOpt, inst: Shl) -> OptOutcome {
        let Shl {
            tyidx: _,
            lhs,
            rhs,
            nuw: _,
            nsw: _,
        } = inst;
        if let Some(lhs_b) = self.as_knownbits(opt, lhs)
            && let Some(ConstKind::Int(rhs_c)) = opt.as_constkind(rhs)
            && let Some(rhs_int) = rhs_c.to_zero_ext_u32()
            && let Some(res) = lhs_b.checked_shl(rhs_int)
        {
            self.set_pending(res.clone());
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_zext(&mut self, opt: &PassOpt, inst: ZExt) -> OptOutcome {
        let ZExt { tyidx, val } = inst;
        if let Some(val_b) = self.as_knownbits(opt, val) {
            let dst_bitw = opt.ty(tyidx).bitw();
            let res = val_b.zero_extend(dst_bitw);
            self.set_pending(res.clone());
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

    /// Bitwidth of the underlying value.
    ///
    /// # Panics
    ///
    /// If the ones' and unknowns' bitwidth do not match.
    fn bitw(&self) -> u32 {
        assert_eq!(self.ones.bitw(), self.unknowns.bitw());
        self.ones.bitw()
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

    fn checked_ashr(&self, bits: u32) -> Option<KnownBitValue> {
        let set_ones = self.ones.checked_ashr(bits)?;
        let unknowns = self.unknowns.checked_ashr(bits)?;
        Some(KnownBitValue {
            ones: set_ones,
            unknowns,
        })
    }

    fn checked_lshr(&self, bits: u32) -> Option<KnownBitValue> {
        let set_ones = self.ones.checked_lshr(bits)?;
        let unknowns = self.unknowns.checked_lshr(bits)?;
        Some(KnownBitValue {
            ones: set_ones,
            unknowns,
        })
    }

    fn checked_shl(&self, bits: u32) -> Option<KnownBitValue> {
        let set_ones = self.ones.checked_shl(bits)?;
        let unknowns = self.unknowns.checked_shl(bits)?;
        Some(KnownBitValue {
            ones: set_ones,
            unknowns,
        })
    }

    /// Two `KnownBitValue` are not equal only if the known parts of their bits are different.
    ///
    /// Note that we do not implement the `PartialEq` trait because this violates the invariant
    /// of the that trait that `eq == !ne`. In this case, `!ne` does not mean `eq`.
    ///
    /// # Panics
    ///
    /// If the bitwidths are different.
    fn definitely_ne(&self, other: &Self) -> bool {
        assert_eq!(self.bitw(), other.bitw());
        let knowns = self.knowns().bitand(&other.knowns());
        knowns.bitand(&self.ones) != knowns.bitand(&other.ones)
    }

    fn sign_extend(&self, bitw: u32) -> KnownBitValue {
        let set_ones = self.ones.sign_extend(bitw);
        let unknowns = self.unknowns.sign_extend(bitw);
        KnownBitValue {
            ones: set_ones,
            unknowns,
        }
    }

    fn zero_extend(&self, bitw: u32) -> KnownBitValue {
        let set_ones = self.ones.zero_extend(bitw);
        let unknowns = self.unknowns.zero_extend(bitw);
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
    fn opt_ashr() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 2
          %4: i8 = ashr %2, %3
          %5: i8 = 96
          %6: i8 = or %4, %5
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 2
          %4: i8 = ashr %2, %3
          %5: i8 = 96
          blackbox %5
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 64
          %2: i8 = or %0, %1
          %3: i8 = 2
          %4: i8 = ashr %2, %3
          %5: i8 = 96
          %6: i8 = or %4, %5
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 64
          %2: i8 = or %0, %1
          %3: i8 = 2
          %4: i8 = ashr %2, %3
          %5: i8 = 96
          %6: i8 = or %4, %5
          blackbox %5
        ",
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

    #[test]
    fn opt_icmp() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = and %0, %1
          %3: i8 = 1
          %4: i1 = icmp eq %2, %3
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = and %0, %1
          %3: i8 = 1
          %4: i1 = 0
          blackbox %3
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = and %0, %1
          %3: i1 = icmp eq %2, %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = and %0, %1
          %3: i1 = icmp eq %2, %1
          blackbox %3
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = and %0, %1
          %3: i8 = 1
          %4: i1 = icmp ne %2, %3
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = and %0, %1
          %3: i8 = 1
          %4: i1 = 1
          blackbox %3
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = and %0, %1
          %3: i1 = icmp ne %2, %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = and %0, %1
          %3: i1 = icmp ne %2, %1
          blackbox %3
        ",
        );
    }

    #[test]
    fn opt_lshr() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 1
          %4: i8 = lshr %2, %3
          %5: i8 = or %4, %1
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 1
          %4: i8 = lshr %2, %3
          %5: i8 = or %4, %1
          blackbox %5
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 2
          %4: i8 = lshr %2, %3
          %5: i8 = 32
          %6: i8 = or %4, %5
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 2
          %4: i8 = lshr %2, %3
          %5: i8 = 32
          blackbox %5
        ",
        );
    }

    #[test]
    fn opt_sext() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i16 = sext %2
          %4: i16 = 32768
          %5: i16 = or %3, %4
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i16 = sext %2
          %4: i16 = 32768
          blackbox %3
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = and %0, %1
          %3: i16 = sext %2
          %4: i16 = 32768
          %5: i16 = and %3, %4
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = and %0, %1
          %3: i16 = sext %2
          %4: i16 = 32768
          %5: i16 = 0
          blackbox %5
        ",
        );
    }

    #[test]
    fn opt_shl() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = shl %0, %1
          %3: i8 = and %2, %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = shl %0, %1
          %3: i8 = 0
          blackbox %3
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = shl %0, %1
          %3: i8 = or %2, %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = shl %0, %1
          %3: i8 = or %2, %1
          blackbox %3
        ",
        );
    }

    #[test]
    fn opt_zext() {
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i16 = zext %2
          %4: i16 = 32768
          %5: i16 = and %3, %4
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i16 = zext %2
          %4: i16 = 32768
          %5: i16 = 0
          blackbox %5
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i16 = zext %2
          %4: i16 = 128
          %5: i16 = or %3, %4
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i16 = zext %2
          %4: i16 = 128
          blackbox %3
        ",
        );
    }
}
