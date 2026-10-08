//! Known bits analysis.
//!
//! Data-flow analysis to gain information about bits. This is heavily influenced by the
//! [PyPy blog post](https://pypy.org/posts/2024/08/toy-knownbits.html)

use crate::compile::{
    j2::{
        hir::*,
        opt::{
            BlockLikeT, EquivIIdxT,
            fullopt::{CommitInstOpt, OptOutcome, PassOpt, PassT},
        },
    },
    jitc_yk::arbbitint::ArbBitInt,
};
use index_type::vec::TypedVec;
use smallvec::SmallVec;

/// Known-bits analysis.
pub(super) struct KnownBits {
    /// Maps an SSA value to its corresponding known bits.  The value is None by default when
    /// unpopulated. When querying for the value, a None value is returned as a `KnownBitValue`
    /// with all bits set to unknown.
    known_bits: TypedVec<InstIdx, Option<KnownBitValue>>,
    /// The KnownBitValue of the current instruction being processed. This is only committed at
    /// the end of the instruction's analysis.
    pending_commit: Option<KnownBitValue>,
}

impl PassT for KnownBits {
    fn feed(&mut self, opt: &mut PassOpt, mut inst: Inst) -> OptOutcome {
        self.pending_commit = None;

        // Canonicalise `sext`s when we've subsequently learnt that the `sext` must be
        // non-negative. The correctness of this transformation depends on known_bits being the
        // first pass: if it is not the first pass, this transformation will be unsafe for
        // `guard`s.
        let mut sexts = SmallVec::<[_; 2]>::new();
        for iidx in inst.iter_iidxs(opt) {
            if let Inst::SExt(SExt { tyidx, val }) = opt.inst(opt.equiv_iidx(iidx)) {
                let val = opt.equiv_iidx(*val);
                if let Some(bits) = &self.as_knownbits(opt, val)
                    && bits
                        .zeroes()
                        .checked_lshr(bits.bitw() - 1)
                        .unwrap()
                        .to_zero_ext_u8()
                        == Some(1)
                    && !sexts.iter().any(|(x, _)| *x == iidx)
                {
                    sexts.push((iidx, Inst::ZExt(ZExt { tyidx: *tyidx, val })));
                }
            }
        }
        if !sexts.is_empty() {
            let mut map = SmallVec::<[_; 2]>::new();
            for (old_iidx, new_inst) in sexts {
                let new_iidx = opt.push_pre_inst(new_inst);
                map.push((old_iidx, new_iidx));
            }

            inst.rewrite_iidxs(opt, |x| {
                map.iter()
                    .find_map(|(y, z)| if x == *y { Some(*z) } else { None })
                    .unwrap_or(x)
            });

            return OptOutcome::Rerun(inst);
        }

        match inst {
            Inst::AShr(x) => self.opt_ashr(opt, x),
            Inst::And(x) => self.opt_and(opt, x),
            Inst::Const(x) => self.opt_const(x),
            Inst::Guard(x) => self.opt_guard(opt, x),
            Inst::ICmp(x) => self.opt_icmp(opt, x),
            Inst::LShr(x) => self.opt_lshr(opt, x),
            Inst::Or(x) => self.opt_or(opt, x),
            Inst::SExt(x) => self.opt_sext(opt, x),
            Inst::Shl(x) => self.opt_shl(opt, x),
            Inst::Xor(x) => self.opt_xor(opt, x),
            Inst::ZExt(x) => self.opt_zext(opt, x),
            _ => OptOutcome::Rewritten(inst),
        }
    }

    fn preinst_committed(&mut self, opt: &CommitInstOpt, iidx: InstIdx) {
        if let Inst::Const(Const {
            kind: ConstKind::Int(x),
            ..
        }) = opt.inst(iidx)
        {
            self.known_bits
                .push(Some(KnownBitValue::from_const(x.to_owned())));
        } else {
            self.known_bits.push(None);
        }
    }

    fn inst_committed(&mut self, opt: &CommitInstOpt, iidx: InstIdx) {
        if let Inst::Const(Const {
            kind: ConstKind::Int(x),
            ..
        }) = opt.inst(iidx)
        {
            self.known_bits
                .push(Some(KnownBitValue::from_const(x.to_owned())));
        } else {
            self.known_bits.push(self.pending_commit.take());
        }
    }

    fn equiv_committed(&mut self, equiv1: InstIdx, equiv2: InstIdx) {
        self.known_bits[equiv1] = self.known_bits[equiv2].clone();
    }

    fn prepare_for_peel(
        &mut self,
        opt: &mut PassOpt,
        entry: &Block,
        map: &TypedVec<InstIdx, InstIdx>,
    ) {
        assert!(self.pending_commit.is_none());
        let mut new = TypedVec::with_capacity(entry.insts_len());
        for iidx in entry.term_vars().iter().cloned() {
            if let Some(ConstKind::Int(x)) = opt.as_constkind(map[iidx]) {
                new.push(Some(KnownBitValue::from_const(x.clone())));
            } else {
                new.push(self.known_bits[iidx].clone());
            }
        }
        self.known_bits = new;
    }
}

impl KnownBits {
    /// Create an empty known bits analysis object.
    pub(super) fn new() -> Self {
        KnownBits {
            known_bits: TypedVec::new(),
            pending_commit: None,
        }
    }

    /// Returns what we know about the bits of `iidx`.
    fn as_knownbits(&self, opt: &PassOpt, iidx: InstIdx) -> Option<KnownBitValue> {
        match opt.ty(opt.inst(iidx).tyidx(opt)) {
            Ty::Func(_) => None,
            Ty::Void => None,
            ty => Some(
                self.known_bits[iidx]
                    .to_owned()
                    .unwrap_or_else(|| KnownBitValue::unknown(ty.bitw())),
            ),
        }
    }

    /// Updates the known bits value at `iidx` with `other`.
    fn knownbits_set(&mut self, iidx: InstIdx, other: KnownBitValue) {
        self.known_bits[iidx] = Some(other);
    }

    fn set_pending(&mut self, bits: KnownBitValue) {
        self.pending_commit = Some(bits);
    }

    fn opt_ashr(&mut self, opt: &mut PassOpt, inst: AShr) -> OptOutcome {
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

    fn opt_guard(
        &mut self,
        opt: &mut PassOpt,
        inst @ Guard { expect, cond, .. }: Guard,
    ) -> OptOutcome {
        if let Some(cond_b) = self.as_knownbits(opt, cond)
            && cond_b.all_known()
        {
            let x = cond_b.as_arbbitint();
            if (expect && x.to_zero_ext_u8() == Some(0))
                || (!expect && x.to_zero_ext_u8() == Some(1))
            {
                // Let contradictions pass through.
                return OptOutcome::Rewritten(inst.into());
            }
            return OptOutcome::NotNeeded;
        } else if expect
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
            if rhs == lhs {
                return OptOutcome::NotNeeded;
            }
            if let Some(lhs_b) = self.as_knownbits(opt, lhs)
                && let Some(rhs_b) = self.as_knownbits(opt, rhs)
            {
                let union = lhs_b.union(&rhs_b);
                // We deduced a constant. Set future values to point to it.
                if union.all_known() {
                    let tyidx = opt.push_ty(Ty::Int(union.bitw())).unwrap();
                    let idx = opt.push_pre_inst(Inst::Const(Const {
                        tyidx,
                        kind: ConstKind::Int(union.as_arbbitint()),
                    }));
                    opt.push_equiv(lhs, idx);
                    opt.push_equiv(rhs, idx);
                }
                self.knownbits_set(lhs, union.clone());
                self.knownbits_set(rhs, union);
            }

            if let Inst::And(And {
                lhs: and_lhs,
                rhs: and_rhs,
                ..
            }) = opt.inst(lhs).to_owned()
                && let Some(ConstKind::Int(mask)) = opt.as_constkind(and_rhs)
                && let Some(ConstKind::Int(x)) = opt.as_constkind(rhs)
                && x.bitand(&mask.bitneg()).to_zero_ext_u8() == Some(0)
                && let Some(bits) = self.as_knownbits(opt, and_lhs)
            {
                // `(x & c1) == c2` gives us more information about `x`'s known bits.
                let bits = bits.union(&KnownBitValue {
                    ones: x,
                    unknowns: mask.bitneg(),
                });
                if let Inst::ZExt(ZExt { val, .. }) = opt.inst(and_lhs).to_owned() {
                    // A `zext` implicitly tells us about the source's low bits.
                    let val = opt.equiv_iidx(val);
                    if let Some(lhs_bits) = self.as_knownbits(opt, val) {
                        let bitw = lhs_bits.bitw();
                        self.knownbits_set(
                            val,
                            lhs_bits.union(&KnownBitValue {
                                ones: bits.ones.truncate(bitw),
                                unknowns: bits.unknowns.truncate(bitw),
                            }),
                        );
                    }
                }
                self.knownbits_set(and_lhs, bits);
            }
        } else if expect
            && let Inst::ICmp(ICmp { pred, lhs, rhs, .. }) = opt.inst(cond).to_owned()
            && matches!(pred, IPred::Sgt | IPred::Sge)
            && let Some(ConstKind::Int(rhs)) = opt.as_constkind(rhs)
            && rhs.to_sign_ext_i64().is_some_and(|rhs| {
                (pred == IPred::Sgt && rhs >= -1) || (pred == IPred::Sge && rhs >= 0)
            })
            && let Some(mut lhs_b) = self.as_knownbits(opt, lhs)
        {
            // When we guard against `sge`/`sgt`, we implicitly learn what the value's sign bit is.
            let sign = ArbBitInt::from_u64(lhs_b.bitw(), 1 << (lhs_b.bitw() - 1));
            lhs_b.unknowns = lhs_b.unknowns.bitand(&sign.bitneg());
            self.knownbits_set(lhs, lhs_b);
        }

        self.knownbits_set(
            cond,
            KnownBitValue::from_const(ArbBitInt::from_u64(1, u64::from(expect))),
        );

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
            let result = match pred {
                IPred::Eq => lhs_b.definitely_ne(&rhs_b).then_some(false),
                IPred::Ne => lhs_b.definitely_ne(&rhs_b).then_some(true),
                _ => {
                    let sign = if pred.is_signed() {
                        1 << (lhs_b.bitw() - 1)
                    } else {
                        0
                    };
                    let lhs_unknowns = lhs_b.unknowns.to_zero_ext_u64().unwrap();
                    let rhs_unknowns = rhs_b.unknowns.to_zero_ext_u64().unwrap();
                    let lmin = (lhs_b.ones.to_zero_ext_u64().unwrap() ^ sign) & !lhs_unknowns;
                    let lmax = lmin | lhs_unknowns;
                    let rmin = (rhs_b.ones.to_zero_ext_u64().unwrap() ^ sign) & !rhs_unknowns;
                    let rmax = rmin | rhs_unknowns;
                    match pred {
                        IPred::Ugt | IPred::Sgt if lmin > rmax => Some(true),
                        IPred::Ugt | IPred::Sgt if lmax <= rmin => Some(false),
                        IPred::Uge | IPred::Sge if lmin >= rmax => Some(true),
                        IPred::Uge | IPred::Sge if lmax < rmin => Some(false),
                        IPred::Ult | IPred::Slt if lmax < rmin => Some(true),
                        IPred::Ult | IPred::Slt if lmin >= rmax => Some(false),
                        IPred::Ule | IPred::Sle if lmax <= rmin => Some(true),
                        IPred::Ule | IPred::Sle if lmin > rmax => Some(false),
                        _ => None,
                    }
                }
            };
            if let Some(result) = result {
                let tyidx = opt.push_ty(Ty::Int(1)).unwrap();
                return OptOutcome::Rewritten(Inst::Const(Const {
                    tyidx,
                    kind: ConstKind::Int(ArbBitInt::from_u64(1, u64::from(result))),
                }));
            }
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_lshr(&mut self, opt: &mut PassOpt, inst: LShr) -> OptOutcome {
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

    fn opt_or(&mut self, opt: &mut PassOpt, mut inst: Or) -> OptOutcome {
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

    fn opt_sext(&mut self, opt: &mut PassOpt, inst: SExt) -> OptOutcome {
        let SExt { tyidx, val } = inst;
        if let Some(val_b) = self.as_knownbits(opt, val) {
            let dst_bitw = opt.ty(tyidx).bitw();
            let res = val_b.sign_extend(dst_bitw);
            self.set_pending(res.clone());

            let sign = ArbBitInt::from_u64(val_b.bitw(), 1 << (val_b.bitw() - 1));
            if val_b.zeroes().bitand(&sign) == sign {
                // Canonicalise `sext` to `zext` when we know the value must be positive: in
                // general, `zext` leads to more efficient code, and the fewer times we mix `sext`
                // and `zext` the better.
                return OptOutcome::Rewritten(ZExt { tyidx, val }.into());
            }
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_shl(&mut self, opt: &mut PassOpt, inst: Shl) -> OptOutcome {
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

    fn opt_xor(&mut self, opt: &mut PassOpt, inst: Xor) -> OptOutcome {
        let Xor { tyidx, lhs, rhs } = inst;
        if let Some(lhs) = self.as_knownbits(opt, lhs)
            && let Some(rhs) = self.as_knownbits(opt, rhs)
        {
            let res = lhs.bitxor(&rhs);
            if res.all_known() {
                return OptOutcome::Rewritten(
                    Const {
                        tyidx,
                        kind: ConstKind::Int(res.as_arbbitint()),
                    }
                    .into(),
                );
            }
            self.set_pending(res);
        }
        OptOutcome::Rewritten(inst.into())
    }

    fn opt_zext(&mut self, opt: &mut PassOpt, inst: ZExt) -> OptOutcome {
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
#[derive(Clone, Debug)]
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

    /// Union all the known ones in `self` with the known ones in `other`.
    fn union(&self, other: &KnownBitValue) -> KnownBitValue {
        let ones = self.ones.bitor(&other.ones);
        let unknowns = self.unknowns.bitand(&other.unknowns);
        KnownBitValue { ones, unknowns }
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

    fn bitxor(&self, other: &KnownBitValue) -> KnownBitValue {
        let unknowns = self.unknowns.bitor(&other.unknowns);
        let ones = self.ones.bitxor(&other.ones).bitand(&unknowns.bitneg());
        KnownBitValue { ones, unknowns }
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
    use crate::compile::j2::opt::{
        cse::CSE,
        fullopt::test::{full_opt_test, user_defined_opt_test},
        strength_fold::StrengthFold,
    };
    use std::{cell::RefCell, rc::Rc};

    fn test_known_bits(mod_s: &str, ptn: &str) {
        let known_bits = Rc::new(RefCell::new(KnownBits::new()));
        let strength_fold = Rc::new(RefCell::new(StrengthFold::new()));
        user_defined_opt_test(
            mod_s,
            |opt, inst| match known_bits.borrow_mut().feed(opt, inst) {
                OptOutcome::Rewritten(new_inst) => strength_fold.borrow_mut().feed(opt, new_inst),
                x => x,
            },
            |opt, iidx| known_bits.borrow_mut().inst_committed(opt, iidx),
            |equiv1, equiv2| known_bits.borrow_mut().equiv_committed(equiv1, equiv2),
            ptn,
        );
    }

    fn test_known_bits_with_cse(mod_s: &str, ptn: &str) {
        let known_bits = Rc::new(RefCell::new(KnownBits::new()));
        let strength_fold = Rc::new(RefCell::new(StrengthFold::new()));
        let cse = Rc::new(RefCell::new(CSE::new()));
        user_defined_opt_test(
            mod_s,
            |opt, mut inst| {
                if let Inst::Guard(_) = inst {
                    strength_fold.borrow_mut().feed(opt, inst)
                } else {
                    inst.canonicalise(opt);
                    match cse.borrow_mut().feed(opt, inst) {
                        OptOutcome::Rewritten(new_inst) => {
                            known_bits.borrow_mut().feed(opt, new_inst)
                        }
                        x => x,
                    }
                }
            },
            |opt, iidx| {
                cse.borrow_mut().inst_committed(opt, iidx);
                known_bits.borrow_mut().inst_committed(opt, iidx);
            },
            |equiv1, equiv2| {
                cse.borrow_mut().equiv_committed(equiv1, equiv2);
                known_bits.borrow_mut().equiv_committed(equiv1, equiv2);
            },
            ptn,
        );
    }

    #[test]
    fn peeling() {
        // Optimise in the peel based on known bits: all we know about `%5` is that it doesn't have
        // the least significant bit set.
        full_opt_test(
            r#"
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i1 = icmp ne %0, %1
          guard true, %2, []
          %4: i8 = 254
          %5: i8 = and %0, %4
          term [%5]
        "#,
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i1 = icmp ne %0, %1
          guard true, %2, []
          %5: i8 = 254
          %6: i8 = and %0, %5
          term [%6]
          ; peel
          %0: i8 = arg
          term [%0]
        ",
        );
    }

    // Individual instructions

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
          %4: i8 = 1
          %5: i8 = and %0, %4
          blackbox %5
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
          %5: i8 = 3
          %6: i8 = or %0, %5
          blackbox %6
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
          %5: i8 = 3
          %6: i8 = or %0, %5
          blackbox %6
        ",
        );
    }

    #[test]
    fn opt_xor() {
        test_known_bits(
            "
           %0: i8 = arg [reg]
           %1: i16 = zext %0
           %2: i16 = 5
           %3: i16 = xor %1, %2
           %4: i16 = 0
           %5: i1 = icmp slt %3, %4
           blackbox %5
         ",
            "
           ...
           %5: i1 = 0
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
    fn opt_guard() {
        // Known bits that passed through guard is correct for `or`.
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = 1
          %3: i8 = or %1, %2
          %4: i1 = icmp eq %3, %0
          guard true, %4, []
          %6: i8 = or %0, %2
          blackbox %6
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = 1
          %3: i8 = or %1, %2
          %4: i1 = icmp eq %3, %0
          %5: i1 = 1
          guard true, %4, []
          blackbox %3
        ",
        );

        // Known bits guard sets `icmp`'s result.
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp eq %0, %1
          guard false, %2, []
          guard false, %2, []
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = icmp eq %0, %1
          %3: i1 = 0
          guard false, %2, []
          ...
        ",
        );

        // Known bits canonicalises `icmp`.
        test_known_bits_with_cse(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          %4: i8 = 1
          %5: i8 = or %0, %4
          %6: i8 = or %1, %4
          %7: i1 = icmp eq %5, %6
          guard true, %7, []
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = icmp eq %0, %1
          %3: i1 = 1
          guard true, %2, []
          %5: i8 = 1
          %6: i8 = or %0, %5
          %7: i1 = icmp eq %6, %6
        ",
        );

        // Guard deduced constant in instruction stream
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = 15
          %3: i8 = 240
          %4: i8 = or %0, %2
          %5: i8 = or %1, %3
          %6: i1 = icmp eq %4, %5
          guard true, %6, []
          %8: i32 = sext %4
          blackbox %8
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = 15
          %3: i8 = 240
          %4: i8 = or %0, %2
          %5: i8 = or %1, %3
          %6: i1 = icmp eq %4, %5
          %7: i8 = 255
          %8: i1 = 1
          guard true, %6, []
          %10: i32 = 4294967295
          blackbox %10
        ",
        );

        // Guards and `and`.
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 15
          %2: i8 = and %0, %1
          %3: i8 = 5
          %4: i1 = icmp eq %2, %3
          guard true, %4, []
          %6: i8 = 7
          %7: i8 = and %0, %6
          blackbox %7
        ",
            "
          %0: i8 = arg
          %1: i8 = 15
          %2: i8 = and %0, %1
          %3: i8 = 5
          %4: i1 = icmp eq %2, %3
          %5: i8 = 5
          %6: i1 = 1
          guard true, %4, []
          %8: i8 = 7
          %9: i8 = 5
          blackbox %9
        ",
        );

        // Guards and `and` and `zext`
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i32 = zext %0
          %2: i32 = 63
          %3: i32 = and %1, %2
          %4: i32 = 19
          %5: i1 = icmp eq %3, %4
          guard true, %5, []
          %7: i8 = 3
          %8: i1 = icmp eq %0, %7
          blackbox %8
          %10: i8 = 192
          %11: i8 = and %0, %10
          blackbox %11
        ",
            "
          %0: i8 = arg
          %1: i32 = zext %0
          %2: i32 = 63
          %3: i32 = and %1, %2
          %4: i32 = 19
          %5: i1 = icmp eq %3, %4
          %6: i32 = 19
          %7: i1 = 1
          guard true, %5, []
          %9: i8 = 3
          %10: i1 = 0
          blackbox %10
          %12: i8 = 192
          %13: i8 = and %0, %12
          blackbox %13
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

        // The non-eq-ne predicates
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = 7
          %3: i8 = and %0, %2
          %4: i8 = 8
          %5: i8 = or %1, %4
          %6: i1 = icmp ult %3, %5
          %7: i1 = icmp uge %3, %5
          blackbox %6
          blackbox %7
        ",
            "
          ...
          %6: i1 = 1
          %7: i1 = 0
          blackbox %6
          blackbox %7
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 128
          %2: i8 = or %0, %1
          %3: i8 = 0
          %4: i1 = icmp sgt %2, %3
          %5: i1 = icmp sle %2, %3
          blackbox %4
          blackbox %5
        ",
            "
          ...
          %4: i1 = 0
          %5: i1 = 1
          blackbox %4
          blackbox %5
        ",
        );

        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 7
          %2: i8 = and %0, %1
          %3: i1 = icmp ugt %2, %1
          %4: i1 = icmp ule %2, %1
          %5: i1 = icmp sge %2, %1
          %6: i1 = icmp slt %2, %1
          blackbox %3
          blackbox %4
          blackbox %5
          blackbox %6
        ",
            "
          ...
          %3: i1 = 0
          %4: i1 = 1
          %5: i1 = icmp sge %2, %1
          %6: i1 = icmp slt %2, %1
          blackbox %3
          blackbox %4
          blackbox %5
          blackbox %6
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

        // sext -> zext canonicalisation
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
          %3: i16 = zext %2
          %4: i16 = 32768
          %5: i16 = 0
          blackbox %5
        ",
        );

        // Deduce that a `sext` can be canonicalised by a later guard to a `zext`.
        test_known_bits(
            "
          %0: i32 = arg [reg]
          %1: i64 = sext %0
          blackbox %1
          %3: i32 = 0
          %4: i1 = icmp sge %0, %3
          guard true, %4, [%1]
          blackbox %1
        ",
            "
          %0: i32 = arg
          %1: i64 = sext %0
          blackbox %1
          %3: i32 = 0
          %4: i1 = icmp sge %0, %3
          %5: i1 = 1
          guard true, %4, [%1]
          %7: i64 = zext %0
          blackbox %7
        ",
        );
    }

    #[test]
    fn guarded_cast_constants() {
        // The source becomes constant after the casts were committed. Its canonical value must
        // be truncated, sign extended, or zero extended at each use, keeping deoptimisation intact.
        test_known_bits(
            "
          %0: i16 = arg [reg]
          %1: i8 = trunc %0
          %2: i32 = sext %0
          %3: i32 = zext %0
          %4: i16 = 65535
          %5: i1 = icmp eq %0, %4
          guard true, %5, [%1, %2, %3]
          %7: i8 = 1
          %8: i8 = add %1, %7
          blackbox %8
          %10: i32 = 2
          %11: i32 = add %2, %10
          blackbox %11
          %13: i32 = add %3, %10
          blackbox %13
        ",
            "
          %0: i16 = arg
          %1: i8 = trunc %0
          %2: i32 = sext %0
          %3: i32 = zext %0
          %4: i16 = 65535
          %5: i1 = icmp eq %0, %4
          %6: i16 = 65535
          %7: i1 = 1
          guard true, %5, [%1, %2, %3]
          %9: i8 = 1
          %10: i8 = 0
          blackbox %10
          %12: i32 = 2
          %13: i32 = 1
          blackbox %13
          %15: i32 = 65537
          blackbox %15
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
