pub(crate) trait SignExtend {
    /// Sign extend an `from-bits`-bit number stored in `self` up to a `to-bits`-bit number.
    ///
    /// "Unused" higher-order bits `Self::BITS..=to_bits` are zeroed.
    ///
    /// The following cases are undefined:
    /// - `from_bits` outside the range `1..=Self::BITS`.
    /// - `to_bits` outside the range `1..=Self::BITS`.
    /// - `from_bits` >= `to_bits`.
    fn sign_extend(&self, from_bits: u32, to_bits: u32) -> Self;
}

impl SignExtend for u64 {
    fn sign_extend(&self, from_bits: u32, to_bits: u32) -> Self {
        debug_assert!(
            from_bits > 0 && from_bits <= Self::BITS,
            "to_bits {to_bits} outside range 1..={}",
            Self::BITS
        );
        debug_assert!(
            to_bits > 0 && to_bits <= Self::BITS,
            "to_bits {to_bits} outside range 1..={}",
            Self::BITS
        );
        debug_assert!(from_bits <= to_bits);
        // There are probably more clever ways to do this.
        if from_bits == to_bits {
            *self
        } else if self & (1 << (from_bits - 1)) == 0 {
            // Extend with zeros.
            let shift = Self::BITS - from_bits;
            (*self << shift) >> shift
        } else {
            // Extend with ones.
            // How many high-order zero bits do we need?
            let num_zeros = Self::BITS - to_bits;
            let mask = ((Self::MAX << from_bits) << num_zeros) >> num_zeros;
            *self | mask
        }
    }
}

pub(crate) trait Truncate {
    /// Truncate the value to a `bits`-bit value by unsetting higher order bits.
    ///
    /// `bits` must be in the range `1..=Self::BITS`.
    fn truncate(&self, bits: u32) -> Self;
}

impl Truncate for u64 {
    fn truncate(&self, bits: u32) -> Self {
        debug_assert!(
            bits > 0 && bits <= Self::BITS,
            "{bits} outside range 1..{}",
            Self::BITS
        );
        if bits == Self::BITS {
            *self
        } else {
            *self & ((1 as Self).wrapping_shl(bits) - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SignExtend, Truncate};

    #[test]
    fn sign_extend() {
        assert_eq!(0xffffffffu64.sign_extend(32, 64), u64::MAX);
        assert_eq!(0xffffffffu64.sign_extend(32, 64), u64::MAX);
        assert_eq!(u64::MAX.sign_extend(64, 64), u64::MAX);
        assert_eq!(0xffffu64.sign_extend(16, 32), 0xffffffffu64);
        assert_eq!(0x7fffu64.sign_extend(16, 32), 0x7fffu64);
        assert_eq!(0xffu64.sign_extend(8, 8), 0xff);
        assert_eq!(0x00u64.sign_extend(8, 8), 0x00);
        for i in i8::MIN..i8::MAX {
            // cast the value up to the backing store without sign extend.
            let iu = i as u8 as u64;
            // sign extending up to 16 bits should always give the same numeric value.
            assert_eq!(iu.sign_extend(8, 16) as i16, i as i16);
        }
    }

    #[test]
    fn truncate() {
        assert_eq!(u64::MAX.truncate(8), 0xff);
        assert_eq!(u64::MAX.truncate(16), 0xffff);
        assert_eq!(u64::MAX.truncate(24), 0xffffff);
        assert_eq!(u64::MAX.truncate(32), 0xffffffff);
        assert_eq!(u64::MAX.truncate(63), 0x7fffffffffffffff);
        assert_eq!(u64::MAX.truncate(64), u64::MAX);
        for i in 0u64..255 {
            // These should all be no-ops.
            assert_eq!(i.truncate(8), i);
        }
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn zero_bits() {
        u64::MAX.truncate(0);
    }
}
