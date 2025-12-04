//! An integer of an arbitrary, dynamic, bit width.
//!
//! This hides away the underlying representation, and forces the user to consider whether they
//! want to zero or sign extend the underlying the integer whenever they want access to a
//! Rust-level integer.
//!
//! Currently only up to 64 bits are supported, though the API is flexible enough to transparently
//! support greater bit widths in the future.

use super::int_signs::{SignExtend, Truncate};
use std::{
    fmt,
    hash::Hash,
    ops::{BitAnd, BitOr, BitXor},
};

/// An integer of an arbitrary, dynamic, bit width.
///
/// Currently can only represent a max of 64 bits: this could be extended in the future.
#[derive(Clone, Debug)]
pub(crate) struct ArbBitInt {
    bitw: u32,
    /// The underlying value. Any bits above `self.bitw` have an undefined value: they may be set
    /// or unset.
    ///
    /// Currently we can only store ints that can fit in 64 bits: in the future we could use
    /// another scheme to e.g `Box` bigger integers.
    val: u64,
}

impl ArbBitInt {
    /// Create a new `ArbBitInt` that is `width` bits wide and has a value `val`. Any bits above
    /// `width` bits are ignored (i.e. it is safe for those bits to be set or unset when calling
    /// this function).
    pub(crate) fn from_u64(bitw: u32, val: u64) -> Self {
        debug_assert!(bitw <= 64);
        Self { bitw, val }
    }

    /// Create a new `ArbBitInt` that is `width` bits wide and has a value `val`. Any bits above
    /// `width` bits are ignored (i.e. it is safe for those bits to be set or unset when calling
    /// this function).
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn from_usize(val: usize) -> Self {
        Self {
            bitw: 64,
            val: val as u64,
        }
    }

    /// Create a new `ArbBitInt` that is `width` bits wide and has a value `val`. Any bits above
    /// `width` bits are ignored (i.e. it is safe for those bits to be set or unset when calling
    /// this function).
    #[cfg(test)]
    pub(crate) fn from_i64(bitw: u32, val: i64) -> Self {
        debug_assert!(bitw <= 64);
        Self {
            bitw,
            val: val as u64,
        }
    }

    /// Create a new `ArbBitInt` with all `bitw` bits set. This can be seen as equivalent to
    /// creating a value of `ubitw::MAX` (when `ubitw` is also a valid Rust type).
    pub(crate) fn all_bits_set(bitw: u32) -> Self {
        Self {
            bitw,
            val: u64::MAX,
        }
    }

    /// How many bits wide is this `ArbBitInt`?
    pub(crate) fn bitw(&self) -> u32 {
        self.bitw
    }

    /// Sign extend this `ArbBitInt` to `to_bitw` bits.
    ///
    /// # Panics
    ///
    /// If `to_bitw` is smaller than `self.bitw()`.
    pub(crate) fn sign_extend(&self, to_bitw: u32) -> Self {
        debug_assert!(to_bitw >= self.bitw && to_bitw <= 64);
        Self {
            bitw: to_bitw,
            val: self.val.sign_extend(self.bitw, to_bitw),
        }
    }

    /// Zero extend this `ArbBitInt` to `to_bitw` bits.
    ///
    /// # Panics
    ///
    /// If `to_bitw` is smaller than `self.bitw()`.
    pub(crate) fn zero_extend(&self, to_bitw: u32) -> Self {
        debug_assert!(to_bitw >= self.bitw && to_bitw <= 64);
        Self {
            bitw: to_bitw,
            val: self.val.truncate(self.bitw),
        }
    }

    /// Truncate this `ArbBitInt` to `to_bitw` bits.
    ///
    /// # Panics
    ///
    /// If `to_bitw` is larger than `self.bitw()`.
    pub(crate) fn truncate(&self, to_bitw: u32) -> Self {
        debug_assert!(to_bitw <= self.bitw && to_bitw <= 64);
        Self {
            bitw: to_bitw,
            val: self.val,
        }
    }

    /// Sign extend the underlying value and, if it is representable as an `i8`, return it.
    #[allow(dead_code)]
    pub(crate) fn to_sign_ext_i8(&self) -> Option<i8> {
        i8::try_from(self.val.sign_extend(self.bitw, 64) as i64).ok()
    }

    /// Sign extend the underlying value and, if it is representable as an `i16`, return it.
    #[allow(dead_code)]
    pub(crate) fn to_sign_ext_i16(&self) -> Option<i16> {
        i16::try_from(self.val.sign_extend(self.bitw, 64) as i64).ok()
    }

    /// Sign extend the underlying value and, if it is representable as an `i32`, return it.
    pub(crate) fn to_sign_ext_i32(&self) -> Option<i32> {
        i32::try_from(self.val.sign_extend(self.bitw, 64) as i64).ok()
    }

    /// Sign extend the underlying value and, if it is representable as an `i64`, return it.
    pub(crate) fn to_sign_ext_i64(&self) -> Option<i64> {
        Some(self.val.sign_extend(self.bitw, 64) as i64)
    }

    /// Sign extend the underlying value and, if it is representable as an `isize`, return it.
    #[allow(dead_code)]
    pub(crate) fn to_sign_ext_isize(&self) -> Option<isize> {
        assert_eq!(
            usize::try_from(isize::BITS).unwrap(),
            std::mem::size_of_val(&self.val) * 8
        );
        Some(self.val.sign_extend(self.bitw, isize::BITS) as isize)
    }

    /// zero extend the underlying value and, if it is representable as an `u8`, return it.
    pub(crate) fn to_zero_ext_u8(&self) -> Option<u8> {
        u8::try_from(self.val.truncate(self.bitw)).ok()
    }

    /// zero extend the underlying value and, if it is representable as an `u16`, return it.
    pub(crate) fn to_zero_ext_u16(&self) -> Option<u16> {
        u16::try_from(self.val.truncate(self.bitw)).ok()
    }

    /// zero extend the underlying value and, if it is representable as an `u32`, return it.
    pub(crate) fn to_zero_ext_u32(&self) -> Option<u32> {
        u32::try_from(self.val.truncate(self.bitw)).ok()
    }

    /// zero extend the underlying value and, if it is representable as an `u64`, return it.
    pub(crate) fn to_zero_ext_u64(&self) -> Option<u64> {
        Some(self.val.truncate(self.bitw))
    }

    /// zero extend the underlying value and, if it is representable as an `u64`, return it.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn to_zero_ext_usize(&self) -> Option<usize> {
        Some(self.val.truncate(self.bitw) as usize)
    }

    /// Return a new [ArbBitInt] that performs two's complement wrapping addition on `self` and
    /// `other`.
    ///
    /// # Panics
    ///
    /// If `self` and `other` are not the same bit width.
    pub(crate) fn wrapping_add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.bitw, other.bitw);
        Self {
            bitw: self.bitw,
            val: self.val.wrapping_add(other.val),
        }
    }

    /// Return a new [ArbBitInt] that performs two's complement wrapping multiplication on `self` and
    /// `other`.
    ///
    /// # Panics
    ///
    /// If `self` and `other` are not the same bit width.
    pub(crate) fn wrapping_mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.bitw, other.bitw);
        Self {
            bitw: self.bitw,
            val: self.val.wrapping_mul(other.val),
        }
    }

    /// Return a new [ArbBitInt] that performs two's complement wrapping subtraction on `self` and
    /// `other`.
    ///
    /// # Panics
    ///
    /// If `self` and `other` are not the same bit width.
    pub(crate) fn wrapping_sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.bitw, other.bitw);
        Self {
            bitw: self.bitw,
            val: self.val.wrapping_sub(other.val),
        }
    }

    /// Return a new [ArbBitInt] that left shifts `self` by `bits`s or `None` if `bits >=
    /// self.bitw()`.
    pub(crate) fn checked_shl(&self, bits: u32) -> Option<Self> {
        if bits < self.bitw {
            Some(Self {
                bitw: self.bitw,
                val: self.val.checked_shl(bits).unwrap(), // unwrap cannot fail
            })
        } else {
            None
        }
    }

    /// Return a new [ArbBitInt] that arithmetic-right shifts `self` by `bits` or `None` if `bits
    /// >= self.bitw()`.
    pub(crate) fn checked_ashr(&self, bits: u32) -> Option<Self> {
        if bits < self.bitw {
            Some(Self {
                bitw: self.bitw,
                val: self
                    .val
                    .checked_shr(bits)
                    .unwrap()
                    .sign_extend(self.bitw - bits, self.bitw), // unwrap cannot fail
            })
        } else {
            None
        }
    }

    /// Return a new [ArbBitInt] that logic-right shifts `self` by `bits` or `None` if `bits >=
    /// self.bitw()`.
    pub(crate) fn checked_lshr(&self, bits: u32) -> Option<Self> {
        if bits < self.bitw {
            Some(Self {
                bitw: self.bitw,
                val: self.val.checked_shr(bits).unwrap(), // unwrap cannot fail
            })
        } else {
            None
        }
    }

    /// Return a new [ArbBitInt] that performs bitwise `AND` on `self` and `other`.
    ///
    /// # Panics
    ///
    /// If `self` and `other` are not the same bit width.
    pub(crate) fn bitand(&self, other: &Self) -> Self {
        debug_assert_eq!(self.bitw, other.bitw);
        Self {
            bitw: self.bitw,
            val: self.val.bitand(other.val),
        }
    }

    /// Return a new [ArbBitInt] that performs bitwise `OR` on `self` and `other`.
    ///
    /// # Panics
    ///
    /// If `self` and `other` are not the same bit width.
    pub(crate) fn bitor(&self, other: &Self) -> Self {
        debug_assert_eq!(self.bitw, other.bitw);
        Self {
            bitw: self.bitw,
            val: self.val.bitor(other.val),
        }
    }

    /// Return a new [ArbBitInt] that performs bitwise `XOR` on `self` and `other`.
    ///
    /// # Panics
    ///
    /// If `self` and `other` are not the same bit width.
    pub(crate) fn bitxor(&self, other: &Self) -> Self {
        debug_assert_eq!(self.bitw, other.bitw);
        Self {
            bitw: self.bitw,
            val: self.val.bitxor(other.val),
        }
    }
}

impl fmt::Display for ArbBitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val.truncate(self.bitw))
    }
}

impl Hash for ArbBitInt {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bitw.hash(state);
        self.val.truncate(self.bitw).hash(state);
    }
}

impl PartialEq for ArbBitInt {
    fn eq(&self, other: &Self) -> bool {
        self.bitw == other.bitw && self.val.truncate(self.bitw) == other.val.truncate(self.bitw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn arbbitint_8bit(x in any::<i8>(), y in any::<i8>()) {
            assert_eq!(ArbBitInt::from_i64(8, x as i64).to_sign_ext_i8(), Some(x));

            // wrapping_add
            // i8
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i8(),
                Some(x.wrapping_add(y))
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i16(),
                Some(i16::from(x.wrapping_add(y)))
            );

            // wrapping_sub
            // i8
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i8(),
                Some(x.wrapping_sub(y))
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i16(),
                Some(i16::from(x.wrapping_sub(y)))
            );

            // wrapping_mul
            // i8
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i8(),
                Some(x.wrapping_mul(y))
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i16(),
                Some(i16::from(x.wrapping_mul(y)))
            );

            // bitadd
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .bitand(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i8(),
                Some(x.bitand(y))
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .bitand(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i16(),
                Some(i16::from(x.bitand(y)))
            );

            // bitor
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .bitor(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i8(),
                Some(x.bitor(y))
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .bitor(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i16(),
                Some(i16::from(x.bitor(y)))
            );

            // bitxor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .bitxor(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i8(),
                Some(x.bitxor(y))
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64)
                    .bitxor(&ArbBitInt::from_i64(8, y as i64)).to_sign_ext_i16(),
                Some(i16::from(x.bitxor(y)))
            );
        }

        #[test]
        fn arbbitint_16bit(x in any::<i16>(), y in any::<i16>()) {
            match (i8::try_from(x), ArbBitInt::from_i64(16, x as i64).to_sign_ext_i8()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            assert_eq!(ArbBitInt::from_i64(16, x as i64).to_sign_ext_i16(), Some(x));

            // wrapping_add
            // i8
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_add(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i16(),
                Some(x.wrapping_add(y))
            );

            // wrapping_sub
            // i8
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_sub(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i16(),
                Some(x.wrapping_sub(y))
            );

            // wrapping_mul
            // i8
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_mul(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i16(),
                Some(x.wrapping_mul(y))
            );

            // bitand
            // i8
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .bitand(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.bitand(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .bitand(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i16(),
                Some(x.bitand(y))
            );

            // bitor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .bitor(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.bitor(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .bitor(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i16(),
                Some(x.bitor(y))
            );

            // bitxor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .bitxor(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.bitxor(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64)
                    .bitxor(&ArbBitInt::from_i64(16, y as i64)).to_sign_ext_i16(),
                Some(x.bitxor(y))
            );
        }

        #[test]
        fn arbbitint_32bit(x in any::<i32>(), y in any::<i32>()) {
            match (i8::try_from(x), ArbBitInt::from_i64(32, x as i64).to_sign_ext_i8()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            match (i16::try_from(x), ArbBitInt::from_i64(32, x as i64).to_sign_ext_i16()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            assert_eq!(ArbBitInt::from_i64(32, x as i64).to_sign_ext_i32(), Some(x));

            // wrapping_add
            // i8
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_add(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_add(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_add(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i32(),
                Some(x.wrapping_add(y))
            );

            // wrapping_sub
            // i8
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_sub(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_sub(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_sub(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i32(),
                Some(x.wrapping_sub(y))
            );

            // wrapping_mul
            // i8
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_mul(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_mul(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .wrapping_mul(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i32(),
                Some(x.wrapping_mul(y))
            );

            // bitand
            // i8
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitand(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.bitand(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitand(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i16(),
                i16::try_from(x.bitand(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitand(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i32(),
                Some(x.bitand(y))
            );

            // bitor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitor(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.bitor(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitor(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i16(),
                i16::try_from(x.bitor(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitor(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i32(),
                Some(x.bitor(y))
            );

            // bitxor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitxor(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i8(),
                i8::try_from(x.bitxor(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitxor(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i16(),
                i16::try_from(x.bitxor(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64)
                    .bitxor(&ArbBitInt::from_i64(32, y as i64)).to_sign_ext_i32(),
                Some(x.bitxor(y))
            );
        }

        #[test]
        fn arbbitint_64bit(x in any::<i64>(), y in any::<i64>()) {
            match (i8::try_from(x), ArbBitInt::from_i64(64, x).to_sign_ext_i8()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            match (i16::try_from(x), ArbBitInt::from_i64(64, x).to_sign_ext_i16()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            match (i32::try_from(x), ArbBitInt::from_i64(64, x).to_sign_ext_i32()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            assert_eq!(ArbBitInt::from_i64(64, x).to_sign_ext_i64(), Some(x));

            // wrapping_add
            // i8
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_add(&ArbBitInt::from_i64(64, y)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_add(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_add(&ArbBitInt::from_i64(64, y)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_add(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_add(&ArbBitInt::from_i64(64, y)).to_sign_ext_i32(),
                i32::try_from(x.wrapping_add(y)).ok()
            );
            // i64
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_add(&ArbBitInt::from_i64(64, y)).to_sign_ext_i64(),
                Some(x.wrapping_add(y))
            );

            // wrapping_sub
            // i8
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_sub(&ArbBitInt::from_i64(64, y)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_sub(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_sub(&ArbBitInt::from_i64(64, y)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_sub(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_sub(&ArbBitInt::from_i64(64, y)).to_sign_ext_i32(),
                i32::try_from(x.wrapping_sub(y)).ok()
            );
            // i64
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_sub(&ArbBitInt::from_i64(64, y)).to_sign_ext_i64(),
                Some(x.wrapping_sub(y))
            );

            // wrapping_mul
            // i8
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_mul(&ArbBitInt::from_i64(64, y)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_mul(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_mul(&ArbBitInt::from_i64(64, y)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_mul(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_mul(&ArbBitInt::from_i64(64, y)).to_sign_ext_i32(),
                i32::try_from(x.wrapping_mul(y)).ok()
            );
            // i64
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .wrapping_mul(&ArbBitInt::from_i64(64, y)).to_sign_ext_i64(),
                Some(x.wrapping_mul(y))
            );

            // bitand
            // i8
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitand(&ArbBitInt::from_i64(64, y)).to_sign_ext_i8(),
                i8::try_from(x.bitand(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitand(&ArbBitInt::from_i64(64, y)).to_sign_ext_i16(),
                i16::try_from(x.bitand(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitand(&ArbBitInt::from_i64(64, y)).to_sign_ext_i32(),
                i32::try_from(x.bitand(y)).ok()
            );
            // i64
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitand(&ArbBitInt::from_i64(64, y)).to_sign_ext_i64(),
                Some(x.bitand(y))
            );

            // bitor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i8(),
                i8::try_from(x.bitor(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i16(),
                i16::try_from(x.bitor(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i32(),
                i32::try_from(x.bitor(y)).ok()
            );
            // i64
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i64(),
                Some(x.bitor(y))
            );

            // bitxor
            // i8
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitxor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i8(),
                i8::try_from(x.bitxor(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitxor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i16(),
                i16::try_from(x.bitxor(y)).ok()
            );
            // i32
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitxor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i32(),
                i32::try_from(x.bitxor(y)).ok()
            );
            // i64
            assert_eq!(
                ArbBitInt::from_i64(64, x)
                    .bitxor(&ArbBitInt::from_i64(64, y)).to_sign_ext_i64(),
                Some(x.bitxor(y))
            );
        }

        #[test]
        fn arbbitint_usize(x in any::<usize>(), y in any::<usize>()) {
            match (i8::try_from(x), ArbBitInt::from_usize(x).to_sign_ext_i8()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }
            match (i16::try_from(x), ArbBitInt::from_usize(x).to_sign_ext_i16()) {
                (Ok(a), Some(b)) if a == b => (),
                (Err(_), None) => (),
                a => panic!("{a:?}")
            }

            // wrapping_add
            // i8
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .wrapping_add(&ArbBitInt::from_usize(y)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_add(y)).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .wrapping_add(&ArbBitInt::from_usize( y)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_add(y) as i64).ok()
            );

            // wrapping_sub
            // i8
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .wrapping_sub(&ArbBitInt::from_usize(y)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_sub(y) as i64).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .wrapping_sub(&ArbBitInt::from_usize(y)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_sub(y) as i64).ok()
            );

            // wrapping_mul
            // i8
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .wrapping_mul(&ArbBitInt::from_usize(y)).to_sign_ext_i8(),
                i8::try_from(x.wrapping_mul(y) as i64).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .wrapping_mul(&ArbBitInt::from_usize(y)).to_sign_ext_i16(),
                i16::try_from(x.wrapping_mul(y) as i64).ok()
            );

            // bitand
            // i8
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .bitand(&ArbBitInt::from_usize(y)).to_sign_ext_i8(),
                i8::try_from(x.bitand(y) as i64).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .bitand(&ArbBitInt::from_usize(y)).to_sign_ext_i16(),
                i16::try_from(x.bitand(y) as i64).ok()
            );

            // bitor
            // i8
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .bitor(&ArbBitInt::from_usize(y)).to_sign_ext_i8(),
                i8::try_from(x.bitor(y) as i64).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .bitor(&ArbBitInt::from_usize(y)).to_sign_ext_i16(),
                i16::try_from(x.bitor(y) as i64).ok()
            );

            // bitxor
            // i8
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .bitxor(&ArbBitInt::from_usize(y)).to_sign_ext_i8(),
                i8::try_from(x.bitxor(y) as i64).ok()
            );
            // i16
            assert_eq!(
                ArbBitInt::from_usize(x)
                    .bitxor(&ArbBitInt::from_usize(y)).to_sign_ext_i16(),
                i16::try_from(x.bitxor(y) as i64).ok()
            );
        }

        #[test]
        fn arbbitint_8bit_shl(x in any::<u8>(), y in 0u32..=8) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(8, x as u64).checked_shl(y).map(|x| x.to_zero_ext_u8()),
                x.checked_shl(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_16bit_shl(x in any::<u16>(), y in 0u32..=16) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(16, x as u64).checked_shl(y).map(|x| x.to_zero_ext_u16()),
                x.checked_shl(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_32bit_shl(x in any::<u32>(), y in 0u32..=32) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(32, x as u64).checked_shl(y).map(|x| x.to_zero_ext_u32()),
                x.checked_shl(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_64bit_shl(x in any::<u64>(), y in 0u32..=63) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(64, x).checked_shl(y).map(|x| x.to_zero_ext_u64()),
                x.checked_shl(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_8bit_ashr(x in any::<i8>(), y in 0u32..=8) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_i64(8, x as i64).checked_ashr(y).map(|x| x.to_sign_ext_i8()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_16bit_ashr(x in any::<i16>(), y in 0u32..=16) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_i64(16, x as i64).checked_ashr(y).map(|x| x.to_sign_ext_i16()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_32bit_ashr(x in any::<i32>(), y in 0u32..=32) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_i64(32, x as i64).checked_ashr(y).map(|x| x.to_sign_ext_i32()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_64bit_ashr(x in any::<i64>(), y in 0u32..=63) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_i64(64, x).checked_ashr(y).map(|x| x.to_sign_ext_i64()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_8bit_lshr(x in any::<u8>(), y in 0u32..=8) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(8, x as u64).checked_lshr(y).map(|x| x.to_zero_ext_u8()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_16bit_lshr(x in any::<u16>(), y in 0u32..=16) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(16, x as u64).checked_lshr(y).map(|x| x.to_zero_ext_u16()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_32bit_lshr(x in any::<u32>(), y in 0u32..=32) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(32, x as u64).checked_lshr(y).map(|x| x.to_zero_ext_u32()),
                x.checked_shr(y).map(Some)
            );
        }

        #[test]
        fn arbbitint_64bit_lshr(x in any::<u64>(), y in 0u32..=63) {
            // Notice that we deliberately allow y to extend beyond to the shiftable range, to make
            // sure that we test the "failure" case, while not biasing our testing too-far to
            // "failure cases only".
            assert_eq!(
                ArbBitInt::from_u64(64, x).checked_lshr(y).map(|x| x.to_zero_ext_u64()),
                x.checked_shr(y).map(Some)
            );
        }
    }
}
