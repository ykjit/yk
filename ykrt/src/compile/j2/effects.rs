//! Specify and query instruction effects.
//!
//! This module provides a simple interface to effects. Effects form a simple lattice:
//!
//! ```text
//!                 All
//!                  |
//!      ____________|_________
//!     /     |      |         \
//! Guard   Heap  Internal  Volatile
//!     \_____|______|_________/
//!                  |
//!                None
//! ```
//!
//! This allows us to answer the question: do two [Effects] instances interfere with each other?
//! `All` interferes with everything; `None` interferes with nothing; `Guard` and `Heap` do not
//! interfere with each other; etc.
//!
//! Instructions in [super::hir::Inst] can inform the user of their read and write effects: note
//! that a given instruction will sometimes have different read and write effects!
//!
//! [Effects] instances are built up from a builder API. One can start with any individual element
//! (e.g. `all`/`guard`/`none`) or `all`/`none` and then add / minus effects. If in doubt, it is
//! better to start with `all` and minus effects.

// The simple bit mask that we use to represent effects.
pub const EFFECT_INTERNAL: u8 = 0b0001;
pub const EFFECT_GUARD: u8 = 0b0010;
pub const EFFECT_VOLATILE: u8 = 0b0100;
pub const EFFECT_HEAP: u8 = 0b1000;

/// A specification of an instruction's effects. This is an immutable struct: new [Effects] are
/// created with the various builder-style methods herein.
pub(super) struct Effects(u8);

#[allow(unused)]
impl Effects {
    /// An effects instance that interferes with everything.
    pub const fn all() -> Self {
        Self(!0)
    }

    /// An effects instance that interferes with nothing.
    pub const fn none() -> Self {
        Self(0)
    }

    /// Create a new `Effects` with `self` plus `Guard`.
    pub const fn add_guard(self) -> Self {
        Self(self.0 | EFFECT_GUARD)
    }

    /// Create a new `Effects` with `self` minus `Guard`.
    pub const fn minus_guard(self) -> Self {
        Self(self.0 & !EFFECT_GUARD)
    }

    /// Create a new `Effects` with `self` plus `Heap`.
    pub const fn add_heap(self) -> Self {
        Self(self.0 | EFFECT_HEAP)
    }

    /// Create a new `Effects` with `self` minus `Heap`.
    pub const fn minus_heap(self) -> Self {
        Self(self.0 & !EFFECT_HEAP)
    }

    /// Create a new `Effects` with `self` plus `Internal`.
    pub const fn add_internal(self) -> Self {
        Self(self.0 | EFFECT_INTERNAL)
    }

    /// Create a new `Effects` with `self` minus `Internal`.
    pub const fn minus_internal(self) -> Self {
        Self(self.0 & !EFFECT_INTERNAL)
    }

    /// Create a new `Effects` with `self` plus `Volatile`.
    pub const fn add_volatile(self) -> Self {
        Self(self.0 | EFFECT_VOLATILE)
    }

    /// Create a new `Effects` with `self` minus `Volatile`.
    pub const fn minus_volatile(self) -> Self {
        Self(self.0 & !EFFECT_VOLATILE)
    }

    /// Create a new `Effects` with the complement of `self` with `other`.
    pub const fn complement(self, other: Effects) -> Self {
        Self(self.0 & !other.0)
    }

    /// Create a new `Effects` with the union of `self` and `other`.
    pub const fn union(self, other: Effects) -> Self {
        Self(self.0 | other.0)
    }

    /// Do `self` and `other` interfere with each other? Since an [Effects] instance represents a
    /// set, this is equivalent to "is the intersection of `self` and `other` non-empty?"
    pub const fn interferes(&self, other: Effects) -> bool {
        (self.0 & other.0) != 0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn interferes() {
        // `all` interferes with everything except `none`
        assert!(Effects::all().interferes(Effects::all()));
        assert!(Effects::all().interferes(Effects::none().add_heap()));
        assert!(Effects::all().interferes(Effects::none().add_guard()));
        assert!(Effects::all().interferes(Effects::none().add_internal()));
        assert!(Effects::all().interferes(Effects::none().add_volatile()));
        assert!(!Effects::all().interferes(Effects::none()));

        // `none` interferes with nothing
        assert!(!Effects::none().interferes(Effects::none()));
        assert!(!Effects::none().interferes(Effects::all()));
        assert!(!Effects::none().interferes(Effects::none().add_heap()));
        assert!(!Effects::none().interferes(Effects::none().add_guard()));
        assert!(!Effects::none().interferes(Effects::none().add_internal()));
        assert!(!Effects::none().interferes(Effects::none().add_volatile()));
    }
}
