//! A stack builder for allocating objects on the stack.
//!
//! The builder is naive in that it allocates objects immediately (when `alloc` is called), thus
//! potentially inserting more alignment padding than would be necessary if we deferred and
//! reordered allocations.
//!
//! The stack is assumed to be appropriately aligned before the builder is used.

use crate::Location;
use dynasmrt::{x64::Rq::RBP, Register};
use std::convert::{TryFrom, TryInto};

#[derive(Default, Debug)]
pub(crate) struct StackBuilder {
    /// Keeps track of how many bytes have been allocated.
    stack_top: u64,
}

impl StackBuilder {
    /// Allocate an object of given size and alignment on the stack, returning a `Location::Mem`
    /// describing the position of the allocation. The stack is assumed to grow down.
    pub(crate) fn alloc(&mut self, size: u64, align: u64) -> Location {
        self.align(align);
        self.stack_top += size;
        Location::new_mem(RBP.code(), -i32::try_from(self.stack_top).unwrap())
    }

    /// Aligns `offset` to `align` bytes.
    fn align(&mut self, align: u64) {
        let mask = align - 1;
        self.stack_top = (self.stack_top + mask) & !mask
    }

    /// Total allocated stack size in bytes.
    pub(crate) fn size(&self) -> u32 {
        self.stack_top.try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::StackBuilder;

    #[test]
    fn stackbuilder() {
        let mut sb = StackBuilder::default();

        assert_eq!(sb.alloc(8, 8).unwrap_mem().off, -8);
        assert_eq!(sb.alloc(1, 1).unwrap_mem().off, -9);
        assert_eq!(sb.alloc(8, 8).unwrap_mem().off, -24);
        assert_eq!(sb.alloc(1, 1).unwrap_mem().off, -25);
        assert_eq!(sb.alloc(4, 2).unwrap_mem().off, -30);
    }
}
