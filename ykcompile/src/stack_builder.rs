use crate::Location;
use dynasmrt::{x64::Rq::RBP, Register};
use std::convert::{TryFrom, TryInto};

#[derive(Default, Debug)]
pub struct StackBuilder {
    /// Keeps track of how many bytes have been allocated.
    stack_top: u64,
}

/// A naive stack builder for allocating objects on the stack.
///
/// Naive because it could allocate less space by caching up the requested allocations and
/// better-packing them.
///
/// The top of the stack, before any calls to `alloc()` is assumed to be appropriately aligned.
impl StackBuilder {
    /// Allocate an object of given size and alignment on the stack, returning a `Location::Mem`
    /// describing the position of the allocation. The stack is assumed to grow down.
    pub fn alloc(&mut self, size: u64, align: u64) -> Location {
        self.align(align);
        self.stack_top += size;
        Location::new_mem(RBP.code(), -1 * i32::try_from(self.stack_top).unwrap())
    }

    /// Aligns `offset` to `align` bytes.
    fn align(&mut self, align: u64) {
        let mask = align - 1;
        self.stack_top = (self.stack_top + mask) & !mask
    }

    /// Total allocated stack size in bytes.
    pub fn size(&self) -> u32 {
        self.stack_top.try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::StackBuilder;

    #[test]
    fn test_stackbuilder() {
        let mut sb = StackBuilder::default();

        assert_eq!(sb.alloc(8, 8).unwrap_mem().offs, -8);
        assert_eq!(sb.alloc(1, 1).unwrap_mem().offs, -9);
        assert_eq!(sb.alloc(8, 8).unwrap_mem().offs, -24);
        assert_eq!(sb.alloc(1, 1).unwrap_mem().offs, -25);
        assert_eq!(sb.alloc(4, 2).unwrap_mem().offs, -30);
    }
}
