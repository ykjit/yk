#[derive(Default, Debug)]
pub struct StackBuilder {
    stack_top: u64,
}

/// A naive stack builder for allocating objects on the stack.
///
/// Naive because it could allocate less space by caching up the requested allocations and
/// better-packing them.
///
/// The top of the stack, before any calls to `alloc()` is assumed to be appropriately aligned.
impl StackBuilder {
    /// Allocate an object of given size and alignment on the stack, returning an offset which
    /// is intended to be subtracted from the base pointer.
    pub fn alloc(&mut self, size: u64, align: u64) -> u64 {
        self.align(align);
        self.stack_top += size;
        self.stack_top
    }

    /// Aligns `offset` to `align` bytes.
    fn align(&mut self, align: u64) {
        let mask = align - 1;
        self.stack_top = (self.stack_top + mask) & !mask
    }
}

#[cfg(test)]
mod tests {
    use super::StackBuilder;

    #[test]
    fn test_stackbuilder() {
        let mut sb = StackBuilder::default();

        assert_eq!(sb.alloc(8, 8), 8);
        assert_eq!(sb.alloc(1, 1), 9);
        assert_eq!(sb.alloc(8, 8), 24);
        assert_eq!(sb.alloc(1, 1), 25);
        assert_eq!(sb.alloc(4, 2), 30);
    }
}
