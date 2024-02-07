//! The abstract stack.

/// This data structure keeps track of an abstract stack pointer for a JIT frame during code
/// generation. The abstract stack pointer is zero-based, so the stack pointer value also serves as
/// the size of the stack.
///
/// The implementation is platform agnostic: as the abstract stack gets bigger, the abstract stack
/// pointer grows upwards, even on architectures where the stack grows downwards.
#[derive(Debug, Default)]
pub(crate) struct AbstractStack(usize);

impl AbstractStack {
    /// Aligns the abstract stack pointer to the specified number of bytes.
    ///
    /// Returns the newly aligned stack pointer.
    pub(crate) fn align(&mut self, to: usize) -> usize {
        let rem = self.0 % to;
        if rem != 0 {
            self.0 += to - rem;
        }
        self.0
    }

    /// Makes the stack bigger by `nbytes` bytes.
    ///
    /// Returns the new stack pointer.
    pub(crate) fn grow(&mut self, nbytes: usize) -> usize {
        self.0 += nbytes;
        self.0
    }

    /// Returns the stack pointer value.
    pub(crate) fn size(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::AbstractStack;

    #[test]
    fn grow() {
        let mut s = AbstractStack::default();
        assert_eq!(s.grow(8), 8);
        assert_eq!(s.grow(8), 16);
        assert_eq!(s.grow(1), 17);
        assert_eq!(s.grow(0), 17);
        assert_eq!(s.grow(1000), 1017);
    }

    #[test]
    fn align() {
        let mut s = AbstractStack::default();
        for i in 1..100 {
            assert_eq!(s.align(i), 0);
            assert_eq!(s.align(i), 0);
        }
        for i in 1..100 {
            s.grow(1);
            assert_eq!(s.align(1), i);
            assert_eq!(s.align(1), i);
        }
        assert_eq!(s.align(8), 104);
        for i in 105..205 {
            assert_eq!(s.align(i), i);
            assert_eq!(s.align(i), i);
        }
        assert_eq!(s.align(12345678), 12345678);
        assert_eq!(s.align(12345678), 12345678);
    }

    #[test]
    fn size() {
        let mut s = AbstractStack::default();
        for i in 1..100 {
            s.grow(1);
            assert_eq!(s.size(), i);
        }
    }
}
