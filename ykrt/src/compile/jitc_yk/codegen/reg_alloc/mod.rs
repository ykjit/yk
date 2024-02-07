//! Register allocation.
//!
//! This module:
//!  - describes the generic interface to register allocators.
//!  - contains concrete implementations of register allocators.

use super::{super::jit_ir, abs_stack::AbstractStack};

mod spill_alloc;
#[cfg(test)]
pub(crate) use spill_alloc::SpillAllocator;

/// Describes a local variable allocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum LocalAlloc {
    /// The local variable is on the stack.
    Stack {
        /// The offset (from the base pointer) of the allocation.
        ///
        /// This is independent of which direction the stack grows. In other words, for
        /// architectures where the stack grows downwards, you'd subtract this from the base
        /// pointer to find the address of the allocation.
        ///
        /// OPT: consider addressing relative to the stack pointer, thus freeing up the base
        /// pointer for general purpose use.
        frame_off: usize,
    },
    /// The local variable is in a register.
    ///
    /// FIXME: unimplemented.
    Register,
}

impl LocalAlloc {
    /// Create a [Self::Stack] allocation.
    pub(crate) fn new_stack(frame_off: usize) -> Self {
        Self::Stack { frame_off }
    }
}

/// Indicates the direction of stack growth.
pub(crate) enum StackDirection {
    GrowsUp,
    GrowsDown,
}

/// The API to regsiter allocators.
///
/// Register allocators are responsible for assigning storage for local variables.
pub(crate) trait RegisterAllocator {
    /// Creates a register allocator for a stack growing in the specified direction.
    fn new(stack_dir: StackDirection) -> Self
    where
        Self: Sized;

    /// Allocates `size` bytes storage space for the local variable defined by the instruction with
    /// index `local`.
    fn allocate(
        &mut self,
        local: jit_ir::InstrIdx,
        size: usize,
        stack: &mut AbstractStack,
    ) -> LocalAlloc;

    /// Return the allocation for the value computed by the instruction at the specified
    /// instruction index.
    fn allocation<'a>(&'a self, idx: jit_ir::InstrIdx) -> &'a LocalAlloc;
}
