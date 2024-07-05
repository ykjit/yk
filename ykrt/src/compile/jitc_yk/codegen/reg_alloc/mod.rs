//! Register allocation.
//!
//! This module:
//!  - describes the generic interface to register allocators.
//!  - contains concrete implementations of register allocators.

use super::{super::jit_ir, abs_stack::AbstractStack};

pub(crate) mod spill_alloc;
#[cfg(test)]
pub(crate) use spill_alloc::SpillAllocator;

/// Where is an SSA variable stored?
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VarLocation {
    /// The SSA variable is on the stack.
    Stack {
        /// The offset from the base of the trace's function frame.
        frame_off: usize,
        /// Size in bytes of the allocation.
        size: usize,
    },
    /// The SSA variable is in a register.
    Register,
    /// A constant integer `bits` wide (see [jit_ir::Const::ConstInt] for the constraints on the
    /// bit width) and with value `v`.
    ConstInt { bits: u32, v: u64 },
    /// A constant float.
    ConstFloat(f64),
}

impl VarLocation {
    /// Create a [Self::Stack] allocation.
    pub(crate) fn new_stack(frame_off: usize, size: usize) -> Self {
        Self::Stack { frame_off, size }
    }
}

/// Indicates the direction of stack growth.
pub(crate) enum StackDirection {
    GrowsUp,
    GrowsDown,
}

/// The API to register allocators.
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
        local: jit_ir::InstIdx,
        size: usize,
        stack: &mut AbstractStack,
    ) -> VarLocation;

    /// Returns the location for the SSA variable defined by `iidx`.
    ///
    /// # Panics
    ///
    /// Panics if there is no such SSA variable.
    fn location(&self, iidx: jit_ir::InstIdx) -> &VarLocation;
}
