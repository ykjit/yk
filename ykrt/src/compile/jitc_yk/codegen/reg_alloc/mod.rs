//! Register allocation.
//!
//! This module:
//!  - describes the generic interface to register allocators.
//!  - contains concrete implementations of register allocators.

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

/// Indicates the direction of stack growth.
pub(crate) enum StackDirection {
    GrowsUp,
    GrowsDown,
}
