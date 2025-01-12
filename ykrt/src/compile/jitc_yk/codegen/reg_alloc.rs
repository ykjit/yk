//! Register allocation.
//!
//! This module:
//!  - describes the generic interface to register allocators.
//!  - contains concrete implementations of register allocators.

/// Where is an SSA variable stored?
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VarLocation<R> {
    /// The SSA variable is on the stack of the of the executed trace or the main interpreter loop.
    /// Since we execute the trace on the main interpreter frame we can't distinguish the two.
    ///
    /// Note: two SSA variables can alias to the same `Stack` location.
    Stack {
        /// The offset from the base of the trace's function frame.
        frame_off: u32,
        /// Size in bytes of the allocation.
        size: usize,
    },
    /// The SSA variable is a stack pointer with the value `RBP-frame_off`.
    ///
    /// Note: two SSA variables can alias to the same `Direct` location.
    Direct {
        /// The offset from the base of the trace's function frame.
        frame_off: i32,
        /// Size in bytes of the allocation.
        size: usize,
    },
    /// The SSA variable is in a register.
    ///
    /// Note: two SSA variables can alias to the same `Register` location.
    Register(R),
    /// A constant integer `bits` wide (see [jit_ir::Const::ConstInt] for the constraints on the
    /// bit width) and with value `v`.
    ConstInt { bits: u32, v: u64 },
    /// A constant float.
    ConstFloat(f64),
    /// A constant pointer.
    ConstPtr(usize),
}
