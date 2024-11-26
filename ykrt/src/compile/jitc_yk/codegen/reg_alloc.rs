//! Register allocation.
//!
//! This module:
//!  - describes the generic interface to register allocators.
//!  - contains concrete implementations of register allocators.

use dynasmrt::x64::{Rq, Rx};

/// Where is an SSA variable stored?
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VarLocation {
    /// The SSA variable is on the stack of the of the executed trace or the main interpreter loop.
    /// Since we execute the trace on the main interpreter frame we can't distinguish the two.
    Stack {
        /// The offset from the base of the trace's function frame.
        frame_off: u32,
        /// Size in bytes of the allocation.
        size: usize,
    },
    /// The SSA variable is a stack pointer with the value `RBP-frame_off`.
    Direct {
        /// The offset from the base of the trace's function frame.
        frame_off: i32,
        /// Size in bytes of the allocation.
        size: usize,
    },
    /// The SSA variable is in a register.
    Register(Register),
    /// A constant integer `bits` wide (see [jit_ir::Const::ConstInt] for the constraints on the
    /// bit width) and with value `v`.
    ConstInt { bits: u32, v: u64 },
    /// A constant float.
    ConstFloat(f64),
    /// A constant pointer.
    ConstPtr(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Register {
    GP(Rq), // general purpose
    FP(Rx), // floating point
}

/// Indicates the direction of stack growth.
pub(crate) enum StackDirection {
    GrowsUp,
    GrowsDown,
}

impl From<&VarLocation> for yksmp::Location {
    fn from(val: &VarLocation) -> Self {
        match val {
            VarLocation::Stack { frame_off, size } => {
                // A stack location translates is an offset in relation to RBP which has the DWARF
                // number 6.
                yksmp::Location::Indirect(
                    6,
                    -i32::try_from(*frame_off).unwrap(),
                    u16::try_from(*size).unwrap(),
                )
            }
            VarLocation::Direct { frame_off, size } => {
                yksmp::Location::Direct(6, *frame_off, u16::try_from(*size).unwrap())
            }
            VarLocation::Register(reg) => {
                let dwarf = match reg {
                    Register::GP(reg) => match reg {
                        Rq::RAX => 0,
                        Rq::RDX => 1,
                        Rq::RCX => 2,
                        Rq::RBX => 3,
                        Rq::RSI => 4,
                        Rq::RDI => 5,
                        Rq::R8 => 8,
                        Rq::R9 => 9,
                        Rq::R10 => 10,
                        Rq::R11 => 11,
                        Rq::R12 => 12,
                        Rq::R13 => 13,
                        Rq::R14 => 14,
                        Rq::R15 => 15,
                        e => todo!("{:?}", e),
                    },
                    Register::FP(reg) => match reg {
                        Rx::XMM0 => 17,
                        Rx::XMM1 => 18,
                        Rx::XMM2 => 19,
                        Rx::XMM3 => 20,
                        Rx::XMM4 => 21,
                        Rx::XMM5 => 22,
                        Rx::XMM6 => 23,
                        Rx::XMM7 => 24,
                        Rx::XMM8 => 25,
                        Rx::XMM9 => 26,
                        Rx::XMM10 => 27,
                        Rx::XMM11 => 28,
                        Rx::XMM12 => 29,
                        Rx::XMM13 => 30,
                        Rx::XMM14 => 31,
                        Rx::XMM15 => 32,
                    },
                };
                // We currently only use 8 byte registers, so the size is constant. Since these are
                // JIT values there are no extra locations we need to worry about.
                yksmp::Location::Register(dwarf, 8, 0, Vec::new())
            }
            VarLocation::ConstInt { bits, v } => {
                if *bits <= 32 {
                    yksmp::Location::Constant(u32::try_from(*v).unwrap())
                } else {
                    todo!(">32 bit constant")
                }
            }
            e => todo!("{:?}", e),
        }
    }
}
