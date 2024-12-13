//! Register allocation.
//!
//! This module:
//!  - describes the generic interface to register allocators.
//!  - contains concrete implementations of register allocators.

use crate::compile::jitc_yk::jit_ir::{InstIdx, Module};
use dynasmrt::x64::{Rq, Rx};

/// Where is an SSA variable stored?
///
/// FIXME: Too much of this is hard-coded to the x64 backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VarLocation {
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

#[cfg(target_arch = "x86_64")]
impl VarLocation {
    pub(crate) fn from_yksmp_location(m: &Module, iidx: InstIdx, x: &yksmp::Location) -> Self {
        match x {
            yksmp::Location::Register(0, ..) => VarLocation::Register(Register::GP(Rq::RAX)),
            yksmp::Location::Register(1, ..) => {
                // Since the control point passes the stackmap ID via RDX this case only happens in
                // side-traces.
                VarLocation::Register(Register::GP(Rq::RDX))
            }
            yksmp::Location::Register(2, ..) => VarLocation::Register(Register::GP(Rq::RCX)),
            yksmp::Location::Register(3, ..) => VarLocation::Register(Register::GP(Rq::RBX)),
            yksmp::Location::Register(4, ..) => VarLocation::Register(Register::GP(Rq::RSI)),
            yksmp::Location::Register(5, ..) => VarLocation::Register(Register::GP(Rq::RDI)),
            yksmp::Location::Register(8, ..) => VarLocation::Register(Register::GP(Rq::R8)),
            yksmp::Location::Register(9, ..) => VarLocation::Register(Register::GP(Rq::R9)),
            yksmp::Location::Register(10, ..) => VarLocation::Register(Register::GP(Rq::R10)),
            yksmp::Location::Register(11, ..) => VarLocation::Register(Register::GP(Rq::R11)),
            yksmp::Location::Register(12, ..) => VarLocation::Register(Register::GP(Rq::R12)),
            yksmp::Location::Register(13, ..) => VarLocation::Register(Register::GP(Rq::R13)),
            yksmp::Location::Register(14, ..) => VarLocation::Register(Register::GP(Rq::R14)),
            yksmp::Location::Register(15, ..) => VarLocation::Register(Register::GP(Rq::R15)),
            yksmp::Location::Register(x, ..) if *x >= 17 && *x <= 32 => VarLocation::Register(
                Register::FP(super::x64::lsregalloc::FP_REGS[usize::from(x - 17)]),
            ),
            yksmp::Location::Direct(6, off, size) => VarLocation::Direct {
                frame_off: *off,
                size: usize::from(*size),
            },
            // Since the trace shares the same stack frame as the main interpreter loop, we can
            // translate indirect locations into normal stack locations. Note that while stackmaps
            // use negative offsets, we use positive offsets for stack locations.
            yksmp::Location::Indirect(6, off, size) => VarLocation::Stack {
                frame_off: u32::try_from(*off * -1).unwrap(),
                size: usize::from(*size),
            },
            yksmp::Location::Constant(v) => {
                // FIXME: This isn't fine-grained enough, as there may be constants of any
                // bit-size.
                let byte_size = m.inst(iidx).def_byte_size(m);
                debug_assert!(byte_size <= 8);
                VarLocation::ConstInt {
                    bits: u32::try_from(byte_size).unwrap() * 8,
                    v: u64::from(*v),
                }
            }
            e => {
                todo!("{:?}", e);
            }
        }
    }
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
                yksmp::Location::Register(dwarf, 8, Vec::new())
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
