//! The x64 specific parts of register allocation.

use crate::compile::j2::regalloc::RegT;
#[cfg(test)]
use crate::compile::j2::{hir::Ty, regalloc::TestRegIter};
use iced_x86::Register;
use strum::{EnumCount, FromRepr};

#[derive(Clone, Copy, Debug, EnumCount, FromRepr, PartialEq)]
// If the `repr` changes from `u8`, the `as` in the `Reg::regidx()` function will also need
// updating.
#[repr(u8)]
pub(in crate::compile::j2) enum Reg {
    // The values we assign in this `enum` are irrelevant semantically, though if they're
    // not consecutive, the register allocator will necessarily waste space.
    RAX = 0,
    RCX,
    RDX,
    RBX,
    RSI,
    RDI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,

    // Floating point registers. All of these are available to us.
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,

    Undefined,
}

impl Reg {
    pub(super) fn from_dwarf_reg(dwarf_reg: u16) -> Self {
        match dwarf_reg {
            0 => Reg::RAX,
            1 => Reg::RDX,
            2 => Reg::RCX,
            3 => Reg::RBX,
            4 => Reg::RSI,
            5 => Reg::RDI,
            6 => unreachable!(), // RBP
            7 => unreachable!(), // RSP
            8 => Reg::R8,
            9 => Reg::R9,
            10 => Reg::R10,
            11 => Reg::R11,
            12 => Reg::R12,
            13 => Reg::R13,
            14 => Reg::R14,
            15 => Reg::R15,
            _ => unreachable!(),
        }
    }

    /// Is this a general purpose register?
    #[allow(clippy::match_like_matches_macro)]
    pub(super) fn is_gp(&self) -> bool {
        match self {
            Reg::RAX
            | Reg::RCX
            | Reg::RDX
            | Reg::RBX
            | Reg::RSI
            | Reg::RDI
            | Reg::R8
            | Reg::R9
            | Reg::R10
            | Reg::R11
            | Reg::R12
            | Reg::R13
            | Reg::R14
            | Reg::R15 => true,
            _ => false,
        }
    }

    pub(super) fn to_reg8(self) -> Register {
        match self {
            Reg::RAX => Register::AL,
            Reg::RCX => Register::CL,
            Reg::RDX => Register::DL,
            Reg::RBX => Register::BL,
            Reg::RSI => Register::SIL,
            Reg::RDI => Register::DIL,
            Reg::R8 => Register::R8L,
            Reg::R9 => Register::R9L,
            Reg::R10 => Register::R10L,
            Reg::R11 => Register::R11L,
            Reg::R12 => Register::R12L,
            Reg::R13 => Register::R13L,
            Reg::R14 => Register::R14L,
            Reg::R15 => Register::R15L,
            _ => unreachable!(),
        }
    }

    pub(super) fn to_reg16(self) -> Register {
        match self {
            Reg::RAX => Register::AX,
            Reg::RCX => Register::CX,
            Reg::RDX => Register::DX,
            Reg::RBX => Register::BX,
            Reg::RSI => Register::SI,
            Reg::RDI => Register::DI,
            Reg::R8 => Register::R8W,
            Reg::R9 => Register::R9W,
            Reg::R10 => Register::R10W,
            Reg::R11 => Register::R11W,
            Reg::R12 => Register::R12W,
            Reg::R13 => Register::R13W,
            Reg::R14 => Register::R14W,
            Reg::R15 => Register::R15W,
            _ => unreachable!(),
        }
    }

    pub(super) fn to_reg32(self) -> Register {
        match self {
            Reg::RAX => Register::EAX,
            Reg::RCX => Register::ECX,
            Reg::RDX => Register::EDX,
            Reg::RBX => Register::EBX,
            Reg::RSI => Register::ESI,
            Reg::RDI => Register::EDI,
            Reg::R8 => Register::R8D,
            Reg::R9 => Register::R9D,
            Reg::R10 => Register::R10D,
            Reg::R11 => Register::R11D,
            Reg::R12 => Register::R12D,
            Reg::R13 => Register::R13D,
            Reg::R14 => Register::R14D,
            Reg::R15 => Register::R15D,
            _ => unreachable!(),
        }
    }

    pub(super) fn to_reg64(self) -> Register {
        match self {
            Reg::RAX => Register::RAX,
            Reg::RCX => Register::RCX,
            Reg::RDX => Register::RDX,
            Reg::RBX => Register::RBX,
            Reg::RSI => Register::RSI,
            Reg::RDI => Register::RDI,
            Reg::R8 => Register::R8,
            Reg::R9 => Register::R9,
            Reg::R10 => Register::R10,
            Reg::R11 => Register::R11,
            Reg::R12 => Register::R12,
            Reg::R13 => Register::R13,
            Reg::R14 => Register::R14,
            Reg::R15 => Register::R15,
            x => unreachable!("{x:?}"),
        }
    }
}

impl RegT for Reg {
    type RegIdx = RegIdx;
    const MAX_REGIDX: RegIdx = RegIdx::from_usize_unchecked(Reg::COUNT);

    fn undefined() -> Reg {
        Reg::Undefined
    }

    fn from_regidx(idx: Self::RegIdx) -> Self {
        Reg::from_repr(idx.raw()).unwrap()
    }

    fn regidx(&self) -> Self::RegIdx {
        RegIdx::from(*self as u8)
    }

    #[cfg(test)]
    fn iter_test_regs() -> impl TestRegIter<Self> {
        X64TestRegIter::new()
    }

    #[cfg(test)]
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "RAX" => Some(Reg::RAX),
            "RCX" => Some(Reg::RCX),
            "RDX" => Some(Reg::RDX),
            "RBX" => Some(Reg::RBX),
            "RSI" => Some(Reg::RSI),
            "RDI" => Some(Reg::RDI),
            "R8" => Some(Reg::R8),
            "R9" => Some(Reg::R9),
            "R10" => Some(Reg::R10),
            "R11" => Some(Reg::R11),
            "R12" => Some(Reg::R12),
            "R13" => Some(Reg::R13),
            "R14" => Some(Reg::R14),
            "R15" => Some(Reg::R15),
            _ => None,
        }
    }
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Reg::RAX => "rax",
            Reg::RCX => "rcx",
            Reg::RDX => "rdx",
            Reg::RBX => "rbx",
            Reg::RSI => "rsi",
            Reg::RDI => "rdi",
            Reg::R8 => "r8",
            Reg::R9 => "r9",
            Reg::R10 => "r10",
            Reg::R11 => "r11",
            Reg::R12 => "r12",
            Reg::R13 => "r13",
            Reg::R14 => "r14",
            Reg::R15 => "r15",
            Reg::XMM0 => "xmm0",
            Reg::XMM1 => "xmm1",
            Reg::XMM2 => "xmm2",
            Reg::XMM3 => "xmm3",
            Reg::XMM4 => "xmm4",
            Reg::XMM5 => "xmm5",
            Reg::XMM6 => "xmm6",
            Reg::XMM7 => "xmm7",
            Reg::XMM8 => "xmm8",
            Reg::XMM9 => "xmm9",
            Reg::XMM10 => "xmm10",
            Reg::XMM11 => "xmm11",
            Reg::XMM12 => "xmm12",
            Reg::XMM13 => "xmm13",
            Reg::XMM14 => "xmm14",
            Reg::XMM15 => "xmm15",
            Reg::Undefined => todo!(),
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
struct X64TestRegIter<Reg> {
    gp_regs: Box<dyn Iterator<Item = Reg>>,
}

#[cfg(test)]
impl X64TestRegIter<Reg> {
    fn new() -> Self {
        Self {
            gp_regs: Box::new(
                [
                    Reg::R15,
                    Reg::R14,
                    Reg::R13,
                    Reg::R12,
                    Reg::R11,
                    Reg::R10,
                    Reg::R9,
                    Reg::R8,
                    Reg::RDI,
                    Reg::RBX,
                    Reg::RDX,
                    Reg::RCX,
                    Reg::RAX,
                ]
                .iter()
                .cloned(),
            ),
        }
    }
}

#[cfg(test)]
impl TestRegIter<Reg> for X64TestRegIter<Reg> {
    fn next_reg(&mut self, ty: &Ty) -> Option<Reg> {
        match ty {
            Ty::Func(_func_ty) => todo!(),
            Ty::Int(bitw) => {
                if *bitw <= 64 {
                    self.gp_regs.next()
                } else {
                    todo!()
                }
            }
            Ty::Ptr(addrspace) => {
                assert_eq!(*addrspace, 0);
                self.gp_regs.next()
            }
            Ty::Void => todo!(),
        }
    }
}

index_vec::define_index_type! {
    pub(in crate::compile::j2) struct RegIdx = u8;
    IMPL_RAW_CONVERSIONS = true;
}

// We prefer allocating registers such as R15, as they are not clobbered by CALLs / MULs (etc), and
// are less likely to need to be copied around.
pub(super) const NORMAL_GP_REGS: [Reg; 14] = [
    Reg::R15,
    Reg::R14,
    Reg::R13,
    Reg::R12,
    Reg::R11,
    Reg::R10,
    Reg::R9,
    Reg::R8,
    Reg::RDI,
    Reg::RSI,
    Reg::RBX,
    Reg::RDX,
    Reg::RCX,
    Reg::RAX,
];

pub(super) const ALL_XMM_REGS: [Reg; 16] = [
    Reg::XMM0,
    Reg::XMM1,
    Reg::XMM2,
    Reg::XMM3,
    Reg::XMM4,
    Reg::XMM5,
    Reg::XMM6,
    Reg::XMM7,
    Reg::XMM8,
    Reg::XMM9,
    Reg::XMM10,
    Reg::XMM11,
    Reg::XMM12,
    Reg::XMM13,
    Reg::XMM14,
    Reg::XMM15,
];
