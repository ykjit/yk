//! The X64 backend.

mod asm;
mod deopt;
pub(super) mod x64hir_to_asm;
mod x64regalloc;

pub(super) use x64regalloc::Reg;
