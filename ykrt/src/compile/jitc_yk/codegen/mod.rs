//! The JIT's Code Generator.

// FIXME: eventually delete.
#![allow(dead_code)]

use super::{jit_ir, CompilationError};
use dynasmrt::{ExecutableBuffer, Executor};
use reg_alloc::RegisterAllocator;
use std::fmt;

mod abs_stack;
pub(crate) mod reg_alloc;

// Note that we make no attempt to cross-arch-test code generators.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

/// A trait that defines access to JIT compiled code.
pub(crate) trait CodeGenOutput: fmt::Debug + Send + Sync {
    /// Disassemble the code-genned trace into a string.
    #[cfg(any(debug_assertions, test))]
    fn disassemble(&self) -> Result<String, CompilationError>;
    fn ptr(&self) -> *const libc::c_void;
}

/// All code generators conform to this contract.
pub(crate) trait CodeGen<'a> {
    /// Instantiate a code generator for the specified JIT module.
    fn new(
        jit_mod: &'a jit_ir::Module,
        ra: &'a mut dyn RegisterAllocator,
    ) -> Result<Self, CompilationError>
    where
        Self: Sized;

    /// Perform code generation.
    fn codegen(self) -> Result<Box<dyn CodeGenOutput>, CompilationError>;
}

#[cfg(test)]
mod tests {
    use super::CodeGenOutput;
    use fm::FMatcher;

    /// Test helper to use `fm` to match a disassembled trace.
    pub(crate) fn match_asm(cgo: Box<dyn CodeGenOutput>, pattern: &str) {
        let dis = cgo.disassemble().unwrap();
        match FMatcher::new(pattern).unwrap().matches(&dis) {
            Ok(()) => (),
            Err(e) => panic!(
                "\n!!! Emitted code didn't match !!!\n\n{}\nFull asm:\n{}\n",
                e, dis
            ),
        }
    }
}
