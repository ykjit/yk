//! The JIT's Code Generator.

// FIXME: eventually delete.
#![allow(dead_code)]

use super::{jit_ir, CompilationError};
use crate::compile::CompiledTrace;
use reg_alloc::RegisterAllocator;
use std::sync::Arc;

mod abs_stack;
pub(crate) mod reg_alloc;

// Note that we make no attempt to cross-arch-test code generators.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

/// All code generators conform to this contract.
pub(crate) trait CodeGen<'a> {
    /// Instantiate a code generator for the specified JIT module.
    fn new(
        jit_mod: &'a jit_ir::Module,
        ra: Box<dyn RegisterAllocator>,
    ) -> Result<Box<Self>, CompilationError>
    where
        Self: Sized;

    /// Perform code generation.
    fn codegen(self: Box<Self>) -> Result<Arc<dyn CompiledTrace>, CompilationError>;
}
