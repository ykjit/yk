//! The JIT's Code Generator.

// FIXME: eventually delete.
#![allow(dead_code)]

use super::CompilationError;
use crate::{compile::jitc_yk::jit_ir::Module, compile::CompiledTrace, location::HotLocation, MT};
use parking_lot::Mutex;
use std::{error::Error, sync::Arc};

mod abs_stack;
pub(crate) mod reg_alloc;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

/// A code generator.
///
/// This must be capable of generating code for multiple modules, possibly in parallel.
pub(crate) trait CodeGen: Send + Sync {
    /// Generate code for the module `m`.
    fn codegen(
        &self,
        m: Module,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError>;
}

pub(crate) fn default_codegen() -> Result<Arc<dyn CodeGen>, Box<dyn Error>> {
    #[cfg(target_arch = "x86_64")]
    return Ok(x86_64::X86_64CodeGen::new()?);

    #[cfg(not(target_arch = "x86_64"))]
    return Err("No code generator available for this platform".into());
}
