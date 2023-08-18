use crate::trace::IRTrace;
use libc::c_void;
use std::{error::Error, sync::Arc};
use tempfile::NamedTempFile;

#[cfg(jitc_llvm)]
mod jitc_llvm;

/// The trait that every JIT compiler backend must implement.
pub(crate) trait Compiler: Send + Sync {
    /// Compile an [IRTrace] into machine code.
    fn compile(
        &self,
        irtrace: IRTrace,
    ) -> Result<(*const c_void, Option<NamedTempFile>), Box<dyn Error>>;
}

pub(crate) fn default_compiler() -> Result<Arc<dyn Compiler>, Box<dyn Error>> {
    #[cfg(jitc_llvm)]
    {
        return Ok(jitc_llvm::JITCLLVM::new()?);
    }

    #[allow(unreachable_code)]
    Err("No JIT compiler supported on this platform/configuration.".into())
}
