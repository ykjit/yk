use super::jit_ir::Module;
use crate::compile::CompilationError;

mod canonicalise;
mod lvn;

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
#[allow(dead_code)]
pub(super) fn opt(m: Module) -> Result<Module, CompilationError> {
    let m = canonicalise::canonicalise(m)?;
    lvn::lvn(m)
}
