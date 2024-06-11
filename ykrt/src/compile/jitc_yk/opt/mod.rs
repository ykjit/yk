use super::jit_ir::Module;
use crate::compile::CompilationError;

mod simple;

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
#[allow(dead_code)]
pub(super) fn opt(m: Module) -> Result<Module, CompilationError> {
    simple::simple(m)
}
