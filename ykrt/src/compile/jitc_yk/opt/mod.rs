use super::jit_ir::Module;
use crate::compile::CompilationError;

mod simple;

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn opt(m: Module) -> Result<Module, CompilationError> {
    let mut m = simple::simple(m)?;
    // FIXME: When code generation supports backwards register allocation, we won't need to
    // explicitly perform dead code elimination and this function can be made `#[cfg(test)]` only.
    m.dead_code_elimination();
    Ok(m)
}
