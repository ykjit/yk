//! JIT IR canonicalisation.
//!
//! After canonicalisation, code can rely on the following properties:
//!
//!   * [BinOpInst]s are never of the form `<binop> <const>, <local var>`.

use crate::compile::{
    jitc_yk::jit_ir::{BinOpInst, Inst, Module, Operand},
    CompilationError,
};

/// Perform simple canonicalisation of the JIT IR.
pub(super) fn canonicalise(mut m: Module) -> Result<Module, CompilationError> {
    for inst in m.iter_mut_insts() {
        if let Inst::BinOp(x) = inst {
            // Canonicalise `(<const>, <local var>)` to `(<local var>, <const>)`.
            let (lhs, rhs) = (x.lhs(), x.rhs());
            if let (&Operand::Const(_), &Operand::Local(_)) = (&lhs, &rhs) {
                *inst = BinOpInst::new(rhs, x.binop(), lhs).into();
            }
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canon_bin_op() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i32 = load_ti 0
            %1: i32 = add 2i32, %0
            black_box %1
        ",
            |m| canonicalise(m).unwrap(),
            "
          ...
          entry:
            %0: i32 = load_ti 0
            %1: i32 = add %0, 2i32
            black_box %1
        ",
        );
    }
}
