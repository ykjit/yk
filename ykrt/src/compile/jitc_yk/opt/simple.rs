//! Simple, local optimisations.
//!
//! These include strength reductions and other optimisations that can be performed with little
//! analysis.

use crate::compile::{
    jitc_yk::{
        aot_ir::BinOp,
        jit_ir::{BinOpInst, Inst, Module, Operand},
    },
    CompilationError,
};

pub(super) fn simple(mut m: Module) -> Result<Module, CompilationError> {
    for inst_i in m.iter_inst_idxs() {
        match m.inst(inst_i).clone() {
            Inst::BinOp(x) if x.binop() == BinOp::Mul => {
                if let (Operand::Local(_), Operand::Const(c_idx)) = (x.lhs(), x.rhs()) {
                    let old_const = m.const_(c_idx);
                    if let Some(y) = old_const.int_to_i64() {
                        if y % 2 == 0 {
                            let shl = i64::from(y.ilog2());
                            let new_const =
                                Operand::Const(m.insert_const(old_const.i64_to_int(shl))?);
                            let new_inst = BinOpInst::new(x.lhs(), BinOp::Shl, new_const).into();
                            m.replace(inst_i, new_inst);
                        }
                    }
                }
            }
            _ => (),
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_mul_shl() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i64 = load_ti 0
            %1: i64 = mul %0, 2i64
            %2: i64 = mul %0, 4i64
            %3: i64 = mul %0, 4611686018427387904i64
            %4: i64 = mul %0, 9223372036854775807i64
            black_box %1
            black_box %2
            black_box %3
            black_box %4
        ",
            |m| simple(m).unwrap(),
            "
          ...
          entry:
            %0: i64 = load_ti 0
            %1: i64 = shl %0, 1i64
            %2: i64 = shl %0, 2i64
            %3: i64 = shl %0, 62i64
            %4: i64 = mul %0, 9223372036854775807i64
            black_box %1
            black_box %2
            black_box %3
            black_box %4
        ",
        );
    }
}
