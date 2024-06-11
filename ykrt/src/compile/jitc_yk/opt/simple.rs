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
    let mut inst_iter = m.iter_inst_idxs();
    while let Some(inst_i) = inst_iter.next(&m) {
        if let Inst::BinOp(BinOpInst {
            lhs,
            binop: BinOp::Mul,
            rhs,
        }) = m.inst(inst_i).clone()
        {
            match (lhs.unpack(&m), rhs.unpack(&m)) {
                (Operand::Local(mul_inst), Operand::Const(mul_const))
                | (Operand::Const(mul_const), Operand::Local(mul_inst)) => {
                    let old_const = m.const_(mul_const);
                    if let Some(y) = old_const.int_to_u64() {
                        if y == 0 {
                            // Replace `x * 0` with `0`.
                            let const_idx = m.insert_const(old_const.u64_to_int(0))?;
                            m.replace(inst_i, Inst::ProxyConst(const_idx));
                        } else if y == 1 {
                            // Replace `x * 1` with `x`.
                            m.replace(inst_i, Inst::ProxyInst(mul_inst));
                        } else if y % 2 == 0 {
                            // Replace `x * y` with `x << ...`.
                            let shl = u64::from(y.ilog2());
                            let new_const =
                                Operand::Const(m.insert_const(old_const.u64_to_int(shl))?);
                            let new_inst =
                                BinOpInst::new(Operand::Local(mul_inst), BinOp::Shl, new_const)
                                    .into();
                            m.replace(inst_i, new_inst);
                        }
                    }
                }
                (Operand::Const(x), Operand::Const(y)) => {
                    // Constant fold the unsigned multiplication of two constants.
                    let x = m.const_(x);
                    let y = m.const_(y);
                    // If `x_val * y_val` overflows, we're fine with the UB, as the interpreter
                    // author is at fault.
                    let new_val = x.int_to_u64().unwrap() * y.int_to_u64().unwrap();
                    let new_const = m.insert_const(x.u64_to_int(new_val))?;
                    m.replace(inst_i, Inst::ProxyConst(new_const));
                }
                (Operand::Local(_), Operand::Local(_)) => (),
            }
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_mul_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %2: i8 = mul %0, 0i8
            %3: i8 = add %1, %2
            %4: i8 = mul 0i8, %0
            %5: i8 = add %1, %2
        ",
            |m| simple(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %3: i8 = add %1, 0i8
            %5: i8 = add %1, 0i8
        ",
        );
    }

    #[test]
    fn opt_mul_one() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %2: i8 = mul %0, 1i8
            %3: i8 = add %1, %2
            %4: i8 = mul 1i8, %0
            %5: i8 = add %1, %2
        ",
            |m| simple(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = load_ti 0
            %1: i8 = load_ti 1
            %3: i8 = add %1, %0
            %5: i8 = add %1, %0
        ",
        );
    }

    #[test]
    fn opt_mul_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = mul %0, 0i8
            %2: i8 = mul %0, 0i8
            %3: i8 = mul %1, %2
            black_box %3
        ",
            |m| simple(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = load_ti 0
            black_box 0i8
        ",
        );
    }

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
