//! Simple, local optimisations.
//!
//! These include strength reductions and other optimisations that can be performed with little
//! analysis.

use crate::compile::{
    jitc_yk::jit_ir::{
        BinOp, BinOpInst, IcmpInst, Inst, InstIdx, Module, Operand, PackedOperand, Predicate,
    },
    CompilationError,
};

pub(super) fn simple(mut m: Module) -> Result<Module, CompilationError> {
    let mut inst_iter = m.iter_inst_idxs();
    while let Some(inst_i) = inst_iter.next(&m) {
        let inst = m.inst(inst_i).clone();
        match inst {
            Inst::BinOp(BinOpInst {
                lhs,
                binop: BinOp::Mul,
                rhs,
            }) => opt_mul(&mut m, inst_i, lhs, rhs)?,
            Inst::Icmp(x) => opt_icmp(&mut m, inst_i, x)?,
            _ => (),
        }
    }
    Ok(m)
}

fn opt_mul(
    m: &mut Module,
    inst_i: InstIdx,
    lhs: PackedOperand,
    rhs: PackedOperand,
) -> Result<(), CompilationError> {
    match (lhs.unpack(m), rhs.unpack(m)) {
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
                } else if y & (y - 1) == 0 {
                    // Replace `x * y` with `x << ...`.
                    let shl = u64::from(y.ilog2());
                    let new_const = Operand::Const(m.insert_const(old_const.u64_to_int(shl))?);
                    let new_inst =
                        BinOpInst::new(Operand::Local(mul_inst), BinOp::Shl, new_const).into();
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
    Ok(())
}

fn opt_icmp(
    m: &mut Module,
    inst_i: InstIdx,
    IcmpInst { lhs, pred, rhs }: IcmpInst,
) -> Result<(), CompilationError> {
    if let (Operand::Const(x), Operand::Const(y)) = (lhs.unpack(m), rhs.unpack(m)) {
        if let (Some(x), Some(y)) = (m.const_(x).int_to_u64(), m.const_(y).int_to_u64()) {
            // Constant fold comparisons of simple integers. Note that we have to follow the
            // LLVM semantics carefully. The quotes in the `match` below are from
            // https://llvm.org/docs/LangRef.html#icmp-instruction.
            let r = match pred {
                // "eq: yields true if the operands are equal, false otherwise. No sign
                // interpretation is necessary or performed."
                Predicate::Equal => x == y,
                // "ne: yields true if the operands are unequal, false otherwise. No sign
                // interpretation is necessary or performed."
                Predicate::NotEqual => x != y,
                // "ugt: interprets the operands as unsigned values and yields true if op1 is
                // greater than op2."
                Predicate::UnsignedGreater => x > y,
                // "uge: interprets the operands as unsigned values and yields true if op1 is
                // greater than or equal to op2."
                Predicate::UnsignedGreaterEqual => x >= y,
                // "ult: interprets the operands as unsigned values and yields true if op1 is
                // less than op2."
                Predicate::UnsignedLess => x < y,
                // "ule: interprets the operands as unsigned values and yields true if op1 is
                // less than or equal to op2."
                Predicate::UnsignedLessEqual => x <= y,
                // "interprets the operands as signed values and yields true if op1 is greater
                // than op2."
                Predicate::SignedGreater => (x as i64) > (y as i64),
                // "sge: interprets the operands as signed values and yields true if op1 is
                // greater than or equal to op2."
                Predicate::SignedGreaterEqual => (x as i64) >= (y as i64),
                // "slt: interprets the operands as signed values and yields true if op1 is less
                // than op2."
                Predicate::SignedLess => (x as i64) < (y as i64),
                // "sle: interprets the operands as signed values and yields true if op1 is less
                // than or equal to op2."
                Predicate::SignedLessEqual => (x as i64) <= (y as i64),
            };

            if r {
                m.replace(inst_i, Inst::ProxyConst(m.true_constidx()));
            } else {
                m.replace(inst_i, Inst::ProxyConst(m.false_constidx()));
            }
        }
    }

    Ok(())
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
            %5: i64 = mul %0, 12i64
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
            %5: i64 = mul %0, 12i64
        ",
        );
    }

    #[test]
    fn opt_icmp_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = eq 0i8, 0i8
            %1: i8 = eq 0i8, 1i8
            %2: i8 = ne 0i8, 0i8
            %3: i8 = ne 0i8, 1i8
            %4: i8 = ugt 0i8, 0i8
            %5: i8 = ugt 0i8, 1i8
            %6: i8 = ugt 1i8, 0i8
            %7: i8 = uge 0i8, 0i8
            %8: i8 = uge 0i8, 1i8
            %9: i8 = uge 1i8, 0i8
            %10: i8 = ult 0i8, 0i8
            %11: i8 = ult 0i8, 1i8
            %12: i8 = ult 1i8, 0i8
            %13: i8 = ule 0i8, 0i8
            %14: i8 = ule 0i8, 1i8
            %15: i8 = ule 1i8, 0i8
            %16: i8 = sgt 0i8, 0i8
            %17: i8 = sgt 0i8, -1i8
            %18: i8 = sgt -1i8, 0i8
            %19: i8 = sge 0i8, 0i8
            %20: i8 = sge 0i8, -1i8
            %21: i8 = sge -1i8, 0i8
            %22: i8 = slt 0i8, 0i8
            %23: i8 = slt 0i8, -1i8
            %24: i8 = slt -1i8, 0i8
            %25: i8 = sle 0i8, 0i8
            %26: i8 = sle 0i8, -1i8
            %27: i8 = sle -1i8, 0i8
            black_box %0
            black_box %1
            black_box %2
            black_box %3
            black_box %4
            black_box %5
            black_box %6
            black_box %7
            black_box %8
            black_box %9
            black_box %10
            black_box %11
            black_box %12
            black_box %13
            black_box %14
            black_box %15
            black_box %16
            black_box %17
            black_box %18
            black_box %19
            black_box %20
            black_box %21
            black_box %22
            black_box %23
            black_box %24
            black_box %25
            black_box %26
            black_box %27
        ",
            |m| simple(m).unwrap(),
            "
          ...
          entry:
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 0i1
            black_box 0i1
            black_box 1i1
            black_box 1i1
            black_box 0i1
            black_box 1i1
        ",
        );
    }
}
