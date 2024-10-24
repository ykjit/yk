// A trace IR optimiser.
//
// The optimiser works in a single forward pass (well, it also does a single backwards pass at the
// end too, but that's only because we can't yet do backwards code generation). As it progresses
// through a trace, it both mutates the trace IR directly and also refines its idea about what
// value an instruction might produce. These two actions are subtly different: mutation is done
// in this module; the refinement of values in the [Analyse] module.

use super::{
    int_signs::{SignExtend, Truncate},
    jit_ir::{
        BinOp, BinOpInst, Const, ConstIdx, ICmpInst, Inst, InstIdx, Module, Operand, Predicate, Ty,
    },
};
use crate::compile::CompilationError;

mod analyse;

use analyse::Analyse;

struct Opt {
    m: Module,
    an: Analyse,
}

impl Opt {
    fn new(m: Module) -> Self {
        let an = Analyse::new(&m);
        Self { m, an }
    }

    fn opt(mut self) -> Result<Module, CompilationError> {
        for iidx in self.m.iter_all_inst_idxs() {
            match self.m.inst_raw(iidx) {
                #[cfg(test)]
                Inst::BlackBox(_) => (),
                Inst::Const(cidx) => self.an.set_value(iidx, Value::Const(cidx)),
                Inst::BinOp(x) => match x.binop() {
                    BinOp::Add => (),
                    BinOp::Mul => match (
                        self.an.op_map(&self.m, x.lhs(&self.m)),
                        self.an.op_map(&self.m, x.rhs(&self.m)),
                    ) {
                        (Operand::Const(cidx), Operand::Var(copy_iidx))
                        | (Operand::Var(copy_iidx), Operand::Const(cidx)) => {
                            match self.m.const_(cidx) {
                                Const::Int(_, 0) => {
                                    // Replace `x * 0` with `0`.
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                Const::Int(_, 1) => {
                                    // Replace `x * 1` with `x`.
                                    self.m.replace(iidx, Inst::Copy(copy_iidx));
                                }
                                Const::Int(ty_idx, x) if x.is_power_of_two() => {
                                    // Replace `x * y` with `x << ...`.
                                    let shl = u64::from(x.ilog2());
                                    let shl_op = Operand::Const(
                                        self.m.insert_const(Const::Int(*ty_idx, shl))?,
                                    );
                                    let new_inst =
                                        BinOpInst::new(Operand::Var(copy_iidx), BinOp::Shl, shl_op)
                                            .into();
                                    self.m.replace(iidx, new_inst);
                                }
                                _ => (),
                            }
                        }
                        (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                            let lhs_c = self.m.const_(lhs_cidx);
                            let rhs_c = self.m.const_(rhs_cidx);
                            match (lhs_c, rhs_c) {
                                (Const::Int(lhs_ty, lhs_v), Const::Int(rhs_ty, rhs_v)) => {
                                    debug_assert_eq!(lhs_ty, rhs_ty);
                                    let Ty::Integer(bits) = self.m.type_(*lhs_ty) else {
                                        panic!()
                                    };
                                    let mul: u64 = lhs_v.truncate(*bits) * rhs_v.truncate(*bits);
                                    let trun = mul.truncate(*bits);
                                    let cidx = self.m.insert_const(lhs_c.u64_to_int(trun))?;
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                _ => todo!(),
                            }
                        }
                        (Operand::Var(_), Operand::Var(_)) => (),
                    },
                    _ => (),
                },
                Inst::ICmp(x) => {
                    self.icmp(iidx, x);
                }
                Inst::Guard(x) => {
                    if let Operand::Const(_) = self.an.op_map(&self.m, x.cond(&self.m)) {
                        // A guard that references a constant is, by definition, not needed and
                        // doesn't affect future analyses.
                        self.m.replace(iidx, Inst::Tombstone);
                    } else {
                        self.an.guard(&self.m, x);
                    }
                }
                Inst::SExt(x) => {
                    if let Operand::Const(cidx) = self.an.op_map(&self.m, x.val(&self.m)) {
                        let Const::Int(src_ty, src_val) = self.m.const_(cidx) else {
                            unreachable!()
                        };
                        let src_ty = self.m.type_(*src_ty);
                        let dst_ty = self.m.type_(x.dest_tyidx());
                        let (Ty::Integer(src_bits), Ty::Integer(dst_bits)) = (src_ty, dst_ty)
                        else {
                            unreachable!()
                        };
                        let dst_val = match (src_bits, dst_bits) {
                            (32, 64) => Const::Int(x.dest_tyidx(), src_val.sign_extend(32, 64)),
                            _ => todo!("{src_bits} {dst_bits}"),
                        };
                        let dst_cidx = self.m.insert_const(dst_val)?;
                        self.m.replace(iidx, Inst::Const(dst_cidx));
                    }
                }
                _ => (),
            }
            self.cse(iidx);
        }
        // FIXME: When code generation supports backwards register allocation, we won't need to
        // explicitly perform dead code elimination and this function can be made `#[cfg(test)]` only.
        self.m.dead_code_elimination();
        Ok(self.m)
    }

    /// Attempt common subexpression elimination on `iidx`, replacing it with a `Copy` if possible.
    fn cse(&mut self, iidx: InstIdx) {
        // If this instruction is already a `Copy`, then there is nothing for CSE to do.
        let Some(inst) = self.m.inst_nocopy(iidx) else {
            return;
        };
        // There's no point in trying CSE on a `Tombstone`.
        if let Inst::Tombstone = inst {
            return;
        }
        // FIXME: This is O(n), but most instructions can't possibly be CSE candidates.
        for back_iidx in (0..usize::from(iidx)).rev() {
            let back_iidx = InstIdx::unchecked_from(back_iidx);
            // Only examine non-`Copy` instructions, to avoid us continually checking the same
            // (subset of) instructions over and over again.
            let Some(back) = self.m.inst_nocopy(back_iidx) else {
                continue;
            };
            if !inst.has_store_effect(&self.m)
                && !inst.has_load_effect(&self.m)
                && !inst.is_barrier(&self.m)
                && inst.decopy_eq(&self.m, back)
            {
                self.m.replace(iidx, Inst::Copy(back_iidx));
                return;
            }
        }
    }

    /// Optimise an [ICmpInst].
    fn icmp(&mut self, iidx: InstIdx, inst: ICmpInst) {
        let lhs = self.an.op_map(&self.m, inst.lhs(&self.m));
        let pred = inst.predicate();
        let rhs = self.an.op_map(&self.m, inst.rhs(&self.m));
        match (&lhs, &rhs) {
            (&Operand::Const(lhs_cidx), &Operand::Const(rhs_cidx)) => {
                self.icmp_both_const(iidx, lhs_cidx, pred, rhs_cidx)
            }
            (&Operand::Var(_), &Operand::Const(_)) | (&Operand::Const(_), &Operand::Var(_)) => (),
            (&Operand::Var(_), &Operand::Var(_)) => (),
        }
    }

    /// Optimise an `ICmp` if both sides are constants. It is required that [Opt::op_map] has been
    /// called on both `lhs` and `rhs` to obtain the `ConstIdx`s.
    fn icmp_both_const(&mut self, iidx: InstIdx, lhs: ConstIdx, pred: Predicate, rhs: ConstIdx) {
        let lhs_c = self.m.const_(lhs);
        let rhs_c = self.m.const_(rhs);
        match (lhs_c, rhs_c) {
            (Const::Float(..), Const::Float(..)) => (),
            (Const::Int(lhs_tyidx, x), Const::Int(rhs_tyidx, y)) => {
                debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                // Constant fold comparisons of simple integers.
                let x = *x;
                let y = *y;
                let r = match pred {
                    Predicate::Equal => x == y,
                    Predicate::NotEqual => x != y,
                    Predicate::UnsignedGreater => x > y,
                    Predicate::UnsignedGreaterEqual => x >= y,
                    Predicate::UnsignedLess => x < y,
                    Predicate::UnsignedLessEqual => x <= y,
                    Predicate::SignedGreater => (x as i64) > (y as i64),
                    Predicate::SignedGreaterEqual => (x as i64) >= (y as i64),
                    Predicate::SignedLess => (x as i64) < (y as i64),
                    Predicate::SignedLessEqual => (x as i64) <= (y as i64),
                };

                self.m.replace(
                    iidx,
                    Inst::Const(if r {
                        self.m.true_constidx()
                    } else {
                        self.m.false_constidx()
                    }),
                );
            }
            (Const::Ptr(x), Const::Ptr(y)) => {
                // Constant fold comparisons of pointers.
                let x = *x;
                let y = *y;
                let r = match pred {
                    Predicate::Equal => x == y,
                    Predicate::NotEqual => x != y,
                    Predicate::UnsignedGreater => x > y,
                    Predicate::UnsignedGreaterEqual => x >= y,
                    Predicate::UnsignedLess => x < y,
                    Predicate::UnsignedLessEqual => x <= y,
                    Predicate::SignedGreater => (x as i64) > (y as i64),
                    Predicate::SignedGreaterEqual => (x as i64) >= (y as i64),
                    Predicate::SignedLess => (x as i64) < (y as i64),
                    Predicate::SignedLessEqual => (x as i64) <= (y as i64),
                };

                self.m.replace(
                    iidx,
                    Inst::Const(if r {
                        self.m.true_constidx()
                    } else {
                        self.m.false_constidx()
                    }),
                );
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
enum Value {
    Unknown,
    Const(ConstIdx),
}

/// Create JIT IR from the (`aot_mod`, `ta_iter`) tuple.
pub(super) fn opt(m: Module) -> Result<Module, CompilationError> {
    Opt::new(m).opt()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn opt_const_guard() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = load_ti 0
            guard false, 0i1, [%0]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
        ",
        );
    }

    #[test]
    fn opt_const_guard_indirect() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = eq 0i8, 0i8
            guard true, %0, []
            %1: i1 = eq 0i8, 1i8
            guard false, %1, [%0]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
        ",
        );
    }

    #[test]
    fn opt_const_guard_chain() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = load_ti 0
            %1: i8 = mul %0, 0i8
            %2: i1 = eq %1, 0i8
            guard true, %2, [%0, %1]
            black_box %0
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = load_ti ...
            black_box %0
        ",
        );
    }

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
            black_box %3
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %1: i8 = load_ti ...
            %3: i8 = add %1, 0i8
            black_box %3
            black_box %3
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
            %5: i8 = add %1, %4
            black_box %3
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = load_ti ...
            %1: i8 = load_ti ...
            %3: i8 = add %1, %0
            black_box %3
            black_box %3
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
            |m| opt(m).unwrap(),
            "
          ...
          entry:
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
            black_box %1
            black_box %2
            black_box %3
            black_box %4
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i64 = load_ti ...
            %1: i64 = shl %0, 1i64
            %2: i64 = shl %0, 2i64
            %3: i64 = shl %0, 62i64
            %4: i64 = mul %0, 9223372036854775807i64
            %5: i64 = mul %0, 12i64
            black_box ...
            ...
        ",
        );
    }

    #[test]
    fn opt_icmp_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i1 = eq 0i8, 0i8
            %1: i1 = eq 0i8, 1i8
            %2: i1 = ne 0i8, 0i8
            %3: i1 = ne 0i8, 1i8
            %4: i1 = ugt 0i8, 0i8
            %5: i1 = ugt 0i8, 1i8
            %6: i1 = ugt 1i8, 0i8
            %7: i1 = uge 0i8, 0i8
            %8: i1 = uge 0i8, 1i8
            %9: i1 = uge 1i8, 0i8
            %10: i1 = ult 0i8, 0i8
            %11: i1 = ult 0i8, 1i8
            %12: i1 = ult 1i8, 0i8
            %13: i1 = ule 0i8, 0i8
            %14: i1 = ule 0i8, 1i8
            %15: i1 = ule 1i8, 0i8
            %16: i1 = sgt 0i8, 0i8
            %17: i1 = sgt 0i8, -1i8
            %18: i1 = sgt -1i8, 0i8
            %19: i1 = sge 0i8, 0i8
            %20: i1 = sge 0i8, -1i8
            %21: i1 = sge -1i8, 0i8
            %22: i1 = slt 0i8, 0i8
            %23: i1 = slt 0i8, -1i8
            %24: i1 = slt -1i8, 0i8
            %25: i1 = sle 0i8, 0i8
            %26: i1 = sle 0i8, -1i8
            %27: i1 = sle -1i8, 0i8
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
            |m| opt(m).unwrap(),
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
