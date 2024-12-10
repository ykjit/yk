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
        BinOp, BinOpInst, Const, ConstIdx, ICmpInst, Inst, InstIdx, Module, Operand, Predicate,
        PtrAddInst, Ty,
    },
};
use crate::compile::CompilationError;

mod analyse;
mod instll;

use analyse::Analyse;
use instll::InstLinkedList;

struct Opt {
    m: Module,
    an: Analyse,
    /// Needed for common subexpression elimination.
    instll: InstLinkedList,
}

impl Opt {
    fn new(m: Module) -> Self {
        let an = Analyse::new(&m);
        let instll = InstLinkedList::new(&m);
        Self { m, an, instll }
    }

    fn opt(mut self) -> Result<Module, CompilationError> {
        let skipping = self.m.iter_skipping_insts().collect::<Vec<_>>();
        for (iidx, inst) in skipping.into_iter() {
            match inst {
                #[cfg(test)]
                Inst::BlackBox(_) => (),
                Inst::Const(_) | Inst::Copy(_) | Inst::Tombstone => unreachable!(),
                Inst::BinOp(x) => match x.binop() {
                    BinOp::Add => match (
                        self.an.op_map(&self.m, x.lhs(&self.m)),
                        self.an.op_map(&self.m, x.rhs(&self.m)),
                    ) {
                        (Operand::Const(op_cidx), Operand::Var(op_iidx))
                        | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                            match self.m.const_(op_cidx) {
                                Const::Int(_, 0) => {
                                    // Replace `x + 0` with `x`.
                                    self.m.replace(iidx, Inst::Copy(op_iidx));
                                }
                                _ => {
                                    // Canonicalise to (Var, Const).
                                    self.m.replace(
                                        iidx,
                                        BinOpInst::new(
                                            Operand::Var(op_iidx),
                                            BinOp::Add,
                                            Operand::Const(op_cidx),
                                        )
                                        .into(),
                                    );
                                }
                            }
                        }
                        (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                            match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                                (Const::Int(lhs_tyidx, lhs_v), Const::Int(rhs_tyidx, rhs_v)) => {
                                    debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                                    let Ty::Integer(bits) = self.m.type_(*lhs_tyidx) else {
                                        panic!()
                                    };
                                    let cidx = self.m.insert_const_int(
                                        *lhs_tyidx,
                                        (lhs_v.wrapping_add(*rhs_v)).truncate(*bits),
                                    )?;
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                _ => todo!(),
                            }
                        }
                        (Operand::Var(_), Operand::Var(_)) => (),
                    },
                    BinOp::And => match (
                        self.an.op_map(&self.m, x.lhs(&self.m)),
                        self.an.op_map(&self.m, x.rhs(&self.m)),
                    ) {
                        (Operand::Const(op_cidx), Operand::Var(op_iidx))
                        | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                            match self.m.const_(op_cidx) {
                                Const::Int(_, 0) => {
                                    // Replace `x & 0` with `0`.
                                    self.m.replace(iidx, Inst::Const(op_cidx));
                                }
                                _ => {
                                    // Canonicalise to (Var, Const).
                                    self.m.replace(
                                        iidx,
                                        BinOpInst::new(
                                            Operand::Var(op_iidx),
                                            BinOp::And,
                                            Operand::Const(op_cidx),
                                        )
                                        .into(),
                                    );
                                }
                            }
                        }
                        (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                            match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                                (Const::Int(lhs_tyidx, lhs_v), Const::Int(rhs_tyidx, rhs_v)) => {
                                    debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                                    let Ty::Integer(bits) = self.m.type_(*lhs_tyidx) else {
                                        panic!()
                                    };
                                    let cidx = self.m.insert_const_int(
                                        *lhs_tyidx,
                                        (lhs_v & rhs_v).truncate(*bits),
                                    )?;
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                _ => todo!(),
                            }
                        }
                        (Operand::Var(_), Operand::Var(_)) => (),
                    },
                    BinOp::LShr => match (
                        self.an.op_map(&self.m, x.lhs(&self.m)),
                        self.an.op_map(&self.m, x.rhs(&self.m)),
                    ) {
                        (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                            match self.m.const_(op_cidx) {
                                Const::Int(_, 0) => {
                                    // Replace `x >> 0` with `x`.
                                    self.m.replace(iidx, Inst::Copy(op_iidx));
                                }
                                _ => {
                                    // Canonicalise to (Var, Const).
                                    self.m.replace(
                                        iidx,
                                        BinOpInst::new(
                                            Operand::Var(op_iidx),
                                            BinOp::LShr,
                                            Operand::Const(op_cidx),
                                        )
                                        .into(),
                                    );
                                }
                            }
                        }
                        (Operand::Const(_), Operand::Var(_)) => (),
                        (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                            match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                                (Const::Int(lhs_tyidx, lhs_v), Const::Int(rhs_tyidx, rhs_v)) => {
                                    debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                                    let Ty::Integer(bits) = self.m.type_(*lhs_tyidx) else {
                                        panic!()
                                    };
                                    let cidx = self.m.insert_const_int(
                                        *lhs_tyidx,
                                        (lhs_v >> rhs_v).truncate(*bits),
                                    )?;
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                _ => todo!(),
                            }
                        }
                        (Operand::Var(_), Operand::Var(_)) => (),
                    },
                    BinOp::Mul => match (
                        self.an.op_map(&self.m, x.lhs(&self.m)),
                        self.an.op_map(&self.m, x.rhs(&self.m)),
                    ) {
                        (Operand::Const(op_cidx), Operand::Var(op_iidx))
                        | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                            match self.m.const_(op_cidx) {
                                Const::Int(_, 0) => {
                                    // Replace `x * 0` with `0`.
                                    self.m.replace(iidx, Inst::Const(op_cidx));
                                }
                                Const::Int(_, 1) => {
                                    // Replace `x * 1` with `x`.
                                    self.m.replace(iidx, Inst::Copy(op_iidx));
                                }
                                Const::Int(ty_idx, x) if x.is_power_of_two() => {
                                    // Replace `x * y` with `x << ...`.
                                    let shl = u64::from(x.ilog2());
                                    let shl_op = Operand::Const(
                                        self.m.insert_const(Const::Int(*ty_idx, shl))?,
                                    );
                                    let new_inst =
                                        BinOpInst::new(Operand::Var(op_iidx), BinOp::Shl, shl_op)
                                            .into();
                                    self.m.replace(iidx, new_inst);
                                }
                                _ => {
                                    // Canonicalise to (Var, Const).
                                    self.m.replace(
                                        iidx,
                                        BinOpInst::new(
                                            Operand::Var(op_iidx),
                                            BinOp::Mul,
                                            Operand::Const(op_cidx),
                                        )
                                        .into(),
                                    );
                                }
                            }
                        }
                        (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                            match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                                (Const::Int(lhs_tyidx, lhs_v), Const::Int(rhs_tyidx, rhs_v)) => {
                                    debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                                    let Ty::Integer(bits) = self.m.type_(*lhs_tyidx) else {
                                        panic!()
                                    };
                                    let cidx = self.m.insert_const_int(
                                        *lhs_tyidx,
                                        (lhs_v.wrapping_mul(*rhs_v)).truncate(*bits),
                                    )?;
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                _ => todo!(),
                            }
                        }
                        (Operand::Var(_), Operand::Var(_)) => (),
                    },
                    BinOp::Or => match (
                        self.an.op_map(&self.m, x.lhs(&self.m)),
                        self.an.op_map(&self.m, x.rhs(&self.m)),
                    ) {
                        (Operand::Const(op_cidx), Operand::Var(op_iidx))
                        | (Operand::Var(op_iidx), Operand::Const(op_cidx)) => {
                            match self.m.const_(op_cidx) {
                                Const::Int(_, 0) => {
                                    // Replace `x | 0` with `x`.
                                    self.m.replace(iidx, Inst::Copy(op_iidx));
                                }
                                _ => {
                                    // Canonicalise to (Var, Const).
                                    self.m.replace(
                                        iidx,
                                        BinOpInst::new(
                                            Operand::Var(op_iidx),
                                            BinOp::Or,
                                            Operand::Const(op_cidx),
                                        )
                                        .into(),
                                    );
                                }
                            }
                        }
                        (Operand::Const(lhs_cidx), Operand::Const(rhs_cidx)) => {
                            match (self.m.const_(lhs_cidx), self.m.const_(rhs_cidx)) {
                                (Const::Int(lhs_tyidx, lhs_v), Const::Int(rhs_tyidx, rhs_v)) => {
                                    debug_assert_eq!(lhs_tyidx, rhs_tyidx);
                                    let Ty::Integer(bits) = self.m.type_(*lhs_tyidx) else {
                                        panic!()
                                    };
                                    let cidx = self.m.insert_const_int(
                                        *lhs_tyidx,
                                        (lhs_v | rhs_v).truncate(*bits),
                                    )?;
                                    self.m.replace(iidx, Inst::Const(cidx));
                                }
                                _ => todo!(),
                            }
                        }
                        (Operand::Var(_), Operand::Var(_)) => (),
                    },
                    _ => (),
                },
                Inst::DynPtrAdd(x) => {
                    if let Operand::Const(cidx) = self.an.op_map(&self.m, x.num_elems(&self.m)) {
                        let Const::Int(_, v) = self.m.const_(cidx) else {
                            panic!()
                        };
                        // DynPtrAdd indices are signed, so we have to be careful to interpret the
                        // constant as such.
                        let v = *v as i64;
                        // LLVM IR allows `off` to be an `i64` but our IR currently allows only an
                        // `i32`. On that basis, we can hit our limits before the program has
                        // itself hit UB, at which point we can't go any further.
                        let off = i32::try_from(v)
                            .map_err(|_| ())
                            .and_then(|v| v.checked_mul(i32::from(x.elem_size())).ok_or(()))
                            .map_err(|_| {
                                CompilationError::LimitExceeded(
                                    "`DynPtrAdd` offset exceeded `i32` bounds".into(),
                                )
                            })?;
                        self.m
                            .replace(iidx, Inst::PtrAdd(PtrAddInst::new(x.ptr(&self.m), off)));
                    }
                }
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
                Inst::PtrAdd(x) => match self.an.op_map(&self.m, x.ptr(&self.m)) {
                    Operand::Const(_) => todo!(),
                    Operand::Var(op_iidx) => {
                        if x.off() == 0 {
                            self.m.replace(iidx, Inst::Copy(op_iidx));
                        }
                    }
                },
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
                Inst::Param(x) => {
                    // FIXME: This feels like it should be handled by trace_builder, but we can't
                    // do so yet because of https://github.com/ykjit/yk/issues/1435.
                    if let yksmp::Location::Constant(v) = self.m.param(x.paramidx()) {
                        let cidx = self.m.insert_const(Const::Int(x.tyidx(), u64::from(*v)))?;
                        self.an.set_value(iidx, Value::Const(cidx));
                    }
                }
                Inst::ZExt(x) => {
                    if let Operand::Const(cidx) = self.an.op_map(&self.m, x.val(&self.m)) {
                        let Const::Int(_src_ty, src_val) = self.m.const_(cidx) else {
                            unreachable!()
                        };
                        #[cfg(debug_assertions)]
                        {
                            let src_ty = self.m.type_(*_src_ty);
                            let dst_ty = self.m.type_(x.dest_tyidx());
                            let (Ty::Integer(src_bits), Ty::Integer(dst_bits)) = (src_ty, dst_ty)
                            else {
                                unreachable!()
                            };
                            debug_assert!(src_bits <= dst_bits);
                            debug_assert!(*dst_bits <= 64);
                        }
                        let dst_cidx = self.m.insert_const(Const::Int(x.dest_tyidx(), *src_val))?;
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

    /// Attempt common subexpression elimination on `iidx`, replacing it with a `Copy` or
    /// `Tombstone` if possible.
    fn cse(&mut self, iidx: InstIdx) {
        let inst = match self.m.inst_nocopy(iidx) {
            // If this instruction is already a `Copy`, then there is nothing for CSE to do.
            None => return,
            // There's no point in trying CSE on a `Const` or `Tombstone`.
            Some(Inst::Const(_)) | Some(Inst::Tombstone) => return,
            Some(inst @ Inst::Guard(ginst)) => {
                for (_, back_inst) in self.instll.rev_iter(&self.m, inst) {
                    if let Inst::Guard(back_ginst) = back_inst {
                        if ginst.cond(&self.m) == back_ginst.cond(&self.m)
                            && ginst.expect() == back_ginst.expect()
                        {
                            self.m.replace(iidx, Inst::Tombstone);
                            return;
                        }
                    }
                }
                inst
            }
            Some(inst) => {
                // We don't perform CSE on instructions that have / enforce effects.
                if inst.has_store_effect(&self.m)
                    || inst.has_load_effect(&self.m)
                    || inst.is_barrier(&self.m)
                {
                    return;
                }

                // Can we CSE the instruction at `iidx`?
                for (back_iidx, back_inst) in self.instll.rev_iter(&self.m, inst) {
                    if inst.decopy_eq(&self.m, back_inst) {
                        self.m.replace(iidx, Inst::Copy(back_iidx));
                        return;
                    }
                }
                inst
            }
        };

        // We only need to `push` instructions that:
        //
        //   1. Can possibly be CSE candidates. Earlier in the function we've ruled out a number of
        //      things that aren't CSE candidates, which saves us some pointless work.
        //   2. Haven't been turned into `Copy`s. So only if we've failed to CSE a given
        //      instruction is it worth pushing to the `instll`.
        self.instll.push(iidx, inst);
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
            %0: i1 = param 0
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
            %2: i1 = eq 0i8, 1i8
            guard false, %2, [%0]
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
            %0: i8 = param 0
            %1: i8 = mul %0, 0i8
            %2: i1 = eq %1, 0i8
            guard true, %2, [%0, %1]
            black_box %0
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_add_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = add %0, 0i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_add_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 0i8
            %1: i8 = add %0, 1i8
            %2: i64 = 18446744073709551614i64
            %3: i64 = add %2, 4i64
            black_box %1
            black_box %3
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i8
            black_box 2i64
        ",
        );
    }

    #[test]
    fn opt_and_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = and %0, 0i8
            black_box %1
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
    fn opt_and_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 3i8
            %2: i8 = and %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 2i8
        ",
        );
    }

    #[test]
    fn opt_dyn_ptr_add_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = dyn_ptr_add %0, 2i8, 3
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 6
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_lshr_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = lshr %0, 0i8
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_lshr_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 1i8
            %2: i8 = lshr %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i8
        ",
        );
    }

    #[test]
    fn opt_or_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = or %0, 0i8
            %2: i8 = or 0i8, %0
            black_box %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            black_box %0
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_or_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = 2i8
            %1: i8 = 1i8
            %2: i8 = or %0, %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 3i8
        ",
        );
    }

    #[test]
    fn opt_mul_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
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
            %1: i8 = param ...
            black_box %1
            black_box %1
        ",
        );
    }

    #[test]
    fn opt_mul_one() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
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
            %0: i8 = param ...
            %1: i8 = param ...
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
            %0: i8 = 0i8
            %1: i8 = mul %0, 1i8
            %2: i8 = 1i8
            %3: i8 = mul %2, 1i8
            %4: i64 = 9223372036854775809i64
            %5: i64 = mul %4, 2i64
            black_box %1
            black_box %3
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 0i8
            black_box 1i8
            black_box 2i64
        ",
        );
    }

    #[test]
    fn opt_mul_shl() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i64 = param 0
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
            %0: i64 = param ...
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

    #[test]
    fn opt_zext_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i16 = zext 1i8
            %1: i32 = zext 4294967295i32
            %2: i64 = zext 4294967295i32
            black_box %0
            black_box %1
            black_box %2
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            black_box 1i16
            black_box 4294967295i32
            black_box 4294967295i64
        ",
        );
    }

    #[test]
    fn opt_ptradd_zero() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = ptr_add %0, 0
            black_box %1
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            black_box %0
        ",
        );
    }

    #[test]
    fn opt_dynptradd_const() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: ptr = param 0
            %1: ptr = dyn_ptr_add %0, 2i64, 8
            black_box %1
            %3: ptr = dyn_ptr_add %0, -1i64, 8
            black_box %3
            %5: ptr = dyn_ptr_add %0, -8i64, 16
            black_box %5
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: ptr = param ...
            %1: ptr = ptr_add %0, 16
            black_box %1
            %3: ptr = ptr_add %0, -8
            black_box %3
            %5: ptr = ptr_add %0, -128
            black_box %5
        ",
        );
    }

    #[test]
    fn opt_guard_dup() {
        Module::assert_ir_transform_eq(
            "
          entry:
            %0: i8 = param 0
            %1: i8 = param 1
            %2: i1 = eq %0, %1
            guard true, %2, [%0, %1]
            guard true, %2, [%0, %1]
            guard false, %2, [%0, %1]
        ",
            |m| opt(m).unwrap(),
            "
          ...
          entry:
            %0: i8 = param ...
            %1: i8 = param ...
            %2: i1 = eq %0, %1
            guard true, %2, ...
            guard false, %2, ...
        ",
        );
    }
}
