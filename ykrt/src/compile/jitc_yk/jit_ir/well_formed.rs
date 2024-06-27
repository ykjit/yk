//! This module adds some basic well-formedness checks to the JIT IR. These are intended both to
//! help debugging incorrectly formed IR and to provide guarantees about what IR different stages
//! of the compiler can expect.
//!
//! Specifically, after calling [Module::assert_well_formed] one can safely assume:
//!
//!   * [super::BinOpInst]s:
//!       * Have left and right hand side operands with the same [Ty]s.
//!       * Have left and right hand side operands compatible with the operation in question.
//!   * [super::DirectCallInst]s pass the correct number of arguments to a [super::FuncTy] and each
//!     of those arguments has the correct [super::Ty].
//!   * [super::FPExtInst]s:
//!       * Have an float-typed source operand.
//!       * Have a float-type as the destination type operand.
//!       * Have a destination type operand strictly larger than the type of the source operand.
//!   * [super::GuardInst]s:
//!       * Have a `cond` whose type is [super::Ty::Integer(1)] (i.e. an `i1`).
//!       * If `cond` references a constant, that constant matches the guard's `expect` attribute.
//!   * [super::IcmpInst]s left and right hand side operands have the same [Ty]s.
//!   * [super::SIToFPInst]s:
//!       * Have an integer-typed source operand.
//!       * Have a float-type as the destination type operand.
//!       * Have a destination type operand at least as big as the type of the source operand.
//!   * [Const::Int]s cannot use more bits than the corresponding [Ty::Integer] type.

use super::{BinOp, BinOpInst, Const, GuardInst, Inst, Module, Operand, Ty};

impl Module {
    pub(crate) fn assert_well_formed(&self) {
        for iidx in self.iter_skipping_inst_idxs() {
            let inst = self.inst(iidx);
            match inst {
                Inst::BinOp(BinOpInst { lhs, binop, rhs }) => {
                    let lhs_tyidx = lhs.unpack(self).tyidx(self);
                    if lhs_tyidx != rhs.unpack(self).tyidx(self) {
                        panic!(
                            "Instruction at position {iidx} has different types on lhs and rhs\n  {}",
                            self.inst(iidx).display(iidx, self)
                        );
                    }
                    match binop {
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Or
                        | BinOp::And
                        | BinOp::Xor
                        | BinOp::Shl
                        | BinOp::AShr
                        | BinOp::LShr
                        | BinOp::SDiv
                        | BinOp::SRem
                        | BinOp::UDiv
                        | BinOp::URem => {
                            if matches!(self.type_(lhs_tyidx), Ty::Float(_)) {
                                panic!(
                                    "Integer binop at position {iidx} operates on float operands\n  {}",
                                    self.inst(iidx).display(iidx, self)
                                );
                            }
                        }
                        BinOp::FAdd | BinOp::FDiv | BinOp::FMul | BinOp::FRem | BinOp::FSub => {
                            if !matches!(self.type_(lhs_tyidx), Ty::Float(_)) {
                                panic!(
                                    "Float binop at position {iidx} operates on integer operands\n  {}",
                                    self.inst(iidx).display(iidx, self)
                                );
                            }
                        }
                    }
                }
                Inst::Call(x) => {
                    // Check number of parameters/arguments.
                    let fdecl = self.func_decl(x.target());
                    let Ty::Func(fty) = self.type_(fdecl.tyidx()) else {
                        panic!()
                    };
                    if x.num_args() < fty.num_params() {
                        panic!(
                            "Instruction at position {iidx} passing too few arguments:\n  {}",
                            inst.display(iidx, self)
                        );
                    }
                    if x.num_args() > fty.num_params() && !fty.is_vararg() {
                        panic!(
                            "Instruction at position {iidx} passing too many arguments:\n  {}",
                            inst.display(iidx, self)
                        );
                    }

                    // Check parameter/argument types.
                    for (j, (par_ty, arg_ty)) in fty
                        .param_tys()
                        .iter()
                        .zip(x.iter_args_idx().map(|x| self.arg(x).tyidx(self)))
                        .enumerate()
                    {
                        if *par_ty != arg_ty {
                            panic!("Instruction at position {iidx} passing argument {j} of wrong type ({}, but should be {})\n  {}",
                                self.type_(arg_ty).display(self),
                                self.type_(*par_ty).display(self),
                                inst.display(iidx, self));
                        }
                    }
                }
                Inst::Guard(GuardInst { cond, expect, .. }) => {
                    let cond = cond.unpack(self);
                    let tyidx = cond.tyidx(self);
                    let Ty::Integer(1) = self.type_(tyidx) else {
                        panic!(
                            "Guard at position {iidx} does not have 'cond' of type 'i1'\n  {}",
                            self.inst(iidx).display(iidx, self)
                        )
                    };
                    if let Operand::Const(x) = cond {
                        let Const::Int(_, v) = self.const_(x) else {
                            unreachable!()
                        };
                        if (*expect && *v == 0) || (!*expect && *v == 1) {
                            panic!(
                                "Guard at position {iidx} references a constant that is at odds with the guard itself\n  {}",
                                self.inst(iidx).display(iidx, self)
                            );
                        }
                    }
                }
                Inst::Icmp(x) => {
                    if x.lhs(self).tyidx(self) != x.rhs(self).tyidx(self) {
                        panic!(
                            "Instruction at position {iidx} has different types on lhs and rhs\n  {}",
                            self.inst(iidx).display(iidx, self)
                        );
                    }
                }
                Inst::SExt(x) => {
                    if self.type_(x.val(self).tyidx(self)).byte_size()
                        >= self.type_(x.dest_tyidx()).byte_size()
                    {
                        panic!(
                            "Instruction at position {iidx} trying to sign extend from an equal-or-larger-than integer type\n  {}",
                            self.inst(iidx).display(iidx, self)
                        );
                    }
                }
                Inst::SIToFP(x) => {
                    let from_type = self.type_(x.val(self).tyidx(self));
                    let to_type = self.type_(x.dest_ty_idx());

                    if !matches!(from_type, Ty::Integer(_)) {
                        panic!("Instruction at position {iidx} trying to convert a non-integer type\n  {}",
                            self.inst(iidx).display(iidx, self));
                    }
                    if !matches!(to_type, Ty::Float(_)) {
                        panic!("Instruction at position {iidx} trying to convert to a non-float type\n  {}",
                            self.inst(iidx).display(iidx, self));
                    }
                    if to_type.byte_size() < from_type.byte_size() {
                        panic!("Instruction at position {iidx} trying to convert to a smaller-sized float\n  {}",
                            self.inst(iidx).display(iidx, self));
                    }
                }
                Inst::FPExt(x) => {
                    let from_type = self.type_(x.val(self).tyidx(self));
                    let to_type = self.type_(x.dest_ty_idx());
                    if !matches!(from_type, Ty::Float(_)) {
                        panic!("Instruction at position {iidx} trying to extend from a non-float type\n  {}",
                            self.inst(iidx).display(iidx, self));
                    }
                    if !matches!(to_type, Ty::Float(_)) {
                        panic!("Instruction at position {iidx} trying to extend to a non-float type\n  {}",
                            self.inst(iidx).display(iidx, self));
                    }
                    if to_type.byte_size() <= from_type.byte_size() {
                        panic!("Instruction at position {iidx} trying to extend to a smaller-sized float\n  {}",
                            self.inst(iidx).display(iidx, self));
                    }
                }
                _ => (),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BinOp, BinOpInst, Const, Inst, Module, Operand};

    #[should_panic(expected = "Instruction at position 0 passing too few arguments")]
    #[test]
    fn too_few_args() {
        Module::from_str(
            "
              func_decl f(i32)
              entry:
                call @f()
            ",
        );
    }

    #[should_panic(expected = "Instruction at position 0 passing too few arguments")]
    #[test]
    fn too_few_args2() {
        Module::from_str(
            "
              func_decl f(i32, ...)
              entry:
                call @f()
            ",
        );
    }

    #[should_panic(expected = "Instruction at position 1 passing too many arguments")]
    #[test]
    fn too_many_args() {
        Module::from_str(
            "
              func_decl f()
              entry:
                %0: i8 = load_ti 0
                call @f(%0)
            ",
        );
    }

    #[test]
    fn var_args() {
        Module::from_str(
            "
              func_decl f(...)
              entry:
                %0: i8 = load_ti 0
                call @f(%0)
            ",
        );
    }

    #[should_panic(
        expected = "Instruction at position 1 passing argument 0 of wrong type (i8, but should be i32)"
    )]
    #[test]
    fn cg_call_bad_arg_type() {
        Module::from_str(
            "
              func_decl f(i32) -> i32
              entry:
                %0: i8 = load_ti 0
                %1: i32 = call @f(%0)
            ",
        );
    }

    #[should_panic(expected = "Instruction at position 0 has different types on lhs and rhs")]
    #[test]
    fn cg_add_wrong_types() {
        // The parser will reject a binop with a result type different from either operand, so to
        // get the test we want, we can't use the parser.
        let mut m = Module::new(0, 0).unwrap();
        let c1 = m.insert_const(Const::Int(m.int1_tyidx(), 0)).unwrap();
        let c2 = m.insert_const(Const::Int(m.int8_tyidx(), 0)).unwrap();
        m.push(Inst::BinOp(BinOpInst::new(
            Operand::Const(c1),
            BinOp::Add,
            Operand::Const(c2),
        )))
        .unwrap();
        m.assert_well_formed();
    }

    #[test]
    #[should_panic(expected = "Instruction at position 2 has different types on lhs and rhs")]
    fn cg_icmp_diff_types() {
        Module::from_str(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i64 = load_ti 0
                %2: i1 = eq %0, %1
            ",
        );
    }

    #[test]
    #[should_panic(
        expected = "Instruction at position 1 trying to sign extend from an equal-or-larger-than integer type"
    )]
    fn sign_extend_wrong_size() {
        Module::from_str(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = sext %0
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Guard at position 1 does not have 'cond' of type 'i1'")]
    fn guard_i1() {
        Module::from_str(
            "
              entry:
                %0: i8 = load_ti 0
                guard %0, true
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Instruction at position 1 trying to convert a non-integer type")]
    fn si_to_fp_from_non_int() {
        Module::from_str(
            "
              entry:
                %0: float = load_ti 0
                %1: float = si_to_fp %0
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Instruction at position 1 trying to convert to a non-float type")]
    fn si_to_fp_to_non_float() {
        Module::from_str(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i64 = si_to_fp %0
            ",
        );
    }

    #[test]
    #[should_panic(
        expected = "Instruction at position 1 trying to convert to a smaller-sized float"
    )]
    fn si_to_fp_smaller() {
        Module::from_str(
            "
              entry:
                %0: i64 = load_ti 0
                %1: float = si_to_fp %0
            ",
        );
    }

    #[test]
    #[should_panic(
        expected = "Instruction at position 1 trying to extend to a smaller-sized float"
    )]
    fn fp_ext_smaller() {
        Module::from_str(
            "
              entry:
                %0: double = load_ti 0
                %1: float = fp_ext %0
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Instruction at position 1 trying to extend from a non-float type")]
    fn fp_ext_from_non_float() {
        Module::from_str(
            "
              entry:
                %0: i32 = load_ti 0
                %1: double = fp_ext %0
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Instruction at position 1 trying to extend to a non-float type")]
    fn fp_ext_to_non_float() {
        Module::from_str(
            "
              entry:
                %0: float = load_ti 0
                %1: i64 = fp_ext %0
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Integer binop at position 1 operates on float operands")]
    fn int_binop_with_float_opnds() {
        Module::from_str(
            "
              entry:
                %0: float = load_ti 0
                %1: float = add %0, %0
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Float binop at position 1 operates on integer operands")]
    fn float_binop_with_int_opnds() {
        Module::from_str(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = fadd %0, %0
            ",
        );
    }
}
