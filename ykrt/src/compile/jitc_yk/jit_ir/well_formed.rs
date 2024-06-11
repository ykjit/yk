//! This module adds some basic well-formedness checks to the JIT IR. These are intended both to
//! help debugging incorrectly formed IR and to provide guarantees about what IR different stages
//! of the compiler can expect.
//!
//! Specifically, after calling [Module::assert_well_formed] one can safely assume:
//!
//!   * [super::BinOpInst]s left and right hand side operands have the same [Ty]s.
//!   * [super::DirectCallInst]s pass the correct number of arguments to a [super::FuncTy] and each
//!     of those arguments has the correct [super::Ty].
//!   * [super::GuardInst]s:
//!       * Have a `cond` whose type is [super::Ty::Integer(1)] (i.e. an `i1`).
//!       * If `cond` references a constant, that constant matches the guard's `expect` attribute.
//!   * [super::ICmpInst]s left and right hand side operands have the same [Ty]s.
//!   * [Const::Int]s cannot use more bits than the corresponding [Ty::Integer] type.

use super::{BinOpInst, Const, GuardInst, Inst, InstIdx, Module, Operand, Ty};

impl Module {
    pub(crate) fn assert_well_formed(&self) {
        for (i, inst) in self.insts.iter().enumerate() {
            match inst {
                Inst::BinOp(BinOpInst { lhs, binop: _, rhs }) => {
                    if lhs.unpack(self).ty_idx(self) != rhs.unpack(self).ty_idx(self) {
                        let inst_idx = InstIdx::new(i).unwrap();
                        panic!(
                            "Instruction at position {} has different types on lhs and rhs\n  {}",
                            usize::from(inst_idx),
                            self.inst(inst_idx).display(inst_idx, self)
                        );
                    }
                }
                Inst::Call(x) => {
                    // Check number of parameters/arguments.
                    let fdecl = self.func_decl(x.target());
                    let Ty::Func(fty) = self.type_(fdecl.ty_idx()) else {
                        panic!()
                    };
                    if x.num_args() < fty.num_params() {
                        panic!(
                            "Instruction at position {i} passing too few arguments:\n  {}",
                            inst.display(InstIdx::new(i).unwrap(), self)
                        );
                    }
                    if x.num_args() > fty.num_params() && !fty.is_vararg() {
                        panic!(
                            "Instruction at position {i} passing too many arguments:\n  {}",
                            inst.display(InstIdx::new(i).unwrap(), self)
                        );
                    }

                    // Check parameter/argument types.
                    for (j, (par_ty, arg_ty)) in fty
                        .param_tys()
                        .iter()
                        .zip(x.iter_args_idx().map(|x| self.arg(x).ty_idx(self)))
                        .enumerate()
                    {
                        if *par_ty != arg_ty {
                            panic!("Instruction at position {i} passing argument {j} of wrong type ({}, but should be {})\n  {}",
                                self.type_(arg_ty).display(self),
                                self.type_(*par_ty).display(self),
                                inst.display(InstIdx::new(i).unwrap(), self));
                        }
                    }
                }
                Inst::Guard(GuardInst { cond, expect, .. }) => {
                    let cond = cond.unpack(self);
                    let tyidx = cond.ty_idx(self);
                    let Ty::Integer(1) = self.type_(tyidx) else {
                        let inst_idx = InstIdx::new(i).unwrap();
                        panic!(
                            "Guard at position {} does not have 'cond' of type 'i1'\n  {}",
                            usize::from(inst_idx),
                            self.inst(inst_idx).display(inst_idx, self)
                        )
                    };
                    if let Operand::Const(x) = cond {
                        let Const::Int(_, v) = self.const_(x) else {
                            unreachable!()
                        };
                        if (*expect && *v == 0) || (!*expect && *v == 1) {
                            let inst_idx = InstIdx::new(i).unwrap();
                            panic!(
                                "Guard at position {} references a constant that is at odds with the guard itself\n  {}",
                                usize::from(inst_idx),
                                self.inst(inst_idx).display(inst_idx, self)
                            );
                        }
                    }
                }
                Inst::Icmp(x) => {
                    if x.lhs(self).ty_idx(self) != x.rhs(self).ty_idx(self) {
                        let inst_idx = InstIdx::new(i).unwrap();
                        panic!(
                            "Instruction at position {} has different types on lhs and rhs\n  {}",
                            usize::from(inst_idx),
                            self.inst(inst_idx).display(inst_idx, self)
                        );
                    }
                }
                Inst::SExt(x) => {
                    if self.type_(x.val(self).ty_idx(self)).byte_size()
                        >= self.type_(x.dest_ty_idx()).byte_size()
                    {
                        let inst_idx = InstIdx::new(i).unwrap();
                        panic!(
                            "Instruction at position {} trying to sign extend from an equal-or-larger-than integer type\n  {}",
                            usize::from(inst_idx),
                            self.inst(inst_idx).display(inst_idx, self)
                        );
                    }
                }
                _ => (),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Module;

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

    #[should_panic(expected = "Instruction at position 2 has different types on lhs and rhs")]
    #[test]
    fn cg_add_wrong_types() {
        Module::from_str(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i32 = load_ti 1
                %2: i32 = add %0, %1
            ",
        );
    }

    #[test]
    #[should_panic(expected = "Instruction at position 2 has different types on lhs and rhs")]
    fn cg_icmp_diff_types() {
        Module::from_str(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i64 = load_ti 0
                %2: i8 = eq %0, %1
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
}
