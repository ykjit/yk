//! This module adds some basic well-formedness checks to the JIT IR. These are intended both to
//! help debugging incorrectly formed IR and to provide guarantees about what IR different stages
//! of the compiler can expect.
//!
//! Specifically, after calling [Module::assert_well_formed] one can safely assume:
//!
//!   * [super::DirectCallInst]s pass the correct number of arguments to a [super::FuncTy] and each
//!   of those arguments has the correct [super::Ty].
//!   * Binary operations' left and right hand side operands have the same [Ty]s.

use super::{Inst, InstIdx, Module, Ty};

impl Module {
    pub(crate) fn assert_well_formed(&self) {
        for (i, inst) in self.insts.iter().enumerate() {
            match inst {
                Inst::BinOp(x) => {
                    if x.lhs().ty_idx(self) != x.rhs().ty_idx(self) {
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

    #[cfg(debug_assertions)]
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

    #[cfg(debug_assertions)]
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
}
