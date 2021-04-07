//! Test codegen of conditional branches.

use super::{
    I16_MAX, I16_MIN, I32_MAX, I32_MIN, I64_MAX, I64_MIN, I8_MAX, I8_MIN, U16_MAX, U16_MIN,
    U32_MAX, U32_MIN, U64_MAX, U64_MIN, U8_MAX, U8_MIN,
};
use paste::paste;
use to_untraced::{compile_trace, start_tracing, TracingKind};

/// Generates a test for a binary operation.
macro_rules! mk_cond_br_tests {
    ($name: ident, $op: tt, $type: ident, $arg1: expr, $arg2: expr) => {
        paste! {
            #[test]
            fn $name() {
                #[derive(Eq, PartialEq, Debug)]
                struct CmpCtx {
                    arg1: $type,
                    arg2: $type,
                    res: u8,
                }

                impl CmpCtx {
                    fn new(arg1: $type, arg2: $type, res: u8) -> Self {
                        Self { arg1, arg2, res }
                    }
                }

                #[interp_step]
                fn interp_step(ctx: &mut CmpCtx) -> bool {
                    if ctx.arg1 $op ctx.arg2 {
                        ctx.res = 1;
                    } else {
                        ctx.res = 0;
                    }
                    true
                }

                let mut ctx = CmpCtx::new($arg1, $arg2, 255);
                #[cfg(tracermode = "hw")]
                let th = start_tracing(TracingKind::HardwareTracing);
                #[cfg(tracermode = "sw")]
                let th = start_tracing(TracingKind::SoftwareTracing);
                interp_step(&mut ctx);
                let sir_trace = th.stop_tracing().unwrap();
                let ct = compile_trace(sir_trace).unwrap();

                let mut args = CmpCtx::new($arg1, $arg2, 255);
                assert!(unsafe { ct.execute(&mut args).is_null() });
                if $arg1 $op $arg2 {
                    assert_eq!(args, CmpCtx::new($arg1, $arg2, 1));
                } else {
                    assert_eq!(args, CmpCtx::new($arg1, $arg2, 0));
                }
            }

            // FIXME generate tests for constant operands.
            // Currently hits unimplemented! code.
        }
    };
}

/// Generate conditional branch tests for the given type and operator.
macro_rules! mk_cond_br_boundary_tests_op {
    ($name: ident, $op: tt, $ty_lower: ident, $ty_upper: ident) => {
        paste! {
            mk_cond_br_tests!([<$name _min1>], $op, $ty_lower, [<$ty_upper _MIN>], [<$ty_upper _MIN>]);
            mk_cond_br_tests!([<$name _min2>], $op, $ty_lower, [<$ty_upper _MIN>], [<$ty_upper _MIN>] + 1);
            mk_cond_br_tests!([<$name _min3>], $op, $ty_lower, [<$ty_upper _MIN>] + 1, [<$ty_upper _MIN>]);
            mk_cond_br_tests!([<$name _max1>], $op, $ty_lower, [<$ty_upper _MAX>], [<$ty_upper _MAX>]);
            mk_cond_br_tests!([<$name _max2>], $op, $ty_lower, [<$ty_upper _MAX>], [<$ty_upper _MAX>] - 1);
            mk_cond_br_tests!([<$name _max3>], $op, $ty_lower, [<$ty_upper _MAX>] - 1, [<$ty_upper _MAX>]);
            mk_cond_br_tests!([<$name _minmax>], $op, $ty_lower, [<$ty_upper _MIN>], [<$ty_upper _MAX>]);
            mk_cond_br_tests!([<$name _maxmin>], $op, $ty_lower, [<$ty_upper _MAX>], [<$ty_upper _MIN>]);
        }
    }
}

/// Generate conditional branch tests for the given type.
macro_rules! mk_cond_br_boundary_tests_ty {
    ($ty_lower: ident, $ty_upper: ident) => {
        paste! {
            mk_cond_br_boundary_tests_op!([<cond_br_ $ty_lower _lt>], <, $ty_lower, $ty_upper);
            mk_cond_br_boundary_tests_op!([<cond_br_ $ty_lower _gt>], >, $ty_lower, $ty_upper);
            mk_cond_br_boundary_tests_op!([<cond_br_ $ty_lower _eq>], ==, $ty_lower, $ty_upper);
            mk_cond_br_boundary_tests_op!([<cond_br_ $ty_lower _neq>], !=, $ty_lower, $ty_upper);
        }
    };
}

mk_cond_br_boundary_tests_ty!(u8, U8);
mk_cond_br_boundary_tests_ty!(i8, I8);
mk_cond_br_boundary_tests_ty!(u16, U16);
mk_cond_br_boundary_tests_ty!(i16, I16);
mk_cond_br_boundary_tests_ty!(u32, U32);
mk_cond_br_boundary_tests_ty!(i32, I32);
mk_cond_br_boundary_tests_ty!(u64, U64);
mk_cond_br_boundary_tests_ty!(i64, I64);
