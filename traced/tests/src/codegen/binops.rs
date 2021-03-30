//! Test codegen of binary operators.

use super::{I16_MAX, I32_MAX, I64_MAX, U16_MAX, U32_MAX, U64_MAX};
use paste::paste;
use ykshim_client::{compile_trace, start_tracing, TracingKind};

/// Generates a test for a binary operation.
macro_rules! mk_binop_test {
    ($name: ident, $op: tt, $type: ident, $arg1: expr, $arg2: expr, $expect: expr) => {
        paste! {
            #[test]
            fn [<$name _ $type>]() {
                #[derive(Eq, PartialEq, Debug)]
                struct BinopCtx {
                    arg1: $type,
                    arg2: $type,
                    res: $type,
                }

                impl BinopCtx {
                    fn new(arg1: $type, arg2: $type, res: $type) -> Self {
                        Self { arg1, arg2, res }
                    }
                }

                #[interp_step]
                fn interp_step(ctx: &mut BinopCtx) -> bool {
                    ctx.res = ctx.arg1 $op ctx.arg2;
                    true
                }

                let mut ctx = BinopCtx::new($arg1, $arg2, 0);
                #[cfg(tracermode = "hw")]
                let th = start_tracing(TracingKind::HardwareTracing);
                #[cfg(tracermode = "sw")]
                let th = start_tracing(TracingKind::SoftwareTracing);
                interp_step(&mut ctx);
                let sir_trace = th.stop_tracing().unwrap();
                let ct = compile_trace(sir_trace).unwrap();

                let mut args = BinopCtx::new($arg1, $arg2, 0);
                assert!(unsafe { ct.execute(&mut args).is_null() });
                assert_eq!(args, BinopCtx::new($arg1, $arg2, $expect));
            }

            /// The same test as above, but with $arg1 being a constant.
            #[test]
            fn [<$name _ $type _const_arg1>]() {
                #[derive(Eq, PartialEq, Debug)]
                struct BinopCtx {
                    arg2: $type,
                    res: $type,
                }

                impl BinopCtx {
                    fn new(arg2: $type, res: $type) -> Self {
                        Self { arg2, res }
                    }
                }

                #[interp_step]
                fn interp_step(ctx: &mut BinopCtx) -> bool {
                    ctx.res = $arg1 $op ctx.arg2;
                    true
                }

                let mut ctx = BinopCtx::new($arg2, 0);
                #[cfg(tracermode = "hw")]
                let th = start_tracing(TracingKind::HardwareTracing);
                #[cfg(tracermode = "sw")]
                let th = start_tracing(TracingKind::SoftwareTracing);
                interp_step(&mut ctx);
                let sir_trace = th.stop_tracing().unwrap();
                let ct = compile_trace(sir_trace).unwrap();

                let mut args = BinopCtx::new($arg2, 0);
                assert!(unsafe { ct.execute(&mut args).is_null() });
                assert_eq!(args, BinopCtx::new($arg2, $expect));
            }

            /// And finally a test with a const $arg2.
            #[test]
            fn [<$name _ $type _const_arg2>]() {
                #[derive(Eq, PartialEq, Debug)]
                struct BinopCtx {
                    arg1: $type,
                    res: $type,
                }

                impl BinopCtx {
                    fn new(arg1: $type, res: $type) -> Self {
                        Self { arg1, res }
                    }
                }

                #[interp_step]
                fn interp_step(ctx: &mut BinopCtx) -> bool {
                    ctx.res = ctx.arg1 $op $arg2;
                    true
                }

                let mut ctx = BinopCtx::new($arg1, 0);
                #[cfg(tracermode = "hw")]
                let th = start_tracing(TracingKind::HardwareTracing);
                #[cfg(tracermode = "sw")]
                let th = start_tracing(TracingKind::SoftwareTracing);
                interp_step(&mut ctx);
                let sir_trace = th.stop_tracing().unwrap();
                let ct = compile_trace(sir_trace).unwrap();

                let mut args = BinopCtx::new($arg1, 0);
                assert!(unsafe { ct.execute(&mut args).is_null() });
                assert_eq!(args, BinopCtx::new($arg1, $expect));
            }
        }
    };
}

/// Generates binary operation tests for all unsigned types.
/// Since all types are tested, numeric operands must fit in a u8.
macro_rules! mk_binop_tests_unsigned {
    ($name: ident, $op: tt, $arg1: expr, $arg2: expr, $expect: expr) => {
        mk_binop_test!($name, $op, u8, $arg1, $arg2, $expect);
        mk_binop_test!($name, $op, u16, $arg1, $arg2, $expect);
        mk_binop_test!($name, $op, u32, $arg1, $arg2, $expect);
        mk_binop_test!($name, $op, u64, $arg1, $arg2, $expect);
        // FIXME u128 hits unreachable code.
    };
}

/// Generates binary operation tests for all signed types.
/// Since all types are tested, numeric operands must fit in an i8.
macro_rules! mk_binop_tests_signed {
    ($name: ident, $op: tt, $arg1: expr, $arg2: expr, $expect: expr) => {
        mk_binop_test!($name, $op, i8, $arg1, $arg2, $expect);
        mk_binop_test!($name, $op, i16, $arg1, $arg2, $expect);
        mk_binop_test!($name, $op, i32, $arg1, $arg2, $expect);
        mk_binop_test!($name, $op, i64, $arg1, $arg2, $expect);
        // FIXME i128 hits unreachable code.
    };
}

mk_binop_tests_unsigned!(binop_add1, +, 0, 0, 0);
mk_binop_tests_signed!(binop_add2, +, 0, 0, 0);
mk_binop_tests_unsigned!(binop_add3, +, 1, 1, 2);
mk_binop_tests_signed!(binop_add4, +, 1, 1, 2);
mk_binop_tests_unsigned!(binop_add5, +, 253, 2, 255);
mk_binop_tests_signed!(binop_add6, +, 125, 2, 127);
mk_binop_test!(binop_add7, +, u16, U16_MAX - 7, 7, U16_MAX);
mk_binop_test!(binop_add8, +, u32, U32_MAX - 14, 14, U32_MAX);
mk_binop_test!(binop_add9, +, u64, U64_MAX - 100, 100, U64_MAX);
mk_binop_test!(binop_add10, +, i16, I16_MAX - 7, 7, I16_MAX);
mk_binop_test!(binop_add11, +, i32, I32_MAX - 14, 14, I32_MAX);
mk_binop_test!(binop_add13, +, i64, I64_MAX - 100, 100, I64_MAX);

mk_binop_tests_unsigned!(binop_sub1, -, 0, 0, 0);
mk_binop_tests_signed!(binop_sub2, -, 0, 0, 0);
mk_binop_tests_unsigned!(binop_sub3, -, 1, 0, 1);
mk_binop_tests_signed!(binop_sub4, -, 1, 0, 1);
mk_binop_tests_signed!(binop_sub5, -, 0, 1, -1);
mk_binop_tests_signed!(binop_sub6, -, -120, 8, -128);
mk_binop_tests_signed!(binop_sub7, -, -1, -1, 0);
mk_binop_test!(binop_sub8, -, u16, U16_MAX, 7, U16_MAX - 7);
mk_binop_test!(binop_sub9, -, u32, U32_MAX, 8, U32_MAX - 8);
mk_binop_test!(binop_sub10, -, u64, U64_MAX, 33, U64_MAX - 33);
mk_binop_test!(binop_sub11, -, i16, I16_MAX, 7, I16_MAX - 7);
mk_binop_test!(binop_sub12, -, i32, I32_MAX, 8, I32_MAX - 8);
mk_binop_test!(binop_sub13, -, i64, I64_MAX, 33, I64_MAX - 33);

// FIXME implement and test signed multiplication.
mk_binop_tests_unsigned!(binop_mul1, *, 0, 0, 0);
mk_binop_tests_unsigned!(binop_mul2, *, 10, 10, 100);
mk_binop_tests_unsigned!(binop_mul3, *, 15, 15, 225);
mk_binop_test!(binop_mul4, *, u16, 510, 8, 4080);
mk_binop_test!(binop_mul5, *, u32, 131072, 8, 1048576);
mk_binop_test!(binop_mul5, *, u64, 8589934592u64, 8, 68719476736);

// FIXME implement and test signed division.
mk_binop_tests_unsigned!(binop_div1, /, 1, 1, 1);
mk_binop_tests_unsigned!(binop_div2, /, 2, 1, 2);
mk_binop_tests_unsigned!(binop_div3, /, 252, 4, 63);
mk_binop_test!(binop_div4, /, u16, 4080, 8, 510);
mk_binop_test!(binop_div5, /, u32, 1048576, 8, 131072);
mk_binop_test!(binop_div6, /, u64, 68719476736u64, 8, 8589934592);

#[test]
#[cfg(debug_assertions)]
fn binop_add_overflow() {
    #[derive(Eq, PartialEq, Debug)]
    struct InterpCtx(u8, u8);

    #[interp_step]
    fn interp_stepx(io: &mut InterpCtx) -> bool {
        io.1 = io.0 + 1;
        true
    }

    let mut ctx = InterpCtx(254, 0);
    #[cfg(tracermode = "hw")]
    let th = start_tracing(TracingKind::HardwareTracing);
    #[cfg(tracermode = "sw")]
    let th = start_tracing(TracingKind::SoftwareTracing);
    interp_stepx(&mut ctx);
    let sir_trace = th.stop_tracing().unwrap();
    assert_eq!(ctx.1, 255);
    let ct = compile_trace(sir_trace).unwrap();

    // Executing a trace with no overflow shouldn't fail any guards.
    let mut args = InterpCtx(10, 0);
    assert!(unsafe { ct.execute(&mut args).is_null() });
    assert_eq!(args, InterpCtx(10, 11));

    // Executing a trace *with* overflow will fail a guard.
    let mut args = InterpCtx(255, 5);
    assert!(!unsafe { ct.execute(&mut args).is_null() });
}

#[test]
fn binop_other() {
    #[derive(Eq, PartialEq, Debug)]
    struct InterpCtx(u64, u64, u64);

    #[interp_step]
    fn interp_stepx(io: &mut InterpCtx) -> bool {
        io.2 = io.0 * 3 - 5;
        io.1 = io.2 / 2;
        true
    }

    let mut ctx = InterpCtx(5, 2, 0);
    #[cfg(tracermode = "hw")]
    let th = start_tracing(TracingKind::HardwareTracing);
    #[cfg(tracermode = "sw")]
    let th = start_tracing(TracingKind::SoftwareTracing);
    interp_stepx(&mut ctx);
    let sir_trace = th.stop_tracing().unwrap();
    let ct = compile_trace(sir_trace).unwrap();
    let mut args = InterpCtx(5, 2, 0);
    assert!(unsafe { ct.execute(&mut args).is_null() });
    assert_eq!(args, InterpCtx(5, 5, 10));
}

/// Binary operations where a lot of registers are in use.
/// Designed to test if we've correctly handled clobbering of x86_64 MUL and DIV.
#[test]
fn binop_many_locals() {
    #[derive(Eq, PartialEq, Debug, Clone)]
    struct InterpCtx {
        input: u64,
        mul_res: u64,
        div_res: u64,
        x1: u64,
        x2: u64,
        x3: u64,
        x4: u64,
        x5: u64,
        x6: u64,
        x7: u64,
        x8: u64,
        x9: u64,
        x10: u64,
    }

    #[interp_step]
    fn interp_step(ctx: &mut InterpCtx) -> bool {
        // Make a lot of locals to fill many registers.
        let x1 = ctx.input + 1;
        let x2 = ctx.input + 2;
        let x3 = ctx.input + 3;
        let x4 = ctx.input + 4;
        let x5 = ctx.input + 5;
        let x6 = ctx.input + 6;
        let x7 = ctx.input + 7;
        let x8 = ctx.input + 8;
        let x9 = ctx.input + 9;
        let x10 = ctx.input + 10;

        // Perform some multiplication and division.
        ctx.mul_res = ctx.input * 5;
        ctx.div_res = ctx.input / 4;

        // These should not have been clobbered.
        ctx.x1 = x1;
        ctx.x2 = x2;
        ctx.x3 = x3;
        ctx.x4 = x4;
        ctx.x5 = x5;
        ctx.x6 = x6;
        ctx.x7 = x7;
        ctx.x8 = x8;
        ctx.x9 = x9;
        ctx.x10 = x10;
        true
    }

    let ctx = InterpCtx {
        input: 0,
        mul_res: 0,
        div_res: 0,
        x1: 0,
        x2: 0,
        x3: 0,
        x4: 0,
        x5: 0,
        x6: 0,
        x7: 0,
        x8: 0,
        x9: 0,
        x10: 0,
    };
    let expect = InterpCtx {
        input: 0,
        mul_res: 0,
        div_res: 0,
        x1: 1,
        x2: 2,
        x3: 3,
        x4: 4,
        x5: 5,
        x6: 6,
        x7: 7,
        x8: 8,
        x9: 9,
        x10: 10,
    };

    let mut ctx1 = ctx.clone();
    #[cfg(tracermode = "hw")]
    let th = start_tracing(TracingKind::HardwareTracing);
    #[cfg(tracermode = "sw")]
    let th = start_tracing(TracingKind::SoftwareTracing);
    interp_step(&mut ctx1);
    let sir_trace = th.stop_tracing().unwrap();
    assert_eq!(ctx1, expect);

    let ct = compile_trace(sir_trace).unwrap();
    let mut ctx2 = ctx.clone();
    assert!(unsafe { ct.execute(&mut ctx2).is_null() });
    assert_eq!(ctx2, expect);
}
