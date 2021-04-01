//! Artificially testing the stopgap interpreter (without a guard failure).

use untraced_api::interpret_body;

#[test]
fn simple() {
    struct InterpCtx(u8, u8);
    #[no_mangle]
    fn simple(io: &mut InterpCtx) {
        let a = 3;
        io.1 = a;
    }
    let mut ctx = InterpCtx(0, 0);
    interpret_body("simple", &mut ctx);
    assert_eq!(ctx.1, 3);
}

#[test]
fn tuple() {
    struct InterpCtx((u8, u8, u8));
    #[no_mangle]
    fn func_tuple(io: &mut InterpCtx) {
        let a = io.0;
        let b = a.2;
        (io.0).1 = b;
    }

    let mut ctx = InterpCtx((1, 2, 3));
    interpret_body("func_tuple", &mut ctx);
    assert_eq!(ctx.0, (1, 3, 3));
}

#[test]
fn reference() {
    struct InterpCtx(u8, u8);
    #[no_mangle]
    fn func_ref(io: &mut InterpCtx) {
        let a = 5u8;
        let b = &a;
        io.1 = *b;
    }

    let mut ctx = InterpCtx(5, 0);
    interpret_body("func_ref", &mut ctx);
    assert_eq!(ctx.1, 5);
}

#[test]
fn tupleref() {
    struct InterpCtx((u8, u8));
    #[no_mangle]
    fn func_tupleref(io: &mut InterpCtx) {
        let a = io.0;
        (io.0).1 = 5; // Make sure the line above copies.
        let b = &a;
        (io.0).0 = b.1;
    }

    let mut ctx = InterpCtx((0, 3));
    interpret_body("func_tupleref", &mut ctx);
    assert_eq!(ctx.0, (3, 5));
}

#[test]
fn doubleref() {
    struct InterpCtx((u8, u8));
    #[no_mangle]
    fn func_doubleref(io: &mut InterpCtx) {
        let a = &io.0;
        (io.0).0 = a.1;
    }

    let mut ctx = InterpCtx((0, 3));
    interpret_body("func_doubleref", &mut ctx);
    assert_eq!(ctx.0, (3, 3));
}

#[test]
fn call() {
    struct InterpCtx(u8, u8);

    fn foo(i: u8) -> u8 {
        i
    }

    #[no_mangle]
    fn func_call(io: &mut InterpCtx) {
        let a = foo(5);
        io.0 = a;
    }

    let mut ctx = InterpCtx(0, 0);
    interpret_body("func_call", &mut ctx);
    assert_eq!(ctx.0, 5);
}

#[test]
fn binops_arith() {
    struct InterpCtx(u8, u8);

    #[no_mangle]
    fn add(io: &mut InterpCtx) {
        io.0 = io.0 + io.1;
    }

    let mut ctx = InterpCtx(1, 2);
    interpret_body("add", &mut ctx);
    assert_eq!(ctx.0, 3);
}

#[test]
fn binops_cond() {
    struct InterpCtx(u8, u8, bool);

    #[no_mangle]
    fn lt(io: &mut InterpCtx) {
        io.2 = io.0 < io.1;
    }

    let mut ctx = InterpCtx(1, 2, false);
    interpret_body("lt", &mut ctx);
    assert_eq!(ctx.2, true);
}
