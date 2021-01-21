use ykshim_client::interpret_body;

#[test]
fn simple() {
    struct IO(u8, u8);
    #[no_mangle]
    fn simple(io: &mut IO) {
        let a = 3;
        io.1 = a;
    }
    let mut tio = IO(0, 0);
    interpret_body("simple", &mut tio);
    assert_eq!(tio.1, 3);
}

#[test]
fn tuple() {
    struct IO((u8, u8, u8));
    #[no_mangle]
    fn func_tuple(io: &mut IO) {
        let a = io.0;
        let b = a.2;
        (io.0).1 = b;
    }

    let mut tio = IO((1, 2, 3));
    interpret_body("func_tuple", &mut tio);
    assert_eq!(tio.0, (1, 3, 3));
}

#[test]
fn reference() {
    struct IO(u8, u8);
    #[no_mangle]
    fn func_ref(io: &mut IO) {
        let a = 5u8;
        let b = &a;
        io.1 = *b;
    }

    let mut tio = IO(5, 0);
    interpret_body("func_ref", &mut tio);
    assert_eq!(tio.1, 5);
}

#[test]
fn tupleref() {
    struct IO((u8, u8));
    #[no_mangle]
    fn func_tupleref(io: &mut IO) {
        let a = io.0;
        (io.0).1 = 5; // Make sure the line above copies.
        let b = &a;
        (io.0).0 = b.1;
    }

    let mut tio = IO((0, 3));
    interpret_body("func_tupleref", &mut tio);
    assert_eq!(tio.0, (3, 5));
}

#[test]
fn doubleref() {
    struct IO((u8, u8));
    #[no_mangle]
    fn func_doubleref(io: &mut IO) {
        let a = &io.0;
        (io.0).0 = a.1;
    }

    let mut tio = IO((0, 3));
    interpret_body("func_doubleref", &mut tio);
    assert_eq!(tio.0, (3, 3));
}

#[test]
fn call() {
    struct IO(u8, u8);

    fn foo(i: u8) -> u8 {
        i
    }

    #[no_mangle]
    fn func_call(io: &mut IO) {
        let a = foo(5);
        io.0 = a;
    }

    let mut tio = IO(0, 0);
    interpret_body("func_call", &mut tio);
    assert_eq!(tio.0, 5);
}
