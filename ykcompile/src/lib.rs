#![feature(test)]
extern crate test;

use yktrace::tir::{TirTrace, TirOp, Statement, PlaceBase, Rvalue, Operand, Constant, ConstantInt,
UnsignedInt};

use assembler::{ExecutableAnonymousMemoryMap, InstructionStreamHints, InstructionStream};
use assembler::mnemonic_parameter_types::registers::*;
use assembler::mnemonic_parameter_types::immediates::*;

fn local_to_reg(l: u32) -> Register64Bit {
    match l {
        0 => Register64Bit::RAX,
        1 => Register64Bit::RDX,
        _ => panic!("ignore")
    }
}

pub fn compile_local_usize(is: &mut InstructionStream, a: u32, c: usize) {
    if a <= 1 { // Ignore other locals for now
        let left = local_to_reg(a);
        is.mov_Register64Bit_Immediate64Bit(left, Immediate64Bit::from(c as u64));
        println!("mov r{:?} i{:?}", left, c);
    }
}

pub fn compile_local_local(is: &mut InstructionStream, var1: u32, var2: u32) {
    if var1 <= 1 && var2 <= 1 { // Ignore other locals for now
        let left = local_to_reg(var1);
        let right = local_to_reg(var2);
        is.cmova_Register64Bit_Register64Bit(left, right);
        println!("mov r{:?} r{:?}", left, right);
    }
}

pub fn compile_stmt(is: &mut InstructionStream, st: &Statement) {
    match st {
        Statement::Assign(l, r) => {
            let local = match l.base {
                PlaceBase::Local(l) => l.0,
                PlaceBase::Static => panic!("Not implemented: Static")
            };
            let value = match r {
                Rvalue::Use(Operand::Place(p)) => {
                    match p.base {
                        PlaceBase::Local(l) => compile_local_local(is, local, l.0),
                        PlaceBase::Static => panic!("Not implemented: Static")
                    }
                },
                Rvalue::Use(Operand::Constant(c)) => {
                    match c {
                        Constant::Int(ci) => {
                            match ci {
                                ConstantInt::UnsignedInt(UnsignedInt::Usize(i)) => {
                                    compile_local_usize(is, local, *i)
                                },
                                e => panic!("SignedInt, etc: {}", e)
                            }
                        }
                        _ => panic!("Not implemented: int")
                    }
                }
                _ => panic!("Not implemented: Everything else")
            };
        },
        Statement::Nop => {},
        Statement::Unimplemented(mir_stmt) => println!("Can't compile: {}", mir_stmt)
    }
}

pub fn compile_trace(tt: TirTrace) -> u64 {
    dbg!(&tt);

    let mut memory_map = ExecutableAnonymousMemoryMap::new(4096, true, true).unwrap();
    let mut is = memory_map.instruction_stream(&InstructionStreamHints::default());
    let func: unsafe extern "C" fn () -> u64 = is.nullary_function_pointer();

    for i in 0..tt.len() {
        let t = tt.op(i);
        match t {
            TirOp::Statement(st) => compile_stmt(&mut is, st),
            TirOp::Guard(_) => { println!("Not implemented: Guard") }
        }
    }

    is.ret();
    is.finish();
    unsafe {
        func()
    }
}

#[cfg(test)]
mod tests {

    use super::compile_trace;
    use yktrace::{start_tracing, TracingKind};
    use yktrace::tir::TirTrace;
    use test::black_box;
    use yktrace::debug;

    #[inline(never)]
    fn simple() -> usize {
        let x = 13;
        x
    }

    fn simple2(a: usize) -> usize {
        let y = a + 3;
        y
    }

    trait Foo {
        fn bar(&self) -> usize;
    }

    struct S {}
    impl Foo for S {
        fn bar(&self) -> usize {
            return 2;
        }
    }

    #[test]
    pub(crate) fn test_simple() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        simple();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let r = compile_trace(tir_trace);
        assert_eq!(r, 13);
    }

    #[test]
    pub(crate) fn test_simple2() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let k = simple();
        simple2(k);
        let sir_trace = th.stop_tracing().unwrap();
        //assert_eq!(k, 13);
        dbg!(&sir_trace);
        debug::print_sir_trace(&*sir_trace, false, true);
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let r = compile_trace(tir_trace);
        assert_eq!(r, 16);
    }
}
