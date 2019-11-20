//! Tiri -- TIR trace interpreter.
//!
//! This is a proof of concept interpreter for TIR traces. It is fairly simple since TIR traces are
//! straight line code for one path through a SIR flow-graph and (most) conventional control flow
//! has been replaced by guards (calls can still appear in traces).
//!
//! The interpreter takes as input a TIR trace and repeatedly evaluates TIR statements at the index
//! indicated by the program counter (initially zero). Each statement will update the program
//! counter accordingly.
//!
//! FIXME: talk about memory once it is implemented.
//!
//! FIXME: talk about calls once they are implemented.
//!
//! Interpretation continues until either a guard fails, or the end of the trace is reached. When
//! interpretation is complete, the interpreter state is returned, allowing us to inspect the
//! values of local variables and whether the trace prematurely aborted.
//!
//! No effort has been made to make this fast.

#![feature(exclusive_range_pattern)]
#![feature(test)]
extern crate test;

use std::collections::HashMap;
use yktrace::tir::{
    BinOp, Constant, ConstantInt, Guard, GuardKind, Local, Operand, Place, PlaceBase,
    PlaceProjection, Rvalue, SignedInt, Statement, TirOp, TirTrace, UnsignedInt,
};

/// The number of Tir statements to print either side of the PC when dumping the interpreter state.
#[allow(dead_code)]
const DUMP_CONTEXT: usize = 4;

/// A `Value` represents the contents of a local variable.
#[derive(Debug, Clone, Eq, PartialEq)]
enum Value {
    /// A small (u128-sized or smaller) value.
    Scalar(Scalar),
    /// A pair of the above, used for checked operations.
    ScalarPair(Scalar, Scalar),
    // FIXME -- Implement memory accesses.
}

impl Value {
    /// A helper to create a raw scalar value of the given value and size.
    #[cfg(test)]
    fn raw_scalar(data: u128, size: u8) -> Self {
        Self::Scalar(Scalar::Raw { data, size })
    }

    /// A helper to cast a value to a Boolean, if appropriate.
    fn as_bool(&self) -> bool {
        match self {
            // Casting a `ScalarPair` to a `bool` performs a scalar cast on the second element.
            // This is used to inspect the overflow status of a checked binary operation.
            Self::Scalar(s) | Self::ScalarPair(_, s) => s.as_bool(),
        }
    }

    /// A helper to get the u128 value from inside a raw immediate.
    fn as_u128(&self) -> u128 {
        match self {
            Self::Scalar(Scalar::Raw { data, .. }) => *data,
            _ => panic!("Invalid cast from Value to u128"),
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Scalar(Scalar::from(b))
    }
}

/// A single u128-sized (or smaller) value.
#[derive(Debug, Clone, Eq, PartialEq)]
enum Scalar {
    Raw { data: u128, size: u8 },
    _Ptr, // FIXME Not implemented.
}

impl From<bool> for Scalar {
    fn from(b: bool) -> Self {
        if b {
            Scalar::Raw { data: 1, size: 1 }
        } else {
            Scalar::Raw { data: 0, size: 1 }
        }
    }
}

impl Scalar {
    fn as_bool(&self) -> bool {
        let data = match self {
            Scalar::Raw { data, .. } => data,
            Scalar::_Ptr => panic!("Invalid cast from Scalar to bool"),
        };
        *data != 0
    }
}

/// Mutable interpreter state.
pub struct InterpState {
    /// The next position in the trace to interpret.
    pc: usize,
    /// Local variable store.
    locals: HashMap<u32, Value>,
    /// Set true upon aborting the execution of a trace.
    abort: bool,
}

impl InterpState {
    pub fn new() -> Self {
        Self {
            pc: 0,
            locals: HashMap::new(),
            abort: false,
        }
    }
}

impl InterpState {
    fn store_local(&mut self, local: Local, val: Value) {
        self.locals.insert(local.0, val);
    }

    fn local(&self, l: Local) -> Value {
        match self.locals.get(&l.0) {
            Some(v) => v.clone(),
            None => panic!("uninitialised read from ${}", l.0),
        }
    }
}

/// The interpreter itself.
/// The struct itself holds only immutable program information.
pub struct Interp<'t> {
    trace: &'t TirTrace,
}

impl<'t> Interp<'t> {
    /// Create a new interpreter, using the TIR found in the `.yk_tir` section of the binary `bin`.
    pub fn new(trace: &'t TirTrace) -> Self {
        Self { trace }
    }

    /// Start interpreting the trace.
    pub fn run(&self, mut state: InterpState) -> InterpState {
        let trace_len = self.trace.len();

        // The main interpreter loop.
        while !state.abort && state.pc < trace_len {
            let op = self.trace.op(state.pc);
            match op {
                TirOp::Statement(stmt) => self.interp_stmt(&mut state, stmt),
                TirOp::Guard(grd) => self.interp_guard(&mut state, grd),
            };
        }

        state
    }

    /// Prints diagnostic information about the interpreter state.
    /// Used during development/debugging only.
    #[allow(dead_code)]
    fn dump(&self, state: &InterpState) {
        // Dump the code.
        let start = match state.pc {
            0..DUMP_CONTEXT => 0,
            _ => state.pc - DUMP_CONTEXT,
        };
        let end = (self.trace.len() - 1).min(state.pc + DUMP_CONTEXT);

        eprintln!("[Begin Interpreter State Dump]");
        eprintln!("     pc: {}\n", state.pc);
        for idx in start..end {
            let op = self.trace.op(idx);
            let pc_str = if idx == state.pc { "->" } else { "  " };

            eprintln!("  {} {}: {}", pc_str, idx, op);
        }
        eprintln!();

        // Dump the locals.
        for (idx, val) in &state.locals {
            eprintln!("     ${}: {:?}", idx, val);
        }
        eprintln!("[End Interpreter State Dump]\n");
    }

    /// Interpret the specified statement.
    fn interp_stmt(&self, state: &mut InterpState, stmt: &Statement) {
        match stmt {
            Statement::Assign(plc, rval) => match (&plc.base, &plc.projections.len()) {
                (PlaceBase::Local(l), 0) => {
                    state.store_local(*l, self.eval_rvalue(state, rval).clone());
                    state.pc += 1;
                }
                _ => unimplemented!("unhandled assignment"),
            },
            _ => panic!("unhandled statement: {}", stmt),
        }
    }

    fn eval_rvalue(&self, state: &InterpState, rval: &Rvalue) -> Value {
        match rval {
            Rvalue::Use(o) => self.eval_operand(state, o),
            Rvalue::BinaryOp(oper, o1, o2) => self.eval_binop(state, oper, o1, o2, false),
            Rvalue::CheckedBinaryOp(oper, o1, o2) => self.eval_binop(state, oper, o1, o2, true),
            _ => panic!("unimplemented rvalue eval"),
        }
    }

    /// Evaluate the left-hand side of an assignment.
    fn eval_place(&self, state: &InterpState, place: &Place) -> Value {
        let mut val = match place.base {
            PlaceBase::Local(l) => state.local(l),
            PlaceBase::Static => unimplemented!("static place eval"),
        };

        // Apply any projections.
        for proj in &place.projections {
            match (val, proj) {
                (Value::ScalarPair(ref s1, ref s2), PlaceProjection::Field(idx)) => {
                    let sclr = match idx {
                        0 => s1,
                        1 => s2,
                        _ => panic!("out of bounds index for ScalarPair projection"),
                    };
                    val = Value::Scalar(sclr.clone());
                }
                _ => unimplemented!("unhandled place projection"),
            }
        }

        val
    }

    fn eval_constant(&self, cst: &Constant) -> Value {
        let (data, size) = match cst {
            Constant::Int(ci) => {
                // We store all integers as untyped bits (a u128).
                match ci {
                    ConstantInt::UnsignedInt(ui) => match ui {
                        UnsignedInt::U8(v) => (*v as u128, 1),
                        UnsignedInt::U16(v) => (*v as u128, 2),
                        UnsignedInt::U32(v) => (*v as u128, 4),
                        UnsignedInt::U64(v) => (*v as u128, 8),
                        UnsignedInt::U128(v) => (v.val(), 16),
                        UnsignedInt::Usize(v) => (*v as u128, std::mem::size_of::<usize>() as u8),
                    },
                    ConstantInt::SignedInt(si) => match si {
                        SignedInt::I8(v) => (*v as u128, 1),
                        SignedInt::I16(v) => (*v as u128, 2),
                        SignedInt::I32(v) => (*v as u128, 4),
                        SignedInt::I64(v) => (*v as u128, 8),
                        SignedInt::I128(v) => (v.val() as u128, 16),
                        SignedInt::Isize(v) => (*v as u128, std::mem::size_of::<isize>() as u8),
                    },
                }
            }
            Constant::Unimplemented => unimplemented!(),
        };

        Value::Scalar(Scalar::Raw { data, size })
    }

    fn eval_operand(&self, state: &InterpState, opnd: &Operand) -> Value {
        match opnd {
            Operand::Place(p) => self.eval_place(state, p),
            Operand::Constant(c) => self.eval_constant(c),
        }
    }

    fn eval_binop(
        &self,
        state: &InterpState,
        oper: &BinOp,
        o1: &Operand,
        o2: &Operand,
        checked: bool,
    ) -> Value {
        let c1 = self.eval_operand(state, o1);
        let c2 = self.eval_operand(state, o2);

        match (&c1, &c2) {
            (
                Value::Scalar(Scalar::Raw { data: d1, size: s1 }),
                Value::Scalar(Scalar::Raw { data: d2, size: s2 }),
            ) => {
                assert_eq!(s1, s2);
                match oper {
                    BinOp::Add => {
                        let raw_res = d1.overflowing_add(*d2);
                        let sclr = Scalar::Raw {
                            data: raw_res.0,
                            size: *s1,
                        };
                        if checked {
                            // In a checked operation, we generate a pair. The first element is the
                            // result, and the second element indicates overflow.
                            Value::ScalarPair(sclr, Scalar::from(raw_res.1))
                        } else {
                            Value::Scalar(sclr)
                        }
                    }
                    BinOp::Lt => {
                        // FIXME what about signed comparisons?
                        assert!(!checked);
                        Value::from(d1 < d2)
                    }
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }
    }

    /// Interpret the specified terminator.
    fn interp_guard(&self, state: &mut InterpState, guard: &Guard) {
        match &guard.kind {
            GuardKind::OtherInteger(members) => {
                if let Value::Scalar(Scalar::Raw { data, .. }) = self.eval_place(state, &guard.val)
                {
                    if members.contains(&data) {
                        state.abort = true;
                        return;
                    }
                    state.pc += 1;
                } else {
                    panic!("invalid guard: OtherInteger");
                }
            }
            GuardKind::Boolean(must_eq) => {
                if self.eval_place(state, &guard.val).as_bool() != *must_eq {
                    state.abort = true;
                    return;
                }
                state.pc += 1;
            }
            GuardKind::Integer(must_eq) => {
                if self.eval_place(state, &guard.val).as_u128() != *must_eq {
                    state.abort = true;
                    return;
                }
                state.pc += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Interp, InterpState, Value};
    use test::black_box;
    use yktrace::tir::{Local, TirTrace};
    use yktrace::{start_tracing, TracingKind};

    // Some work to trace.
    #[inline(never)]
    fn work(x: usize, y: usize) -> usize {
        let mut res = 0;
        while res < y {
            res += x;
        }
        res
    }

    #[test]
    fn interp_simple_trace() {
        let tracer = start_tracing(Some(TracingKind::SoftwareTracing));
        let res = work(black_box(3), black_box(13));
        let sir_trace = tracer.stop_tracing().unwrap();
        assert_eq!(res, 15);

        let tir_trace = TirTrace::new(sir_trace.as_ref()).unwrap();
        assert!(tir_trace.len() > 0);

        let interp = Interp::new(&tir_trace);
        let mut state = InterpState::new();
        state.store_local(Local(1), Value::raw_scalar(3, 8));
        state.store_local(Local(2), Value::raw_scalar(13, 8));
        let state = interp.run(state);

        assert!(!state.abort); // i.e. normal exit via end of trace.
        assert_eq!(state.local(Local(0)), Value::raw_scalar(15, 8));
    }
}
