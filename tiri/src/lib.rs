// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tiri -- TIR trace interpreter.
//!
//! No effort has been made to make this fast.

use yktrace::tir::{Guard, Statement, TirOp, TirTrace};

/// Storage space for one local variable.
/// FIXME: Not yet populated.
struct Local {}

/// Mutable interpreter state.
struct InterpState {
    /// The next position in the trace to interpret.
    trace_pos: usize,
    /// Local variable store.
    _locals: Vec<Local>,
}

impl InterpState {
    fn new() -> Self {
        Self {
            trace_pos: 0,
            _locals: Vec::new(),
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
    pub fn run(&self) {
        let state = InterpState::new();

        // The main interpreter loop.
        loop {
            let op = self.trace.op(state.trace_pos);
            match op {
                TirOp::Statement(st) => self.interp_stmt(st),
                TirOp::Guard(g) => self.interp_guard(g),
            }
        }
    }

    /// Interpret the specified statement.
    fn interp_stmt(&self, _stmt: &Statement) {
        unimplemented!();
    }

    /// Interpret the specified terminator.
    fn interp_guard(&self, _guard: &Guard) {
        unimplemented!();
    }
}
