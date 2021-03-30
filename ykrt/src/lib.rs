//! Interpreter-facing API to the Yorick meta-tracer.

#![cfg_attr(test, feature(test))]

mod location;
pub(crate) mod mt;

pub use self::location::Location;
pub use self::mt::{MTBuilder, MT};

/// A debugging aid for traces.
/// Calls to this function are recognised by Yorick and a special debug TIR statement is inserted
/// into the trace. Interpreter writers should compile-time guard calls to this so as to only emit
/// the extra bytecodes when explicitly turned on.
#[inline(never)]
#[trace_debug]
pub fn trace_debug(_msg: &'static str) {}
