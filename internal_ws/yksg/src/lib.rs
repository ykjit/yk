//! The stopgap interpreter.
//!
//! After a guard failure, the StopgapInterpreter takes over. It interprets SIR to execute the
//! program from the guard failure until execution arrives back to the control point, at which
//! point the normal interpreter can continue.
//!
//! In other systems, this process is sometimes called "blackholing".
//!
//! Tests for this module are in ../tests/src/stopgap/.

mod interp;

pub use interp::{FrameInfo, LocalMem, StopgapInterpreter};
