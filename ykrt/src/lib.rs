//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]
#![feature(lazy_cell)]
#![feature(naked_functions)]
#![allow(clippy::type_complexity)]
#![allow(clippy::new_without_default)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::upper_case_acronyms)]

pub mod compile;
mod deopt;
mod frame;
mod location;
pub(crate) mod mt;
pub mod promote;
pub(crate) mod thread_intercept;
pub mod trace;
mod ykstats;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MT};

#[cfg(feature = "yk_jitstate_debug")]
use std::{env, sync::LazyLock};

#[cfg(feature = "yk_jitstate_debug")]
static JITSTATE_DEBUG: LazyLock<bool> = LazyLock::new(|| env::var("YKD_PRINT_JITSTATE").is_ok());

/// Print select JIT events to stderr for testing/debugging purposes.
#[cfg(feature = "yk_jitstate_debug")]
pub fn print_jit_state(state: &str) {
    if *JITSTATE_DEBUG {
        eprintln!("jit-state: {}", state);
    }
}

#[cfg(tracer_swt)]
pub fn trace_basicblock(function_index: usize, block_index: usize) {
    if mt::is_tracing() {
        trace::trace_basicblock(function_index, block_index)
    }
}
