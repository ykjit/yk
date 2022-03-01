//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]
#![cfg_attr(test, feature(bench_black_box))]
#![feature(once_cell)]
#![allow(clippy::type_complexity)]
#![allow(clippy::new_without_default)]

use std::{env, lazy::SyncLazy};

mod location;
pub(crate) mod mt;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MT};

#[cfg(feature = "yk_jitstate_debug")]
static JITSTATE_DEBUG: SyncLazy<bool> = SyncLazy::new(|| env::var("YKD_PRINT_JITSTATE").is_ok());

/// Print select JIT events to stderr for testing/debugging purposes.
#[cfg(feature = "yk_jitstate_debug")]
pub fn print_jit_state(state: &str) {
    if *JITSTATE_DEBUG {
        eprintln!("jit-state: {}", state);
    }
}
