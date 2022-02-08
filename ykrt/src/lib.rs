//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]
#![cfg_attr(test, feature(bench_black_box))]
#![feature(once_cell)]

mod location;
pub(crate) mod mt;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MT};
