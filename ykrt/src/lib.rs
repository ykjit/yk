//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]

mod location;
pub(crate) mod mt;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MT};
