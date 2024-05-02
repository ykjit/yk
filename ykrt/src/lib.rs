//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]
#![feature(lazy_cell)]
#![feature(naked_functions)]
#![feature(ptr_sub_ptr)]
#![feature(strict_provenance)]
#![allow(clippy::type_complexity)]
#![allow(clippy::new_without_default)]
#![allow(clippy::upper_case_acronyms)]

pub(crate) mod aotsmp;
pub mod compile;
mod location;
mod log;
pub(crate) mod mt;
pub mod promote;
pub(crate) mod thread_intercept;
pub mod trace;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MT};
