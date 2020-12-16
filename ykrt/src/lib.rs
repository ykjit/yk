#![feature(negative_impls)]
#![cfg_attr(test, feature(test))]

pub mod mt;

pub use self::mt::{Location, MT};
