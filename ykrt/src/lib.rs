#![feature(integer_atomics)]
#![feature(negative_impls)]
#![feature(test)]

pub mod mt;

pub use self::mt::{Location, MT};
