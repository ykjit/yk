//! Tests that would have lived in the internal workspace, but can't due to the compilation model
//! of yk (the tests perform tracing, which must occur in the unoptimised workspace).

#![feature(test)]

#[cfg(test)]
mod codegen;

#[cfg(test)]
mod helpers;

#[cfg(test)]
mod stopgap;

#[cfg(test)]
mod tir;

#[cfg(test)]
mod tracing;
