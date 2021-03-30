//! Tests that would have lived in the untraced workspace, but can't due to the compilation model
//! of yk (the tests perform hardware tracing, which must occur in unoptimised code).

#![feature(test)]

#[cfg(test)]
mod codegen;

#[cfg(test)]
mod helpers;

#[cfg(test)]
mod stopgap;

#[cfg(test)]
mod symbols;

#[cfg(test)]
mod tir;

#[cfg(test)]
mod tracing;
