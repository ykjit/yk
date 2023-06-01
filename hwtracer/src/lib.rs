#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::upper_case_acronyms)]
#![feature(lazy_cell)]
#![feature(ptr_sub_ptr)]

mod block;
pub use block::Block;
mod c_errors;
pub mod collect;
pub use collect::{default_tracer_for_platform, ThreadTracer, Tracer};
pub mod decode;
pub mod errors;
pub mod llvm_blockmap;

#[cfg(test)]
mod work_loop;
#[cfg(test)]
use work_loop::work_loop;

pub use errors::HWTracerError;
use std::fmt::Debug;

/// Represents a generic trace.
///
/// Each trace decoder has its own concrete implementation.
pub trait Trace: Debug + Send {
    fn bytes(&self) -> &[u8];

    /// Get the capacity of the trace in bytes.
    #[cfg(test)]
    fn capacity(&self) -> usize;

    /// Get the size of the trace in bytes.
    fn len(&self) -> usize;
}
