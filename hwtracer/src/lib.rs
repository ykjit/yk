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
#[cfg(collector_perf)]
mod perf;

pub use errors::HWTracerError;
use std::fmt::Debug;
#[cfg(test)]
use std::time::SystemTime;

/// Represents a generic trace.
///
/// Each trace decoder has its own concrete implementation.
pub trait Trace: Debug + Send {
    /// Iterate over the blocks of the trace.
    fn iter_blocks<'a>(&'a self) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + 'a>;

    #[cfg(test)]
    fn bytes(&self) -> &[u8];

    /// Get the capacity of the trace in bytes.
    #[cfg(test)]
    fn capacity(&self) -> usize;

    /// Get the size of the trace in bytes.
    #[cfg(test)]
    fn len(&self) -> usize;
}

/// A loop that does some work that we can use to build a trace.
#[cfg(test)]
#[inline(never)]
pub fn work_loop(iters: u64) -> u64 {
    let mut res = 0;
    for _ in 0..iters {
        // Computation which stops the compiler from eliminating the loop.
        res += SystemTime::now().elapsed().unwrap().subsec_nanos() as u64;
    }
    res
}
