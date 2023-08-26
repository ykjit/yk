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
pub mod errors;
pub mod llvm_blockmap;
#[cfg(collector_perf)]
mod perf;
#[cfg(target_arch = "x86_64")]
mod pt;

pub use errors::HWTracerError;
use std::fmt::Debug;
#[cfg(test)]
use std::{sync::Arc, time::SystemTime};

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
pub(crate) fn work_loop(iters: u64) -> u64 {
    let mut res = 0;
    for _ in 0..iters {
        // Computation which stops the compiler from eliminating the loop.
        res += SystemTime::now().elapsed().unwrap().subsec_nanos() as u64;
    }
    res
}

/// Trace a closure that returns a u64.
#[cfg(test)]
pub(crate) fn trace_closure<F>(tc: &Arc<dyn Tracer>, f: F) -> Box<dyn Trace>
where
    F: FnOnce() -> u64,
{
    let tt = Arc::clone(tc).start_collector().unwrap();
    let res = f();
    let trace = tt.stop_collector().unwrap();
    println!("traced closure with result: {}", res); // To avoid over-optimisation.
    trace
}
