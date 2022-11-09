#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::new_without_default)]
#![feature(once_cell)]
#![feature(ptr_sub_ptr)]

mod block;
pub use block::Block;
mod c_errors;
pub mod collect;
pub mod decode;
pub mod errors;
pub mod llvm_blockmap;

pub use errors::HWTracerError;
use std::fmt::Debug;
#[cfg(test)]
use std::fs::File;

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

    /// Dump the trace to the specified filename.
    ///
    /// The exact format varies depending on what kind of trace it is.
    #[cfg(test)]
    fn to_file(&self, file: &mut File);
}

#[cfg(test)]
mod test_helpers {
    use std::time::SystemTime;

    /// A loop that does some work that we can use to build a trace.
    #[inline(never)]
    pub fn work_loop(iters: u64) -> u64 {
        let mut res = 0;
        for _ in 0..iters {
            // Computation which stops the compiler from eliminating the loop.
            res += SystemTime::now().elapsed().unwrap().subsec_nanos() as u64;
        }
        res
    }
}
