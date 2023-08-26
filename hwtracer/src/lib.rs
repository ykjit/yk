#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::upper_case_acronyms)]
#![feature(lazy_cell)]
#![feature(ptr_sub_ptr)]

mod block;
pub use block::Block;
pub mod errors;
pub mod llvm_blockmap;
#[cfg(collector_perf)]
mod perf;
#[cfg(target_arch = "x86_64")]
mod pt;

pub use errors::HWTracerError;
#[cfg(test)]
use std::time::SystemTime;
use std::{fmt::Debug, sync::Arc};

/// A tracer is an object which can start / stop collecting traces. It may have its own
/// configuration, but that is dependent on the particular tracing backend.
pub trait Tracer: Send + Sync {
    /// Start collecting a trace of the current thread.
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, HWTracerError>;
}

/// Return the default tracer for this platform and configuration.
pub fn default_tracer() -> Result<Arc<dyn Tracer>, HWTracerError> {
    #[cfg(all(collector_perf, target_arch = "x86_64"))]
    {
        if crate::pt::pt_supported() {
            return Ok(crate::perf::collect::PerfTracer::new(
                crate::perf::PerfCollectorConfig::default(),
            )?);
        }
        return Err(HWTracerError::NoHWSupport(
            "CPU doesn't support the Processor Trace (PT) feature".to_owned(),
        ));
    }

    #[allow(unreachable_code)]
    Err(HWTracerError::Custom(
        "No tracer supported on this platform".into(),
    ))
}

/// Represents a thread which is currently tracing.
pub trait ThreadTracer {
    /// Stop collecting a trace of the current thread.
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn Trace>, HWTracerError>;
}

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

#[cfg(test)]
mod test {
    use crate::{default_tracer, trace_closure, work_loop, Tracer};
    use std::{sync::Arc, thread};

    fn all_collectors() -> Vec<Arc<dyn Tracer>> {
        // So far we only support Perf + PT...
        vec![default_tracer().unwrap()]
    }

    #[test]
    fn basic_collection() {
        for c in all_collectors() {
            let trace = trace_closure(&c, || work_loop(500));
            assert_ne!(trace.len(), 0);
        }
    }

    #[test]
    pub fn repeated_collection() {
        for c in all_collectors() {
            for _ in 0..10 {
                let trace = trace_closure(&c, || work_loop(500));
                assert_ne!(trace.len(), 0);
            }
        }
    }

    #[test]
    pub fn repeated_collection_different_collectors() {
        for _ in 0..10 {
            for c in all_collectors() {
                let trace = trace_closure(&c, || work_loop(500));
                assert_ne!(trace.len(), 0);
            }
        }
    }

    #[test]
    fn concurrent_collection() {
        for c in all_collectors() {
            thread::scope(|s| {
                let hndl = s.spawn(|| {
                    trace_closure(&c, || work_loop(500));
                });

                trace_closure(&c, || work_loop(500));
                hndl.join().unwrap();
            });
        }
    }
}
