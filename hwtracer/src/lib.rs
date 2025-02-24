#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::new_without_default)]
#![allow(clippy::upper_case_acronyms)]

mod block;
pub use block::Block;
pub mod errors;
pub mod llvm_blockmap;
#[cfg(linux_perf)]
mod perf;
#[cfg(pt)]
mod pt;

pub use errors::{HWTracerError, TemporaryErrorKind};
#[cfg(test)]
use std::time::SystemTime;
use std::{fmt::Debug, sync::Arc};
use thiserror::Error;

/// A builder for [Tracer]s. By default, will attempt to use the most appropriate [Tracer] for your
/// platform/configuration. This can be overridden with [TracerBuilder::tracer_kind] and
/// [TracerKind].
pub struct TracerBuilder {
    tracer_kind: Option<TracerKind>,
}

impl TracerBuilder {
    /// Create a new [TracerBuilder] with default settings. This will attempt to use the most
    /// appropriate [Tracer] for your platform/configuration. If no suitable [Tracer] can be found,
    /// [TracerKind::None] will be set as the default.
    pub fn new() -> Self {
        #[cfg(all(linux_perf, pt))]
        {
            if crate::pt::pt_supported() {
                return TracerBuilder {
                    tracer_kind: Some(TracerKind::PT(perf::PerfCollectorConfig::default())),
                };
            }
        }

        TracerBuilder { tracer_kind: None }
    }

    /// Change the [TracerKind] of this [TracerBuilder].
    pub fn tracer_kind(mut self, tracer_kind: TracerKind) -> Self {
        self.tracer_kind = Some(tracer_kind);
        self
    }

    /// Build this [TracerBuild] and produce a [Tracer] as output.
    pub fn build(self) -> Result<Arc<dyn Tracer>, HWTracerError> {
        match self.tracer_kind {
            #[cfg(all(linux_perf, pt))]
            Some(TracerKind::PT(config)) => {
                if !crate::pt::pt_supported() {
                    Err(HWTracerError::ConfigError("CPU doesn't support the Processor Trace (PT) feature".into()))
                } else {
                    Ok(crate::perf::collect::PerfTracer::new(config)?)
                }
            }
            None => Err(HWTracerError::ConfigError("No tracer specified: that probably means that no tracers are supported on this platform/configuration".into()))
        }
    }
}

/// The kind of [Tracer] to be built by [TracerBuilder].
pub enum TracerKind {
    // If you add a new variant, don't forget to update `all_collectors` in the `test` mod later in
    // this file.
    #[cfg(all(linux_perf, pt))]
    /// An IntelPT tracer. Note that this currently uses the ykpt decoder.
    PT(perf::PerfCollectorConfig),
}

/// A tracer is an object which can start / stop collecting traces. It may have its own
/// configuration, but that is dependent on the particular tracing backend.
pub trait Tracer: Send + Sync {
    /// Start collecting a trace of the current thread.
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, HWTracerError>;
}

/// Represents a thread which is currently tracing.
pub trait ThreadTracer: Debug {
    /// Stop collecting a trace of the current thread.
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn Trace>, HWTracerError>;
}

/// Represents a generic trace.
///
/// Each trace decoder has its own concrete implementation.
pub trait Trace: Debug + Send {
    /// Iterate over the blocks of the trace.
    fn iter_blocks(
        self: Box<Self>,
    ) -> Box<dyn Iterator<Item = Result<Block, BlockIteratorError>> + Send>;

    #[cfg(test)]
    fn bytes(&self) -> &[u8];

    /// Get the capacity of the trace in bytes.
    #[cfg(test)]
    fn capacity(&self) -> usize;

    /// Get the size of the trace in bytes.
    #[cfg(test)]
    fn len(&self) -> usize;
}

#[derive(Debug, Error)]
pub enum BlockIteratorError {
    #[cfg(ykpt)]
    #[error("dladdr() cannot map vaddr")]
    NoSuchVAddr,
    #[error("HWTracerError: {0}")]
    HWTracerError(HWTracerError),
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
    use crate::{trace_closure, work_loop, Tracer, TracerBuilder, TracerKind};
    use std::{sync::Arc, thread};

    fn all_collectors() -> Vec<Arc<dyn Tracer>> {
        let mut kinds = vec![];

        #[cfg(all(linux_perf, pt))]
        if !crate::pt::pt_supported() {
            kinds.push(TracerKind::PT(crate::perf::PerfCollectorConfig::default()))
        }

        kinds
            .into_iter()
            .map(|k| TracerBuilder::new().tracer_kind(k).build().unwrap())
            .collect::<Vec<_>>()
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
