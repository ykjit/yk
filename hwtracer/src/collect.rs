//! Trace collectors.

use crate::{errors::HWTracerError, Trace};
use std::sync::Arc;

/// A tracer is an object which can start / stop collecting traces. It may have its own
/// configuration, but that is dependent on the particular tracing backend.
pub trait Tracer: Send + Sync {
    /// Start collecting a trace of the current thread.
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, HWTracerError>;
}

/// Represents a thread which is currently tracing.
pub trait ThreadTracer {
    /// Stop collecting a trace of the current thread.
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn Trace>, HWTracerError>;
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

#[cfg(test)]
mod test {
    use crate::{
        collect::{default_tracer, Tracer},
        trace_closure, work_loop,
    };
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
