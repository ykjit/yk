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

pub fn default_tracer_for_platform() -> Result<Arc<dyn Tracer>, HWTracerError> {
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
pub(crate) mod test_helpers {
    use crate::{collect::Tracer, work_loop, Trace};
    use std::{sync::Arc, thread};

    /// Trace a closure that returns a u64.
    pub fn trace_closure<F>(tc: &Arc<dyn Tracer>, f: F) -> Box<dyn Trace>
    where
        F: FnOnce() -> u64,
    {
        let tt = Arc::clone(tc).start_collector().unwrap();
        let res = f();
        let trace = tt.stop_collector().unwrap();
        println!("traced closure with result: {}", res); // To avoid over-optimisation.
        trace
    }

    /// Check that starting and stopping a trace collector works.
    pub fn basic_collection(tc: Arc<dyn Tracer>) {
        let trace = trace_closure(&tc, || work_loop(500));
        assert_ne!(trace.len(), 0);
    }

    /// Check that repeated usage of the same trace collector works.
    pub fn repeated_collection(tc: Arc<dyn Tracer>) {
        for _ in 0..10 {
            trace_closure(&tc, || work_loop(500));
        }
    }

    /// Check that repeated collection using different collectors works.
    pub fn repeated_collection_different_collectors(tcs: [Arc<dyn Tracer>; 10]) {
        for t in tcs {
            trace_closure(&t, || work_loop(500));
        }
    }

    /// Check that traces can be collected concurrently.
    pub fn concurrent_collection(tc: Arc<dyn Tracer>) {
        for _ in 0..10 {
            thread::scope(|s| {
                let hndl = s.spawn(|| {
                    trace_closure(&tc, || work_loop(500));
                });

                trace_closure(&tc, || work_loop(500));
                hndl.join().unwrap();
            });
        }
    }
}
