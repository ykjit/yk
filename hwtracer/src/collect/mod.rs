//! Trace collectors.

use crate::{errors::HWTracerError, Trace};
use core::arch::x86_64::__cpuid_count;
use libc::{size_t, sysconf, _SC_PAGESIZE};
use std::{
    convert::TryFrom,
    sync::{Arc, LazyLock},
};

#[cfg(collector_perf)]
pub(crate) mod perf;
#[cfg(all(collector_perf, feature = "yk_testing"))]
pub use perf::PerfTrace;
#[cfg(collector_perf)]
pub(crate) use perf::PerfTracer;

const PERF_DFLT_DATA_BUFSIZE: size_t = 64;
static PERF_DFLT_AUX_BUFSIZE: LazyLock<size_t> = LazyLock::new(|| {
    // Allocate enough pages for a 64MiB trace buffer.
    let mb64 = 1024 * 1024 * 64;
    let page_sz = size_t::try_from(unsafe { sysconf(_SC_PAGESIZE) }).unwrap();
    mb64 / page_sz + size_t::from(mb64 % page_sz != 0)
});

const PERF_DFLT_INITIAL_TRACE_BUFSIZE: size_t = 1024 * 1024; // 1MiB

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
    if pt_supported() {
        Ok(PerfTracer::new(PerfCollectorConfig::default())?)
    } else {
        todo!();
    }
}

/// Checks if the CPU supports Intel Processor Trace.
fn pt_supported() -> bool {
    let res = unsafe { __cpuid_count(0x7, 0x0) };
    (res.ebx & (1 << 25)) != 0
}

/// Configures the Perf collector.
///
// Must stay in sync with the C code.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct PerfCollectorConfig {
    /// Data buffer size, in pages. Must be a power of 2.
    pub data_bufsize: size_t,
    /// AUX buffer size, in pages. Must be a power of 2.
    pub aux_bufsize: size_t,
    /// The initial trace storage buffer size (in bytes) of new traces.
    pub initial_trace_bufsize: size_t,
}

impl Default for PerfCollectorConfig {
    fn default() -> Self {
        Self {
            data_bufsize: PERF_DFLT_DATA_BUFSIZE,
            aux_bufsize: *PERF_DFLT_AUX_BUFSIZE,
            initial_trace_bufsize: PERF_DFLT_INITIAL_TRACE_BUFSIZE,
        }
    }
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
