//! Trace collectors.

use crate::{errors::HWTracerError, Trace};
use core::arch::x86_64::__cpuid_count;
use libc::{size_t, sysconf, _SC_PAGESIZE};
use std::{cell::RefCell, convert::TryFrom, sync::LazyLock};

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

thread_local! {
    /// When `Some` holds the `ThreadTraceCollector` that is collecting a trace of the current
    /// thread.
    static THREAD_TRACE_COLLECTOR: RefCell<Option<Box<dyn ThreadTracer>>> = RefCell::new(None);
}

pub trait Tracer: Send + Sync {
    /// Start collecting a trace of the current thread.
    fn start_collector(&self) -> Result<(), HWTracerError> {
        THREAD_TRACE_COLLECTOR.with(|inner| {
            let mut inner = inner.borrow_mut();
            if inner.is_some() {
                Err(HWTracerError::AlreadyCollecting)
            } else {
                let thr_col = self.start()?;
                *inner = Some(thr_col);
                Ok(())
            }
        })
    }

    /// Stop collecting a trace of the current thread.
    fn stop_collector(&self) -> Result<Box<dyn Trace>, HWTracerError> {
        THREAD_TRACE_COLLECTOR.with(|inner| {
            if let Some(thr_col) = inner.take() {
                thr_col.stop()
            } else {
                Err(HWTracerError::AlreadyStopped)
            }
        })
    }

    fn start(&self) -> Result<Box<dyn ThreadTracer>, HWTracerError>;
}

pub trait ThreadTracer {
    fn stop(self: Box<Self>) -> Result<Box<dyn Trace>, HWTracerError>;
}

pub fn default_tracer_for_platform() -> Result<Box<dyn Tracer>, HWTracerError> {
    if pt_supported() {
        Ok(Box::new(PerfTracer::new(PerfCollectorConfig::default())?))
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
    use crate::{collect::Tracer, errors::HWTracerError, work_loop, Trace};
    use std::thread;

    /// Trace a closure that returns a u64.
    pub fn trace_closure<F>(tc: &Box<dyn Tracer>, f: F) -> Box<dyn Trace>
    where
        F: FnOnce() -> u64,
    {
        tc.start_collector().unwrap();
        let res = f();
        let trace = tc.stop_collector().unwrap();
        println!("traced closure with result: {}", res); // To avoid over-optimisation.
        trace
    }

    /// Check that starting and stopping a trace collector works.
    pub fn basic_collection(tc: Box<dyn Tracer>) {
        let trace = trace_closure(&tc, || work_loop(500));
        assert_ne!(trace.len(), 0);
    }

    /// Check that repeated usage of the same trace collector works.
    pub fn repeated_collection(tc: Box<dyn Tracer>) {
        for _ in 0..10 {
            trace_closure(&tc, || work_loop(500));
        }
    }

    /// Check that repeated collection using different collectors works.
    pub fn repeated_collection_different_collectors(tcs: [Box<dyn Tracer>; 10]) {
        for i in 0..10 {
            trace_closure(&tcs[i], || work_loop(500));
        }
    }

    /// Check that starting a trace collector twice (without stopping maktracing inbetween) makes
    /// an appropriate error.
    pub fn already_started(tc: Box<dyn Tracer>) {
        tc.start_collector().unwrap();
        match tc.start_collector() {
            Err(HWTracerError::AlreadyCollecting) => (),
            _ => panic!(),
        };
        tc.stop_collector().unwrap();
    }

    /// Check that an attempt to trace the same thread using different collectors fails.
    pub fn already_started_different_collectors(tc1: Box<dyn Tracer>, tc2: Box<dyn Tracer>) {
        tc1.start_collector().unwrap();
        match tc2.start_collector() {
            Err(HWTracerError::AlreadyCollecting) => (),
            _ => panic!(),
        };
        tc1.stop_collector().unwrap();
    }

    /// Check that stopping an unstarted trace collector makes an appropriate error.
    pub fn not_started(tc: Box<dyn Tracer>) {
        match tc.stop_collector() {
            Err(HWTracerError::AlreadyStopped) => (),
            _ => panic!(),
        };
    }

    /// Check that traces can be collected concurrently.
    pub fn concurrent_collection(tc: Box<dyn Tracer>) {
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
