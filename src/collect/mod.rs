//! Trace collectors.

use crate::{errors::HWTracerError, Trace};
use core::arch::x86_64::__cpuid_count;
use libc::{size_t, sysconf, _SC_PAGESIZE};
use std::{convert::TryFrom, sync::LazyLock};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[cfg(collector_perf)]
pub(crate) mod perf;
#[cfg(collector_perf)]
pub(crate) use perf::PerfTraceCollector;

const PERF_DFLT_DATA_BUFSIZE: size_t = 64;
const PERF_DFLT_AUX_BUFSIZE: LazyLock<size_t> = LazyLock::new(|| {
    // Allocate enough pages for a 64MiB trace buffer.
    let mb64 = 1024 * 1024 * 64;
    let page_sz = size_t::try_from(unsafe { sysconf(_SC_PAGESIZE) }).unwrap();
    mb64 / page_sz + size_t::from(mb64 % page_sz != 0)
});

const PERF_DFLT_INITIAL_TRACE_BUFSIZE: size_t = 1024 * 1024; // 1MiB

/// The interface offered by all trace collectors.
pub trait TraceCollector: Send + Sync {
    /// Obtain a `ThreadTraceCollector` for the current thread.
    ///
    /// A thread may obtain multiple `ThreadTraceCollector`s but must only collect a trace with one
    /// at a time.
    ///
    /// FIXME: This API needs to be fixed:
    /// https://github.com/ykjit/hwtracer/issues/101
    unsafe fn thread_collector(&self) -> Box<dyn ThreadTraceCollector>;
}

pub trait ThreadTraceCollector {
    /// Start recording a trace.
    ///
    /// Tracing continues until [stop_collector] is called.
    fn start_collector(&mut self) -> Result<(), HWTracerError>;
    /// Turns off the tracer.
    ///
    /// Tracing continues until [stop_collector] is called.
    fn stop_collector(&mut self) -> Result<Box<dyn Trace>, HWTracerError>;
}

/// Kinds of collector that hwtracer supports (in order of "auto-selection preference").
#[derive(Debug, EnumIter)]
pub enum TraceCollectorKind {
    /// The `perf` subsystem, as found on Linux.
    Perf,
}

impl TraceCollectorKind {
    /// Finds a suitable `TraceCollectorKind` for the current hardware/OS.
    fn default_for_platform() -> Option<Self> {
        for kind in TraceCollectorKind::iter() {
            if Self::match_platform(&kind).is_ok() {
                return Some(kind);
            }
        }
        None
    }

    /// Returns `Ok` if the this collector is appropriate for the current platform.
    fn match_platform(&self) -> Result<(), HWTracerError> {
        match self {
            Self::Perf => {
                #[cfg(not(collector_perf))]
                return Err(HWTracerError::CollectorUnavailable(Self::Perf));
                #[cfg(collector_perf)]
                {
                    if !Self::pt_supported() {
                        return Err(HWTracerError::NoHWSupport(
                            "Intel PT not supported by CPU".into(),
                        ));
                    }
                    Ok(())
                }
            }
        }
    }

    /// Checks if the CPU supports Intel Processor Trace.
    fn pt_supported() -> bool {
        let res = unsafe { __cpuid_count(0x7, 0x0) };
        (res.ebx & (1 << 25)) != 0
    }
}

/// Configuration for trace collectors.
#[derive(Debug)]
pub enum TraceCollectorConfig {
    Perf(PerfCollectorConfig),
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

impl TraceCollectorConfig {
    fn kind(&self) -> TraceCollectorKind {
        match self {
            TraceCollectorConfig::Perf { .. } => TraceCollectorKind::Perf,
        }
    }
}

/// A builder interface for instantiating trace collectors.
///
/// # Make a trace collector using the appropriate defaults.
/// ```
/// use hwtracer::collect::TraceCollectorBuilder;
/// TraceCollectorBuilder::new().build().unwrap();
/// ```
///
/// # Make a trace collector that uses Perf for collection with default options.
/// ```
/// use hwtracer::collect::{TraceCollectorBuilder, TraceCollectorKind};
///
/// let res = TraceCollectorBuilder::new().kind(TraceCollectorKind::Perf).build();
/// if let Ok(tracer) = res {
///     // Use the collector.
/// } else {
///     // Platform doesn't support Linux Perf or CPU doesn't support Intel Processor Trace.
/// }
/// ```
///
/// # Make a collector appropriate for the current platform, using custom Perf collector options if
/// the Perf collector was chosen.
/// ```
/// use hwtracer::collect::{TraceCollectorBuilder, TraceCollectorConfig};
/// let mut bldr = TraceCollectorBuilder::new();
/// if let TraceCollectorConfig::Perf(ref mut ppt_config) = bldr.config() {
///     ppt_config.aux_bufsize = 8192;
/// }
/// bldr.build().unwrap();
/// ```
pub struct TraceCollectorBuilder {
    config: TraceCollectorConfig,
}

impl TraceCollectorBuilder {
    /// Create a new `TraceCollectorBuilder` using sensible defaults.
    pub fn new() -> Self {
        let config = match TraceCollectorKind::default_for_platform().unwrap() {
            TraceCollectorKind::Perf => TraceCollectorConfig::Perf(PerfCollectorConfig::default()),
        };
        Self { config }
    }

    /// Select the kind of trace collector.
    pub fn kind(mut self, kind: TraceCollectorKind) -> Self {
        self.config = match kind {
            TraceCollectorKind::Perf => TraceCollectorConfig::Perf(PerfCollectorConfig::default()),
        };
        self
    }

    /// Get a mutable reference to the collector configuraion.
    pub fn config(&mut self) -> &mut TraceCollectorConfig {
        &mut self.config
    }

    /// Build the trace collector
    ///
    /// An error is returned if the requested collector is inappropriate for the platform or not
    /// compiled in to hwtracer.
    pub fn build(self) -> Result<Box<dyn TraceCollector>, HWTracerError> {
        let kind = self.config.kind();
        kind.match_platform()?;
        match self.config {
            TraceCollectorConfig::Perf(_pt_conf) => {
                #[cfg(collector_perf)]
                return Ok(Box::new(PerfTraceCollector::new(_pt_conf)?));
                #[cfg(not(collector_perf))]
                unreachable!();
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::{
        collect::{ThreadTraceCollector, TraceCollector},
        errors::HWTracerError,
        test_helpers::work_loop,
        Trace,
    };
    use std::thread;

    /// Trace a closure that returns a u64.
    pub fn trace_closure<F>(tc: &mut dyn ThreadTraceCollector, f: F) -> Box<dyn Trace>
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
    pub fn basic_collection<T>(mut tracer: T)
    where
        T: ThreadTraceCollector,
    {
        let trace = trace_closure(&mut tracer, || work_loop(500));
        assert_ne!(trace.len(), 0);
    }

    /// Check that repeated usage of the same trace collector works.
    pub fn repeated_collection<T>(mut tracer: T)
    where
        T: ThreadTraceCollector,
    {
        for _ in 0..10 {
            trace_closure(&mut tracer, || work_loop(500));
        }
    }

    /// Check that starting a trace collector twice (without stopping maktracing inbetween) makes
    /// an appropriate error.
    pub fn already_started<T>(mut tc: T)
    where
        T: ThreadTraceCollector,
    {
        tc.start_collector().unwrap();
        match tc.start_collector() {
            Err(HWTracerError::AlreadyCollecting) => (),
            _ => panic!(),
        };
        tc.stop_collector().unwrap();
    }

    /// Check that stopping an unstarted trace collector makes an appropriate error.
    pub fn not_started<T>(mut tc: T)
    where
        T: ThreadTraceCollector,
    {
        match tc.stop_collector() {
            Err(HWTracerError::AlreadyStopped) => (),
            _ => panic!(),
        };
    }

    /// Check that traces can be collected concurrently.
    pub fn concurrent_collection(tc: &dyn TraceCollector) {
        for _ in 0..10 {
            thread::scope(|s| {
                let hndl = s.spawn(|| {
                    let mut thr_c1 = unsafe { tc.thread_collector() };
                    trace_closure(&mut *thr_c1, || work_loop(500));
                });

                let mut thr_c2 = unsafe { tc.thread_collector() };
                trace_closure(&mut *thr_c2, || work_loop(500));
                hndl.join().unwrap();
            });
        }
    }
}
