//! The Linux Perf trace collector.

use crate::perf::PerfCollectorConfig;
#[cfg(pt)]
use crate::pt::c_errors::PerfPTCError;
#[cfg(ykpt)]
use crate::pt::ykpt::YkPTBlockIterator;
use crate::{
    errors::{HWTracerError, TemporaryErrorKind},
    Block, ThreadTracer, Trace, Tracer,
};
use libc::{c_void, free, geteuid, malloc, size_t};
use std::{fs::read_to_string, slice, sync::Arc};

#[cfg(pt)]
extern "C" {
    fn hwt_perf_init_collector(
        conf: *const PerfCollectorConfig,
        err: *mut PerfPTCError,
    ) -> *mut c_void;
    fn hwt_perf_start_collector(
        tr_ctx: *mut c_void,
        trace: *mut PerfTrace,
        err: *mut PerfPTCError,
    ) -> bool;
    fn hwt_perf_stop_collector(tr_ctx: *mut c_void, err: *mut PerfPTCError) -> bool;
    fn hwt_perf_free_collector(tr_ctx: *mut c_void, err: *mut PerfPTCError) -> bool;
}

const PERF_PERMS_PATH: &str = "/proc/sys/kernel/perf_event_paranoid";

/// The configuration for a Linux Perf collector.
#[derive(Clone, Debug)]
pub(crate) struct PerfTracer {
    config: PerfCollectorConfig,
}

impl Tracer for PerfTracer {
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, HWTracerError> {
        Ok(Box::new(PerfThreadTracer::new(&self)?))
    }
}

impl PerfTracer {
    pub(crate) fn new(config: PerfCollectorConfig) -> Result<Arc<Self>, HWTracerError>
    where
        Self: Sized,
    {
        // Check for inavlid configuration.
        fn power_of_2(v: size_t) -> bool {
            (v & (v - 1)) == 0
        }
        if !power_of_2(config.data_bufsize) {
            return Err(HWTracerError::ConfigError(
                "data_bufsize must be a positive power of 2".into(),
            ));
        }
        if !power_of_2(config.aux_bufsize) {
            return Err(HWTracerError::ConfigError(
                "aux_bufsize must be a positive power of 2".into(),
            ));
        }

        // Check we have permissions to collect a PT trace using perf.
        //
        // Note that root always has permission.
        //
        // FIXME: We just assume that we are collecting a PT trace.
        // https://github.com/ykjit/hwtracer/issues/100
        if !unsafe { geteuid() } == 0 {
            match read_to_string(PERF_PERMS_PATH) {
                Ok(x) if x.trim() == "-1" => (),
                _ => {
                    return Err(HWTracerError::ConfigError(format!("Tracing not permitted: you must be root or {PERF_PERMS_PATH} must contain -1")));
                }
            }
        }

        Ok(Arc::new(Self { config }))
    }
}

/// A collector that uses the Linux Perf interface to Intel Processor Trace.
pub struct PerfThreadTracer {
    // Opaque C pointer representing the collector context.
    ctx: *mut c_void,
    // The trace currently being collected, or `None`.
    trace: Box<PerfTrace>,
}

impl ThreadTracer for PerfThreadTracer {
    #[cfg(pt)]
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn Trace>, HWTracerError> {
        let mut cerr = PerfPTCError::new();
        let rc = unsafe { hwt_perf_stop_collector(self.ctx, &mut cerr) };
        if !rc {
            return Err(cerr.into());
        }

        let mut cerr = PerfPTCError::new();
        if !unsafe { hwt_perf_free_collector(self.ctx, &mut cerr) } {
            return Err(cerr.into());
        }

        Ok(self.trace)
    }
}

impl PerfThreadTracer {
    #[cfg(pt)]
    fn new(tracer: &PerfTracer) -> Result<Self, HWTracerError> {
        // At the time of writing, we have to use a fresh Perf file descriptor to ensure traces
        // start with a `PSB+` packet sequence. This is required for correct instruction-level and
        // block-level decoding. Therefore we have to re-initialise for each new tracing session.
        let mut cerr = PerfPTCError::new();
        let ctx = unsafe {
            hwt_perf_init_collector(&tracer.config as *const PerfCollectorConfig, &mut cerr)
        };
        if ctx.is_null() {
            return Err(cerr.into());
        }

        // It is essential we box the trace now to stop it from moving. If it were to move, then
        // the reference which we pass to C here would become invalid. The interface to
        // `stop_collector` needs to return a Box<Tracer> anyway, so it's no big deal.
        //
        // Note that the C code will mutate the trace's members directly.
        let mut trace = Box::new(PerfTrace::new(tracer.config.initial_trace_bufsize)?);
        let mut cerr = PerfPTCError::new();
        if !unsafe { hwt_perf_start_collector(ctx, &mut *trace, &mut cerr) } {
            return Err(cerr.into());
        }

        Ok(Self { ctx, trace })
    }
}

/// A wrapper around a manually malloc/free'd buffer for holding an Intel PT trace. We've split
/// this out from PerfTrace so that we can mark just this raw pointer as `unsafe Send`.
#[repr(C)]
#[derive(Debug)]
struct PerfTraceBuf(*mut u8);

/// We need to be able to transfer `PerfTraceBuf`s between threads to allow background
/// compilation. However, `PerfTraceBuf` wraps a raw pointer, which is not `Send`, so nor is
/// `PerfTraceBuf`. As long as we take great care to never: a) give out copies of the pointer to
/// the wider world, or b) dereference the pointer when we shouldn't, then we can manually (and
/// unsafely) mark the struct as being Send.
unsafe impl Send for PerfTrace {}

/// An Intel PT trace, obtained via Linux perf.
#[repr(C)]
#[derive(Debug)]
pub struct PerfTrace {
    /// The trace buffer.
    buf: PerfTraceBuf,
    /// The length of the trace (in bytes).
    len: u64,
    /// `buf`'s allocation size (in bytes), <= `len`.
    capacity: u64,
}

impl PerfTrace {
    /// Makes a new trace, initially allocating the specified number of bytes for the PT trace
    /// packet buffer.
    ///
    /// The allocation is automatically freed by Rust when the struct falls out of scope.
    pub(crate) fn new(capacity: size_t) -> Result<Self, HWTracerError> {
        let buf = unsafe { malloc(capacity) as *mut u8 };
        if buf.is_null() {
            return Err(HWTracerError::Temporary(TemporaryErrorKind::CantAllocate));
        }
        Ok(Self {
            buf: PerfTraceBuf(buf),
            len: 0,
            capacity: capacity as u64,
        })
    }
}

impl Trace for PerfTrace {
    #[cfg(ykpt)]
    fn iter_blocks<'a>(&'a self) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + 'a> {
        let bytes =
            unsafe { slice::from_raw_parts(self.buf.0, usize::try_from(self.len).unwrap()) };
        Box::new(YkPTBlockIterator::new(bytes))
    }

    #[cfg(test)]
    fn bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.buf.0, usize::try_from(self.len).unwrap()) }
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.capacity as usize
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        usize::try_from(self.len).unwrap()
    }
}

impl Drop for PerfTrace {
    fn drop(&mut self) {
        if !self.buf.0.is_null() {
            unsafe { free(self.buf.0 as *mut c_void) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work_loop;

    /// Check that a long trace causes the trace buffer to reallocate.
    #[test]
    // FIXME: Reallocating the trace buffer causes synchronisation problems.
    #[ignore]
    fn relloc_trace_buf() {
        let start_bufsize = 512;
        let config = PerfCollectorConfig {
            initial_trace_bufsize: start_bufsize,
            ..Default::default()
        };
        let tracer = PerfTracer::new(config).unwrap();

        let tt = tracer.start_collector().unwrap();
        let res = work_loop(10000);
        let trace = tt.stop_collector().unwrap();

        println!("res: {}", res); // Stop over-optimisation.
        assert!(trace.capacity() > start_bufsize);
    }

    /// Check that an invalid data buffer size causes an error.
    #[test]
    fn test_config_bad_data_bufsize() {
        let cfg = PerfCollectorConfig {
            data_bufsize: 3,
            ..PerfCollectorConfig::default()
        };
        match PerfTracer::new(cfg) {
            Err(HWTracerError::ConfigError(s))
                if s == "data_bufsize must be a positive power of 2" => {}
            _ => panic!(),
        }
    }

    /// Check that an invalid aux buffer size causes an error.
    #[test]
    fn test_config_bad_aux_bufsize() {
        let cfg = PerfCollectorConfig {
            aux_bufsize: 3,
            ..PerfCollectorConfig::default()
        };
        match PerfTracer::new(cfg) {
            Err(HWTracerError::ConfigError(s))
                if s == "aux_bufsize must be a positive power of 2" => {}
            _ => panic!(),
        }
    }
}
