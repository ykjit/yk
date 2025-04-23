//! The Linux Perf trace collector.

use crate::perf::PerfCollectorConfig;
#[cfg(pt)]
use crate::pt::c_errors::PerfPTCError;
#[cfg(ykpt)]
use crate::pt::ykpt::YkPTBlockIterator;
use crate::{
    errors::{HWTracerError, TemporaryErrorKind},
    Block, BlockIteratorError, ThreadTracer, Trace, Tracer,
};
use libc::{c_void, free, geteuid, malloc, size_t, PF_R, PF_X, PT_LOAD};
use std::{
    ffi::CString,
    fs::read_to_string,
    sync::{Arc, LazyLock},
};
use ykaddr::obj::{PHDR_MAIN_OBJ, PHDR_OBJECT_CACHE};

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

/// Make an eternal CString from the binary path that we can hand out pointers to without fear of
/// it being dropped.
pub static SELF_BIN_PATH_CSTRING: LazyLock<CString> =
    LazyLock::new(|| CString::new(ykaddr::obj::SELF_BIN_PATH.to_str().unwrap()).unwrap());

/// Determines:
///  - The object to limit tracing to.
///  - The base address and the length of the executable code (within the above object) to limit
///    tracing to.
///
/// It is assumed that there is one contiguous range of executable code that we wish to trace.
#[no_mangle]
pub extern "C" fn get_tracing_extent(obj: *mut *const i8, base_off: *mut usize, len: *mut usize) {
    let mut found = None;
    // Find the main object.
    for o in &*PHDR_OBJECT_CACHE {
        if o.name().to_str().unwrap() == PHDR_MAIN_OBJ.to_str().unwrap() {
            for h in o.phdrs() {
                // Find a header that is loaded and flagged read+exec.
                if h.type_() == PT_LOAD && (h.flags() & PF_R != 0) && (h.flags() & PF_X != 0) {
                    // We expect only be one such entry.
                    assert!(found.is_none());
                    found = Some(h);
                }
            }
        }
    }
    let Some(h) = found else {
        panic!("couldn't find the executable range of {PHDR_MAIN_OBJ:?}");
    };

    unsafe { std::ptr::write(obj, SELF_BIN_PATH_CSTRING.as_ptr()) };
    unsafe { std::ptr::write(base_off, usize::try_from(h.offset()).unwrap()) };
    unsafe { std::ptr::write(len, usize::try_from(h.filesz()).unwrap()) };
}

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
        if !config.data_bufsize.is_power_of_two() {
            return Err(HWTracerError::ConfigError(
                "data_bufsize must be a positive power of 2".into(),
            ));
        }
        if !config.aux_bufsize.is_power_of_two() {
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
#[derive(Debug)]
pub struct PerfThreadTracer {
    // Opaque C pointer representing the collector context.
    ctx: *mut c_void,
    // The trace currently being collected, or `None`.
    trace: Box<PerfTrace>,
}

impl ThreadTracer for PerfThreadTracer {
    #[cfg(pt)]
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn Trace>, HWTracerError> {
        let mut stop_cerr = PerfPTCError::new();
        let stop_rc = unsafe { hwt_perf_stop_collector(self.ctx, &mut stop_cerr) };

        // Even if stopping the collecor fails, we still have to free the collector to ensure that
        // no resources leak.
        let mut free_cerr = PerfPTCError::new();
        let free_rc = unsafe { hwt_perf_free_collector(self.ctx, &mut free_cerr) };

        // Now we decide how to deal with these two fallable operations.
        //
        // For now we a let the `hwt_perf_stop_collector()` error take precedence.
        if !stop_rc {
            Err(stop_cerr.into())
        } else if !free_rc {
            Err(free_cerr.into())
        } else {
            Ok(self.trace)
        }
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
        let mut trace = Box::new(PerfTrace::new(tracer.config.trace_result_size)?);
        let mut cerr = PerfPTCError::new();
        if !unsafe { hwt_perf_start_collector(ctx, &mut *trace, &mut cerr) } {
            return Err(cerr.into());
        }

        Ok(Self { ctx, trace })
    }
}

/// A wrapper around a manually malloc/free'd buffer for holding an Intel PT trace. We've split
/// this out from PerfTrace so that we can mark just this raw pointer as `unsafe Send`.
///
/// FIXME: Marking a pointer `Send` is necessary but not sufficient: we also need some sort of
/// memory barrier to guarantee that all writes from thread A are seen by thread B. That said, x86
/// (and currently this code can only sensibly run on x86) has a strong memory model makes it
/// relatively unlikely that this will be a problem in practise. There is, though, a tiny chance
/// that a super-smart compiler could try and optimise things across threads.
#[repr(C)]
#[derive(Debug)]
#[cfg(target_arch = "x86_64")]
pub(crate) struct PerfTraceBuf(pub(crate) *mut u8);

unsafe impl Send for PerfTraceBuf {}

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
    fn iter_blocks(
        mut self: Box<Self>,
    ) -> Box<dyn Iterator<Item = Result<Block, BlockIteratorError>> + Send> {
        // We hand ownership for self.buf over to `YkPTBlockIterator` so we need to make sure that
        // we don't try and free it.
        let buf = std::mem::replace(&mut self.buf, PerfTraceBuf(std::ptr::null_mut()));
        Box::new(YkPTBlockIterator::new(
            buf,
            usize::try_from(self.len).unwrap(),
        ))
    }

    #[cfg(test)]
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buf.0, usize::try_from(self.len).unwrap()) }
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
        let start_bufsize = 2;
        let config = PerfCollectorConfig {
            data_bufsize: start_bufsize,
            ..Default::default()
        };
        let tracer = PerfTracer::new(config).unwrap();

        let tt = tracer.start_collector().unwrap();
        let res = work_loop(10000);
        let trace = tt.stop_collector().unwrap();

        println!("res: {res}"); // Stop over-optimisation.
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
