use libc::{size_t, sysconf, _SC_PAGESIZE};
use std::alloc::Layout;

pub(crate) mod collect;

const PERF_DFLT_DATA_BUFSIZE: size_t = 64; // Pages
const PERF_DFLT_AUX_BUFSIZE: size_t = 64 * 1024 * 1024; // MiB
const PERF_DFLT_INITIAL_TRACE_BUFSIZE: size_t = 1024 * 1024; // MiB

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
        // The `as` is safe because `long` on x86 Linux is at most 8 bytes.
        let pagesize = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        // aux_bufsize is given in terms of pages, but we don't know big a page is until runtime,
        // so we convert the MiB constant into number of pages.
        let aux_bufsize = Layout::from_size_align(PERF_DFLT_AUX_BUFSIZE, pagesize)
            .unwrap()
            .pad_to_align()
            .size()
            / pagesize;
        Self {
            data_bufsize: PERF_DFLT_DATA_BUFSIZE,
            aux_bufsize,
            initial_trace_bufsize: PERF_DFLT_INITIAL_TRACE_BUFSIZE,
        }
    }
}
