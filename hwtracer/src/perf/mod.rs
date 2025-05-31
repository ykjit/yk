use libc::size_t;
use std::fs;

pub(crate) mod collect;

/// The default perf data buffer size, measured in pages. Must be a power of 2.
const PERF_DFLT_DATA_BUFSIZE: size_t = 16; // 64KiB (w/4096 byte pages).
/// The default perf aux buffer size, measured in pages. Must be a power of 2.
const PERF_DFLT_AUX_BUFSIZE: size_t = 1024; // 4MiB (w/4096 byte pages)
/// The size of the final memory returned, measured in bytes. There are no constraints on size or
/// alignment but it needs to big enough to contain the AUX buffers.
const TRACE_RESULT_SIZE: size_t = 4 * 1024 * 1024; // 4MiB

/// Configures the Perf collector.
///
// Must stay in sync with the C struct `hwt_perf_collector_config`.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct PerfCollectorConfig {
    /// Data buffer size, in pages. Must be a power of 2.
    pub data_bufsize: size_t,
    /// AUX buffer size, in pages. Must be a power of 2.
    pub aux_bufsize: size_t,
    /// The size of the final memory returned, measured in bytes. There are no constraints on size
    /// or alignment but it needs to big enough to contain the AUX buffers.
    pub trace_result_size: size_t,
    /// Whether or not to use PT IP filtering.
    pub use_pt_filtering: bool,
}

impl Default for PerfCollectorConfig {
    fn default() -> Self {
        // PT IP filtering doesn't work inside docker.
        // https://lore.kernel.org/linux-perf-users/aBCwoq7w8ohBRQCh@fremen.lan/
        let use_pt_filtering = !fs::exists("/.dockerenv").unwrap();
        Self {
            data_bufsize: PERF_DFLT_DATA_BUFSIZE,
            aux_bufsize: PERF_DFLT_AUX_BUFSIZE,
            trace_result_size: TRACE_RESULT_SIZE,
            use_pt_filtering,
        }
    }
}
