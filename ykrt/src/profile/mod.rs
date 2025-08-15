//! Profiler support for JITted code.
//!
//! This module specifies the platform abstraction for trace profiling support.

use crate::compile::CompiledTrace;
use std::{error::Error, sync::Arc};

#[cfg(target_os = "linux")]
mod linux_perf;

pub(crate) trait PlatformTraceProfiler: Send + Sync {
    /// Register newly JITted trace with the platform's profiler.
    fn register_ctr(&self, ctr: &Arc<dyn CompiledTrace>) -> Result<(), Box<dyn Error>>;
}

/// Profiler support that does nothing.
///
/// Used when the current platform has no profiler support implemented, or when the user has not
/// turned on profiling for this run.
struct NullProfiler {}

impl PlatformTraceProfiler for NullProfiler {
    fn register_ctr(&self, _ctr: &Arc<dyn CompiledTrace>) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

pub(crate) fn profiler_for_current_platform() -> Arc<dyn PlatformTraceProfiler> {
    if matches!(std::env::var("YKD_TPROF"), Ok(v) if v == "1") && cfg!(target_os = "linux") {
        Arc::new(linux_perf::LinuxPerf::new())
    } else {
        Arc::new(NullProfiler {})
    }
}
