//! Hardware tracing via ykrustc.

use super::{SirTrace, ThreadTracer, ThreadTracerImpl};
use crate::{errors::InvalidTraceError, SirLoc};
use hwtracer::backends::TracerBuilder;

pub mod mapper;
use mapper::HWTMapper;

/// A trace collected via hardware tracing.
#[derive(Debug)]
struct HWTSirTrace {
    sirtrace: Vec<SirLoc>
}

impl SirTrace for HWTSirTrace {
    fn raw_len(&self) -> usize {
        self.sirtrace.len()
    }

    fn raw_loc(&self, idx: usize) -> &SirLoc {
        &self.sirtrace[idx]
    }
}

/// Hardware thread tracer.
struct HWTThreadTracer {
    active: bool,
    ttracer: Box<dyn hwtracer::ThreadTracer>
}

impl ThreadTracerImpl for HWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<Box<dyn SirTrace>, InvalidTraceError> {
        self.active = false;
        let hwtrace = self.ttracer.stop_tracing().unwrap();
        let mt = HWTMapper::new();
        mt.map_trace(hwtrace)
            .map_err(|_| InvalidTraceError::InternalError)
            .map(|sirtrace| Box::new(HWTSirTrace { sirtrace }) as Box<dyn SirTrace>)
    }
}

impl Drop for HWTThreadTracer {
    fn drop(&mut self) {
        if self.active {
            self.ttracer.stop_tracing().unwrap();
        }
    }
}

pub(crate) fn start_tracing() -> ThreadTracer {
    let tracer = TracerBuilder::new().build().unwrap();
    let mut ttracer = (*tracer).thread_tracer();
    ttracer.start_tracing().expect("Failed to start tracer.");
    ThreadTracer {
        t_impl: Box::new(HWTThreadTracer {
            active: true,
            ttracer
        })
    }
}

#[cfg(test)]
#[cfg(tracermode = "hw")]
mod tests {
    use crate::{test_helpers, TracingKind};

    const TRACING_KIND: TracingKind = TracingKind::HardwareTracing;

    #[test]
    fn trace() {
        test_helpers::trace(TRACING_KIND);
    }

    #[test]
    fn trace_twice() {
        test_helpers::trace_twice(TRACING_KIND);
    }

    #[test]
    fn trace_concurrent() {
        test_helpers::trace_concurrent(TRACING_KIND);
    }

    #[test]
    #[should_panic]
    fn oob_trace_index() {
        test_helpers::oob_trace_index(TRACING_KIND);
    }

    #[test]
    fn in_bounds_trace_indices() {
        test_helpers::in_bounds_trace_indices(TRACING_KIND);
    }
}
