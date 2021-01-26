//! Hardware tracing via ykrustc.

use super::{SirTrace, ThreadTracer, ThreadTracerImpl};
use crate::errors::InvalidTraceError;
use hwtracer::backends::TracerBuilder;

pub mod mapper;
use mapper::HWTMapper;

/// Hardware thread tracer.
struct HWTThreadTracer {
    active: bool,
    ttracer: Box<dyn hwtracer::ThreadTracer>
}

impl ThreadTracerImpl for HWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<SirTrace, InvalidTraceError> {
        self.active = false;
        let hwtrace = self.ttracer.stop_tracing().unwrap();
        let mt = HWTMapper::new();
        mt.map_trace(hwtrace)
            .map_err(|_| InvalidTraceError::InternalError)
            .map(|sirtrace| SirTrace::new(sirtrace))
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
