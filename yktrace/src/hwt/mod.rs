//! Hardware tracing via ykrustc.

use super::{IRTrace, ThreadTracer, ThreadTracerImpl, UnmappedTrace};
use crate::errors::InvalidTraceError;
use hwtracer::backends::TracerBuilder;

pub mod mapper;
use mapper::HWTMapper;

/// Hardware thread tracer.
struct HWTThreadTracer {
    active: bool,
    ttracer: Box<dyn hwtracer::ThreadTracer>,
}

impl ThreadTracerImpl for HWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
        self.active = false;
        match self.ttracer.stop_tracing() {
            Ok(t) => Ok(Box::new(PTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
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
            ttracer,
        }),
    }
}

struct PTTrace(Box<dyn hwtracer::Trace>);

impl UnmappedTrace for PTTrace {
    fn map(self: Box<Self>) -> Result<IRTrace, InvalidTraceError> {
        let mt = HWTMapper::new();
        let mapped = mt
            .map_trace(self.0)
            .map_err(|_| InvalidTraceError::InternalError)
            .map(|(b, f)| IRTrace::new(b, f));

        if let Ok(x) = &mapped {
            if x.len() == 0 {
                return Err(InvalidTraceError::EmptyTrace);
            }
        }
        mapped
    }
}
