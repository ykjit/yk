//! Hardware tracing via ykrustc.

use super::{IRTrace, ThreadTracer, ThreadTracerImpl, UnmappedTrace};
use crate::errors::InvalidTraceError;
use hwtracer::collect::TraceCollectorBuilder;

pub mod mapper;
use mapper::HWTMapper;

/// Hardware thread tracer.
struct HWTThreadTracer {
    active: bool,
    tcol: Box<dyn hwtracer::collect::ThreadTraceCollector>,
}

impl ThreadTracerImpl for HWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
        self.active = false;
        match self.tcol.stop_collector() {
            Ok(t) => Ok(Box::new(PTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
    }
}

impl Drop for HWTThreadTracer {
    fn drop(&mut self) {
        if self.active {
            self.tcol.stop_collector().unwrap();
        }
    }
}

pub(crate) fn start_tracing() -> ThreadTracer {
    let col = TraceCollectorBuilder::new().build().unwrap();
    let mut tcol = unsafe { col.thread_collector() };
    tcol.start_collector()
        .expect("Failed to start trace collector");
    ThreadTracer {
        t_impl: Box::new(HWTThreadTracer { active: true, tcol }),
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
