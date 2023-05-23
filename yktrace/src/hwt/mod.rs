//! Hardware tracing via ykrustc.

use super::{IRTrace, ThreadTracer, UnmappedTrace};
use crate::errors::InvalidTraceError;
use hwtracer::{
    collect::{default_tracer_for_platform, ThreadTracer as HWThreadTracer, Tracer},
    decode::{TraceDecoderBuilder, TraceDecoderKind},
};

pub mod mapper;
pub use mapper::HWTMapper;

/// Hardware thread tracer.
struct HWTThreadTracer {
    tracer: Box<dyn Tracer>,
    thread_tracer: Option<Box<dyn HWThreadTracer>>,
}

impl ThreadTracer for HWTThreadTracer {
    fn stop_collector(&mut self) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
        match self
            .tracer
            .stop_collector(self.thread_tracer.take().unwrap())
        {
            Ok(t) => Ok(Box::new(PTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
    }
}

impl Drop for HWTThreadTracer {
    fn drop(&mut self) {
        if self.thread_tracer.is_some() {
            self.stop_collector().ok();
        }
    }
}

pub(crate) fn start_tracing() -> Box<dyn ThreadTracer> {
    let tracer = default_tracer_for_platform().unwrap();
    let thread_tracer = Some(tracer.start_collector().unwrap());
    Box::new(HWTThreadTracer {
        tracer,
        thread_tracer,
    })
}

struct PTTrace(Box<dyn hwtracer::Trace>);

impl UnmappedTrace for PTTrace {
    fn map(self: Box<Self>, decoder: TraceDecoderKind) -> Result<IRTrace, InvalidTraceError> {
        let tdec = TraceDecoderBuilder::new().kind(decoder).build().unwrap();
        let mut itr = tdec.iter_blocks(self.0.as_ref());
        let mut mt = HWTMapper::new();

        let mapped = mt
            .map_trace(&mut *itr)
            .map_err(|_| InvalidTraceError::InternalError)?;
        if mapped.is_empty() {
            return Err(InvalidTraceError::EmptyTrace);
        }

        Ok(IRTrace::new(mapped, mt.faddrs()))
    }
}
