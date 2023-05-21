//! Hardware tracing via ykrustc.

use super::{IRTrace, ThreadTracer, ThreadTracerImpl, UnmappedTrace};
use crate::errors::InvalidTraceError;
use hwtracer::{
    collect::{default_tracer_for_platform, Tracer},
    decode::{TraceDecoderBuilder, TraceDecoderKind},
};
use std::sync::LazyLock;

pub mod mapper;
pub use mapper::HWTMapper;

static TRACE_COLLECTOR: LazyLock<Box<dyn Tracer>> = LazyLock::new(|| {
    // FIXME: This just makes a default trace collector. We should allow configuration somehow.
    default_tracer_for_platform().unwrap()
});

/// Hardware thread tracer.
struct HWTThreadTracer {
    active: bool,
}

impl ThreadTracerImpl for HWTThreadTracer {
    fn stop_tracing(&mut self) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
        self.active = false;
        match TRACE_COLLECTOR.stop_collector() {
            Ok(t) => Ok(Box::new(PTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
    }
}

impl Drop for HWTThreadTracer {
    fn drop(&mut self) {
        if self.active {
            TRACE_COLLECTOR.stop_collector().unwrap();
        }
    }
}

pub(crate) fn start_tracing() -> ThreadTracer {
    TRACE_COLLECTOR
        .start_collector()
        .expect("Failed to start trace collector");
    ThreadTracer {
        t_impl: Box::new(HWTThreadTracer { active: true }),
    }
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
