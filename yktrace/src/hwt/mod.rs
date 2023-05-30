//! Hardware tracing via ykrustc.

use super::{IRTrace, ThreadTracer, Tracer, UnmappedTrace};
use crate::errors::InvalidTraceError;
use hwtracer::decode::TraceDecoderBuilder;
use std::{error::Error, sync::Arc};

pub mod mapper;
pub use mapper::HWTMapper;

pub struct HWTracer {
    backend: Box<dyn hwtracer::Tracer>,
}

impl super::Tracer for HWTracer {
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, Box<dyn Error>> {
        Ok(Box::new(HWTThreadTracer {
            tracer: Arc::clone(&self),
            thread_tracer: Some(self.backend.start_collector()?),
        }))
    }
}

impl HWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(HWTracer {
            backend: hwtracer::default_tracer_for_platform()?,
        })
    }
}

/// Hardware thread tracer.
struct HWTThreadTracer {
    tracer: Arc<HWTracer>,
    thread_tracer: Option<Box<dyn hwtracer::ThreadTracer>>,
}

impl ThreadTracer for HWTThreadTracer {
    fn stop_collector(mut self: Box<Self>) -> Result<Box<dyn UnmappedTrace>, InvalidTraceError> {
        match self
            .tracer
            .backend
            .stop_collector(self.thread_tracer.take().unwrap())
        {
            Ok(t) => Ok(Box::new(PTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
    }
}

struct PTTrace(Box<dyn hwtracer::Trace>);

impl UnmappedTrace for PTTrace {
    fn map(self: Box<Self>, _tracer: Arc<dyn Tracer>) -> Result<IRTrace, InvalidTraceError> {
        let tdec = TraceDecoderBuilder::new().build().unwrap();
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
