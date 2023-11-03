//! Hardware tracing via hwtracer.

use super::{errors::InvalidTraceError, MappedTrace, RawTrace, TraceCollector};
use std::{error::Error, sync::Arc};

pub(crate) mod mapper;
pub(crate) use mapper::HWTMapper;
mod testing;

pub(crate) struct HWTracer {
    backend: Arc<dyn hwtracer::Tracer>,
}

impl super::Tracer for HWTracer {
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn TraceCollector>, Box<dyn Error>> {
        Ok(Box::new(HWTTraceCollector {
            thread_tracer: Arc::clone(&self.backend).start_collector()?,
        }))
    }
}

impl HWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(HWTracer {
            backend: hwtracer::TracerBuilder::new().build()?,
        })
    }
}

/// Hardware thread tracer.
struct HWTTraceCollector {
    thread_tracer: Box<dyn hwtracer::ThreadTracer>,
}

impl TraceCollector for HWTTraceCollector {
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn RawTrace>, InvalidTraceError> {
        match self.thread_tracer.stop_collector() {
            Ok(t) => Ok(Box::new(HWTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
    }
}

struct HWTTrace(Box<dyn hwtracer::Trace>);

impl RawTrace for HWTTrace {
    fn map(self: Box<Self>) -> Result<MappedTrace, InvalidTraceError> {
        let mut mt = HWTMapper::new();

        let mapped = mt
            .map_trace(self.0)
            .map_err(|_| InvalidTraceError::InternalError)?;
        if mapped.is_empty() {
            return Err(InvalidTraceError::EmptyTrace);
        }

        Ok(MappedTrace::new(mapped, mt.faddrs()))
    }
}
