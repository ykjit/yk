//! Hardware tracing via ykrustc.

use super::{errors::InvalidTraceError, MappedTrace, RawTrace, ThreadTracer, Tracer};
use hwtracer::decode::default_decoder;
use std::{error::Error, sync::Arc};

pub mod mapper;
pub use mapper::HWTMapper;
mod testing;

pub struct HWTracer {
    backend: Arc<dyn hwtracer::Tracer>,
}

impl super::Tracer for HWTracer {
    fn start_collector(self: Arc<Self>) -> Result<Box<dyn ThreadTracer>, Box<dyn Error>> {
        Ok(Box::new(HWTThreadTracer {
            thread_tracer: Arc::clone(&self.backend).start_collector()?,
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
    thread_tracer: Box<dyn hwtracer::ThreadTracer>,
}

impl ThreadTracer for HWTThreadTracer {
    fn stop_collector(self: Box<Self>) -> Result<Box<dyn RawTrace>, InvalidTraceError> {
        match self.thread_tracer.stop_collector() {
            Ok(t) => Ok(Box::new(PTTrace(t))),
            Err(e) => todo!("{e:?}"),
        }
    }
}

struct PTTrace(Box<dyn hwtracer::Trace>);

impl RawTrace for PTTrace {
    fn map(self: Box<Self>) -> Result<MappedTrace, InvalidTraceError> {
        let tdec = default_decoder().map_err(|_| InvalidTraceError::InternalError)?;
        let mut itr = tdec.iter_blocks(self.0.as_ref());
        let mut mt = HWTMapper::new();

        let mapped = mt
            .map_trace(&mut *itr)
            .map_err(|_| InvalidTraceError::InternalError)?;
        if mapped.is_empty() {
            return Err(InvalidTraceError::EmptyTrace);
        }

        Ok(MappedTrace::new(mapped, mt.faddrs()))
    }
}
