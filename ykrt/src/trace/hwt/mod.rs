//! Hardware tracing via hwtracer.

use super::{AOTTraceIterator, TraceRecorder, TraceRecorderError};
use hwtracer::{HWTracerError, TemporaryErrorKind};
use std::{error::Error, sync::Arc};

pub(crate) mod mapper;
pub(crate) use mapper::HWTTraceIterator;
mod testing;

pub(crate) struct HWTracer {
    backend: Arc<dyn hwtracer::Tracer>,
}

impl HWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(HWTracer {
            backend: hwtracer::TracerBuilder::new().build()?,
        })
    }
}

impl super::Tracer for HWTracer {
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>> {
        Ok(Box::new(HWTTraceRecorder {
            thread_tracer: Arc::clone(&self.backend).start_collector()?,
        }))
    }
}

/// Hardware thread tracer.
#[derive(Debug)]
struct HWTTraceRecorder {
    thread_tracer: Box<dyn hwtracer::ThreadTracer>,
}

impl TraceRecorder for HWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
        match self.thread_tracer.stop_collector() {
            Ok(x) => Ok(Box::new(HWTTraceIterator::new(x)?)),
            Err(HWTracerError::Temporary(TemporaryErrorKind::TraceBufferOverflow(s))) => {
                Err(TraceRecorderError::TraceBufferOverflow(s))
            }
            _ => todo!(),
        }
    }
}
