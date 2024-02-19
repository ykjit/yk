//! Hardware tracing via hwtracer.

use super::{errors::InvalidTraceError, AOTTraceIterator, TraceRecorder, TracedAOTBlock};
use std::{error::Error, sync::Arc};

pub(crate) mod mapper;
pub(crate) use mapper::HWTMapper;
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
struct HWTTraceRecorder {
    thread_tracer: Box<dyn hwtracer::ThreadTracer>,
}

impl TraceRecorder for HWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, InvalidTraceError> {
        let tr = self.thread_tracer.stop_collector().unwrap();
        let mut mt = HWTMapper::new();
        let mapped = mt
            .map_trace(tr)
            .map_err(|_| InvalidTraceError::InternalError)?;
        if mapped.is_empty() {
            Err(InvalidTraceError::EmptyTrace)
        } else {
            Ok(Box::new(HWTTraceIterator {
                trace: mapped.into_iter(),
            }))
        }
    }
}

struct HWTTraceIterator {
    trace: std::vec::IntoIter<TracedAOTBlock>,
}

impl AOTTraceIterator for HWTTraceIterator {}

impl Iterator for HWTTraceIterator {
    type Item = TracedAOTBlock;
    fn next(&mut self) -> Option<Self::Item> {
        self.trace.next()
    }
}
