use crate::errors::HWTracerError;
use crate::{Block, ThreadTracer, Trace, Tracer};
#[cfg(test)]
use std::fs::File;
use std::iter::Iterator;

/// An empty dummy trace.
#[derive(Debug)]
struct DummyTrace {}

impl Trace for DummyTrace {
    #[cfg(test)]
    fn to_file(&self, _: &mut File) {}

    fn iter_blocks<'t: 'i, 'i>(
        &'t self,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + 'i> {
        Box::new(DummyBlockIterator {})
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct DummyTracer {}

impl DummyTracer {
    pub(super) fn new() -> Self {
        DummyTracer {}
    }
}

impl Tracer for DummyTracer {
    fn thread_tracer(&self) -> Box<dyn ThreadTracer> {
        Box::new(DummyThreadTracer::new())
    }
}

/// A tracer which doesn't really do anything.
pub struct DummyThreadTracer {
    // Keeps track of the state of the tracer.
    is_tracing: bool,
}

impl DummyThreadTracer {
    /// Create a dummy tracer.
    fn new() -> Self {
        Self { is_tracing: false }
    }
}

impl ThreadTracer for DummyThreadTracer {
    fn start_tracing(&mut self) -> Result<(), HWTracerError> {
        if self.is_tracing {
            return Err(HWTracerError::AlreadyTracing);
        }
        self.is_tracing = true;
        Ok(())
    }

    fn stop_tracing(&mut self) -> Result<Box<dyn Trace>, HWTracerError> {
        if !self.is_tracing {
            return Err(HWTracerError::AlreadyStopped);
        }
        self.is_tracing = false;
        Ok(Box::new(DummyTrace {}))
    }
}

// Iterate over the blocks of a DummyTrace.
//
// Note: there will never be any blocks, but we have to implement the interface.
struct DummyBlockIterator {}

impl Iterator for DummyBlockIterator {
    type Item = Result<Block, HWTracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::DummyThreadTracer;
    use crate::{test_helpers, ThreadTracer};

    #[test]
    fn test_basic_usage() {
        test_helpers::test_basic_usage(DummyThreadTracer::new());
    }

    #[test]
    fn test_repeated_tracing() {
        test_helpers::test_repeated_tracing(DummyThreadTracer::new());
    }

    #[test]
    fn test_already_started() {
        test_helpers::test_already_started(DummyThreadTracer::new());
    }

    #[test]
    fn test_not_started() {
        test_helpers::test_not_started(DummyThreadTracer::new());
    }

    #[test]
    fn test_block_iterator() {
        let mut tracer = DummyThreadTracer::new();
        tracer.start_tracing().unwrap();
        let trace = tracer.stop_tracing().unwrap();

        // We expect exactly 0 blocks.
        let expects = Vec::new();
        test_helpers::test_expected_blocks(trace, expects.iter());
    }
}
