// Copyright (c) 2017 King's College London
// created by the Software Development Team <http://soft-dev.org/>
//
// The Universal Permissive License (UPL), Version 1.0
//
// Subject to the condition set forth below, permission is hereby granted to any
// person obtaining a copy of this software, associated documentation and/or
// data (collectively the "Software"), free of charge and under any and all
// copyright rights in the Software, and any and all patent rights owned or
// freely licensable by each licensor hereunder covering either (i) the
// unmodified Software as contributed to or provided by such licensor, or (ii)
// the Larger Works (as defined below), to deal in both
//
// (a) the Software, and
// (b) any piece of software and/or hardware listed in the lrgrwrks.txt file
// if one is included with the Software (each a "Larger Work" to which the Software
// is contributed by such licensors),
//
// without restriction, including without limitation the rights to copy, create
// derivative works of, display, perform, and distribute the Software and make,
// use, sell, offer for sale, import, export, have made, and have sold the
// Software and the Larger Work(s), and to sublicense the foregoing rights on
// either these or other terms.
//
// This license is subject to the following condition: The above copyright
// notice and either this complete permission notice or at a minimum a reference
// to the UPL must be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use {Tracer, ThreadTracer, Block};
use errors::HWTracerError;
use std::iter::Iterator;
use Trace;
use TracerState;
#[cfg(test)]
use std::fs::File;

/// An empty dummy trace.
#[derive(Debug)]
struct DummyTrace {}

impl Trace for DummyTrace {
    #[cfg(test)]
    fn to_file(&self, _: &mut File) {}

    fn iter_blocks<'t: 'i, 'i>(&'t self) -> Box<dyn Iterator<Item=Result<Block, HWTracerError>> + 'i> {
       Box::new(DummyBlockIterator{})
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct DummyTracer {}

impl DummyTracer {
    pub (super) fn new() -> Self {
        DummyTracer{}
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
    state: TracerState,
}

impl DummyThreadTracer {
    /// Create a dummy tracer.
    fn new() -> Self {
        Self { state: TracerState::Stopped }
    }
}

impl ThreadTracer for DummyThreadTracer {
    fn start_tracing(&mut self) -> Result<(), HWTracerError> {
        if self.state != TracerState::Stopped {
            return Err(TracerState::Started.as_error());
        }
        self.state = TracerState::Started;
        Ok(())
    }

    fn stop_tracing(&mut self) -> Result<Box<dyn Trace>, HWTracerError> {
        if self.state != TracerState::Started {
            return Err(TracerState::Stopped.as_error());
        }
        self.state = TracerState::Stopped;
        Ok(Box::new(DummyTrace{}))
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
    use ::test_helpers;

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
        use ::ThreadTracer;
        let mut tracer = DummyThreadTracer::new();
        tracer.start_tracing().unwrap();
        let trace = tracer.stop_tracing().unwrap();

        // We expect exactly 0 blocks.
        let expects =  Vec::new();
        test_helpers::test_expected_blocks(trace, expects.iter());
    }
}
