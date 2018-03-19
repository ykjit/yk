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

use {Tracer, Block};
use errors::HWTracerError;
use std::iter::Iterator;
use Trace;
use TracerState;

#[cfg(debug_assertions)]
use std::ops::Drop;
#[cfg(test)]
use std::fs::File;

/// An empty dummy trace.
#[derive(Debug)]
struct DummyTrace {}

impl Trace for DummyTrace {
    #[cfg(test)]
    fn to_file(&self, _: &mut File) {}

    fn iter_blocks<'t: 'i, 'i>(&'t self) -> Box<Iterator<Item=Block> + 'i> {
       Box::new(DummyBlockIterator{})
    }
}

/// A tracer which doesn't really do anything.
pub struct DummyTracer {
    // Keeps track of the state of the tracer.
    state: TracerState,
}

impl DummyTracer {
    /// Create a dummy tracer.
    pub fn new() -> Self {
        Self { state: TracerState::Stopped }
    }

    fn err_if_destroyed(&self) -> Result<(), HWTracerError> {
        if self.state == TracerState::Destroyed {
            return Err(HWTracerError::TracerDestroyed);
        }
        Ok(())
    }
}

impl Tracer for DummyTracer {
    fn start_tracing(&mut self) -> Result<(), HWTracerError> {
        self.err_if_destroyed()?;
        if self.state != TracerState::Stopped {
            return Err(HWTracerError::TracerAlreadyStarted);
        }
        self.state = TracerState::Started;
        Ok(())
    }

    fn stop_tracing(&mut self) -> Result<Box<Trace>, HWTracerError> {
        self.err_if_destroyed()?;
        if self.state != TracerState::Started {
            return Err(HWTracerError::TracerNotStarted);
        }
        self.state = TracerState::Stopped;
        Ok(Box::new(DummyTrace{}))
    }

    fn destroy(&mut self) -> Result<(), HWTracerError> {
        self.err_if_destroyed()?;
        self.state = TracerState::Destroyed;
        Ok(())
    }
}

#[cfg(debug_assertions)]
impl Drop for DummyTracer {
    fn drop(&mut self) {
        if self.state != TracerState::Destroyed {
            panic!("DummyTracer dropped without destroy()");
        }
    }
}

// Iterate over the blocks of a DummyTrace.
//
// Note: there will never be any blocks, but we have to implement the interface.
struct DummyBlockIterator {}

impl Iterator for DummyBlockIterator {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::DummyTracer;
    use ::test_helpers;

    #[test]
    fn test_basic_usage() {
        test_helpers::test_basic_usage(DummyTracer::new());
    }

    #[test]
    fn test_repeated_tracing() {
        test_helpers::test_repeated_tracing(DummyTracer::new());
    }

    #[test]
    fn test_already_started() {
        test_helpers::test_already_started(DummyTracer::new());
    }

    #[test]
    fn test_not_started() {
        test_helpers::test_not_started(DummyTracer::new());
    }

    #[test]
    fn test_use_tracer_after_destroy1() {
        test_helpers::test_use_tracer_after_destroy1(DummyTracer::new());
    }

    #[test]
    fn test_use_tracer_after_destroy2() {
        test_helpers::test_use_tracer_after_destroy2(DummyTracer::new());
    }

    #[test]
    fn test_use_tracer_after_destroy3() {
        test_helpers::test_use_tracer_after_destroy3(DummyTracer::new());
    }

    #[test]
    fn test_block_iterator() {
        use ::Tracer;
        let mut tracer = DummyTracer::new();
        tracer.start_tracing().unwrap();
        let trace = tracer.stop_tracing().unwrap();

        // We expect exactly 0 blocks.
        let expects =  Vec::new();
        test_helpers::test_expected_blocks(tracer, trace, expects.iter());
    }

    #[cfg(debug_assertions)]
    #[should_panic]
    #[test]
    fn test_drop_without_destroy() {
        test_helpers::test_drop_without_destroy(DummyTracer::new());
    }
}
