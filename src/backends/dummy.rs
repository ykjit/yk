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

use Tracer;
use errors::HWTracerError;
use HWTrace;
use std::ptr;
#[cfg(debug_assertions)]
use std::ops::Drop;
use TracerState;

/// A tracer which doesn't really do anything.
pub struct DummyTracer {
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

    fn stop_tracing(&mut self) -> Result<HWTrace, HWTracerError> {
        self.err_if_destroyed()?;
        if self.state != TracerState::Started {
            return Err(HWTracerError::TracerNotStarted);
        }
        self.state = TracerState::Stopped;
        Ok(HWTrace::from_buf(ptr::null(), 0)) // An empty trace.
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

    #[cfg(debug_assertions)]
    #[should_panic]
    #[test]
    fn test_drop_without_destroy() {
        test_helpers::test_drop_without_destroy(DummyTracer::new());
    }
}
