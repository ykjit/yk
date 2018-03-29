// Copyright (c) 2017-2018 King's College London
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

#![cfg_attr(perf_pt, feature(asm))]
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![feature(link_args)]

extern crate libc;
extern crate time;
extern crate tempfile;

pub mod errors;
pub mod backends;
pub mod util;

use errors::HWTracerError;
use std::fmt::Debug;
use std::iter::Iterator;
use std::fmt::{self, Display, Formatter};
#[cfg(test)]
use std::fs::File;


/// Information about a basic block.
#[derive(Debug, Eq, PartialEq)]
pub struct Block {
    start_vaddr: u64,
}

impl Block {
    pub fn new(start_vaddr: u64) -> Self {
        Self { start_vaddr: start_vaddr }
    }

    pub fn start_vaddr(&self) -> u64 {
        self.start_vaddr
    }
}

/// Represents a generic trace.
///
/// Each backend has its own concrete implementation.
pub trait Trace: Debug {
    /// Dump the trace to the specified filename.
    ///
    /// The exact format varies per-backend.
    #[cfg(test)]
    fn to_file(&self, file: &mut File);

    /// Iterate over the blocks of the trace.
    fn iter_blocks<'t: 'i, 'i>(&'t self) -> Box<Iterator<Item=Result<Block, HWTracerError>> + 'i>;
}

/// The interface offered by all tracer types.
///
/// It is the job of the consumer of this library to decide which specific tracer type to
/// instantiate, and to set it up using its type-specific configuration methods.
pub trait Tracer {
    /// Start recording a trace.
    ///
    /// Tracing continues until [stop_tracing](trait.Tracer.html#method.stop_tracing) is called.
    fn start_tracing(&mut self) -> Result<(), HWTracerError>;
    /// Turns off the tracer.
    ///
    /// [start_tracing](trait.Tracer.html#method.start_tracing) must have been called prior.
    fn stop_tracing(&mut self) -> Result<Box<Trace>, HWTracerError>;
    /// Destroy a tracer.
    ///
    /// This is explicit because it might fail.
    fn destroy(&mut self) -> Result<(), HWTracerError>;
}

// Keeps track of the internal state of a tracer.
#[derive(PartialEq, Eq, Debug)]
pub enum TracerState {
    Stopped,
    Started,
    Destroyed,
}

impl TracerState {
    /// Returns the error corresponding with the `TracerState`.
    pub fn as_error(self) -> HWTracerError {
        HWTracerError::TracerState(self)
    }
}

impl Display for TracerState {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            TracerState::Started => write!(f, "started"),
            TracerState::Stopped => write!(f, "stopped"),
            TracerState::Destroyed => write!(f, "destroyed"),
        }
    }
}

// Test helpers.
//
// Each struct implementing the [Tracer](trait.Tracer.html) trait should include tests calling the
// following helpers.
#[cfg(test)]
mod test_helpers {
    use super::{HWTracerError, Tracer, Block, TracerState};
    use Trace;
    use std::slice::Iter;
    use std::time::SystemTime;

    // A loop that does some work that we can use to build a trace.
    #[inline(never)]
    pub fn work_loop(iters: u64) -> u64 {
        let mut res = 0;
        for _ in 0..iters {
            // Computation which stops the compiler from eliminating the loop.
            res += SystemTime::now().elapsed().unwrap().subsec_nanos() as u64;
        }
        res
    }

    // Trace a closure that returns a u64.
    pub fn trace_closure<F>(tracer: &mut Tracer, f: F) -> Box<Trace> where F: FnOnce() -> u64 {
        tracer.start_tracing().unwrap();
        let res = f();
        let trace = tracer.stop_tracing().unwrap();
        println!("traced closure with result: {}", res); // To avoid over-optimisation.
        trace
    }

    // Check that starting and stopping a tracer works.
    pub fn test_basic_usage<T>(mut tracer: T) where T: Tracer {
        trace_closure(&mut tracer, || work_loop(500));
        tracer.destroy().unwrap();
    }

    // Check that repeated usage of the same tracer works.
    pub fn test_repeated_tracing<T>(mut tracer: T) where T: Tracer {
        for _ in 0..10 {
            trace_closure(&mut tracer, || work_loop(500));
        }
        tracer.destroy().unwrap();
    }

    // Check that starting a tracer twice makes an appropriate error.
    pub fn test_already_started<T>(mut tracer: T) where T: Tracer {
        tracer.start_tracing().unwrap();
        match tracer.start_tracing() {
            Err(HWTracerError::TracerState(TracerState::Started)) => (),
            _ => panic!(),
        };
        tracer.stop_tracing().unwrap();
        tracer.destroy().unwrap();
    }

    // Check that stopping an unstarted tracer makes an appropriate error.
    pub fn test_not_started<T>(mut tracer: T) where T: Tracer {
        match tracer.stop_tracing() {
            Err(HWTracerError::TracerState(TracerState::Stopped)) => (),
            _ => panic!(),
        };
        tracer.destroy().unwrap();
    }

    // Check that using a tracer after it has been destroyed causes a panic.
    pub fn test_use_tracer_after_destroy1<T>(mut tracer: T) where T: Tracer {
        tracer.destroy().unwrap();
        match tracer.start_tracing() {
            Err(HWTracerError::TracerState(TracerState::Destroyed)) => (),
            _ => panic!(),
        };
    }

    pub fn test_use_tracer_after_destroy2<T>(mut tracer: T) where T: Tracer {
        tracer.start_tracing().unwrap();
        tracer.destroy().unwrap();
        match tracer.stop_tracing() {
            Err(HWTracerError::TracerState(TracerState::Destroyed)) => (),
            _ => panic!(),
        };
    }

    pub fn test_use_tracer_after_destroy3<T>(mut tracer: T) where T: Tracer {
        tracer.destroy().unwrap();
        match tracer.destroy() {
            Err(HWTracerError::TracerState(TracerState::Destroyed)) => (),
            _ => panic!(),
        };
    }

    // Check that in a debug build, a dropped, non-destroyed tracer causes a panic.
    //
    // Tests calling this helper should be marked: #[cfg(debug_assertions)] and #[should_panic].
    #[cfg(debug_assertions)]
    pub fn test_drop_without_destroy<T>(tracer: T) where T: Tracer {
        drop(tracer);
    }

    // Helper to check an expected list of blocks matches what we actually got.
    pub fn test_expected_blocks<T>(mut tracer: T, trace: Box<Trace>,
                                  mut expect_iter: Iter<Block>)
        where T: Tracer {
        let mut got_iter = trace.iter_blocks();
        loop {
            let expect = expect_iter.next();
            let got = got_iter.next();
            if expect.is_none() || got.is_none() {
                break;
            }
            assert_eq!(&got.unwrap().unwrap(), expect.unwrap());
        }
        // Check that both iterators were the same length.
        assert!(expect_iter.next().is_none());
        assert!(got_iter.next().is_none());
        tracer.destroy().unwrap();
    }

    // Trace two loops, one 10x larger than the other, then check the proportions match the number
    // of block the trace passes through.
    #[cfg(perf_pt_test)]
    pub fn test_ten_times_as_many_blocks<T>(mut tracer1: T, mut tracer2: T)  where T: Tracer {
        let trace1 = trace_closure(&mut tracer1, || work_loop(10));
        let trace2 = trace_closure(&mut tracer2, || work_loop(100));

        // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
        // we trace either side of the loop itself. On a smallish trace, that will be significant.
        let (ct1, ct2) = (trace1.iter_blocks().count(), trace2.iter_blocks().count());
        assert!(ct2 > ct1 * 9);

        tracer1.destroy().unwrap();
        tracer2.destroy().unwrap();
    }
}
