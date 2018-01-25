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

#![cfg_attr(all(perf_pt, target_arch = "x86_64"), feature(asm))]
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate libc;
extern crate time;

pub mod errors;
pub mod backends;
pub mod util;

use errors::HWTracerError;
use std::ops::Drop;
use std::ptr;
use libc::{free, c_void};

#[cfg(debug_assertions)]
use std::convert::AsRef;
#[cfg(debug_assertions)]
use std::path::Path;

#[allow(dead_code)]
pub struct HWTrace {
    buf: *const u8,
    len: u64,
}

impl HWTrace {
    /// Makes a new trace from a raw pointer and a size.
    ///
    /// The `buf` argument is assumed to have been allocated on the heap using malloc(3). `len`
    /// must be less than or equal to the allocated size.
    ///
    /// Once an instance is constructed, underlying allocation and it's freeing is the
    /// responsibility of the instance.
    fn from_buf(buf: *const u8, len: u64) -> Self {
        Self {buf: buf, len: len}
    }

    /// Write the raw trace packets into the specified file.
    ///
    /// This can be useful for developers who want to use (e.g.) the pt utility to inspect the raw
    /// packet stream.
    #[cfg(debug_assertions)]
    pub fn to_file<T>(&self, filename: T) where T: AsRef<Path> {
        use std::slice;
        use std::fs::File;
        use std::io::prelude::*;

        let mut f = File::create(filename).unwrap();
        let slice = unsafe { slice::from_raw_parts(self.buf, self.len as usize) };
        f.write(slice).unwrap();
    }
}

/// Once a HWTrace is brought into existence, we say the instance owns the C-level allocation. When
/// the HWTrace falls out of scope, free up the memory.
impl Drop for HWTrace {
    fn drop(&mut self) {
        if self.buf != ptr::null() {
            unsafe { free(self.buf as *mut c_void) };
        }
    }
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
    fn stop_tracing(&mut self) -> Result<HWTrace, HWTracerError>;
}

/// XXX test to_file()
#[cfg(all(test, debug_assertions))]
mod tests {
    use std::fs::File;
    use std::slice;
    use std::io::prelude::*;
    use libc::malloc;
    use super::HWTrace;

    /// Test writing a trace to file.
    #[test]
    fn test_to_file() {
        // Allocate and fill a buffer to make a "trace" from.
        let size = 33;
        let buf = unsafe { malloc(size) as *mut u8 };
        let sl = unsafe { slice::from_raw_parts_mut(buf, size) };
        for (i, byte) in sl.iter_mut().enumerate() {
            *byte = i as u8;
        }

        // Make the trace and Write it to a file.
        let filename = String::from("test_to_file.ptt");
        let trace = HWTrace::from_buf(buf, size as u64);
        trace.to_file(&filename);

        // Check the resulting file makes sense.
        let file = File::open(&filename).unwrap();
        let mut total_bytes = 0;
        for (i, byte) in file.bytes().enumerate() {
            assert_eq!(i as u8, byte.unwrap());
            total_bytes += 1;
        }
        assert_eq!(total_bytes, size);
    }
}

/// Test helpers.
///
/// Each struct implementing the [Tracer](trait.Tracer.html) trait should include tests calling the
/// following helpers.
#[cfg(test)]
mod test_helpers {
    use super::{HWTracerError, Tracer};
    use libc::getpid;

    // A loop that does some work, that we can use to build a trace.
    fn work_loop() {
        let mut res = unsafe { getpid() };
        for _ in 0..100 {
            res += 33;
        }
        println!("{}", res); // Ensure the loop isn't optimised out.
    }

    /// Check that starting and stopping a tracer works.
    pub fn test_basic_usage<T>(mut tracer: T) where T: Tracer {
        tracer.start_tracing().unwrap();
        work_loop();
        tracer.stop_tracing().unwrap();
    }

    /// Check that starting a tracer twice makes an appropriate error.
    pub fn test_already_started<T>(mut tracer: T) where T: Tracer {
        tracer.start_tracing().unwrap();
        match tracer.start_tracing() {
            Err(HWTracerError::TracerAlreadyStarted) => (),
            _ => panic!(),
        };
        tracer.stop_tracing().unwrap();
    }

    /// Check that stopping an unstarted tracer makes an appropriate error.
    pub fn test_not_started<T>(mut tracer: T) where T: Tracer {
        match tracer.stop_tracing() {
            Err(HWTracerError::TracerNotStarted) => (),
            _ => panic!(),
        };
    }
}
