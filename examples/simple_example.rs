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

extern crate hwtracer;
extern crate libc;

use hwtracer::Tracer;
use libc::getpid;

use hwtracer::backends::DummyTracer;
#[cfg(all(perf_pt, target_arch = "x86_64"))]
use hwtracer::backends::PerfPTTracer;

/// Instantiate a tracer suitable for the current platform.
fn tracer() -> Box<Tracer> {
    if cfg!(target_os = "linux") {
        #[cfg(all(perf_pt, target_arch = "x86_64"))] {
            let res = PerfPTTracer::new();
            if res.is_ok() {
                Box::new(res.unwrap())
            } else {
                // CPU doesn't have Intel PT support.
                Box::new(DummyTracer::new())
            }
        }
        #[cfg(not(all(perf_pt, target_arch = "x86_64")))] {
            Box::new(DummyTracer::new())
        }
    } else {
        Box::new(DummyTracer::new())
    }
}

/// Trace a simple computation loop.
///
/// The result is printed to discourage the compiler from optimising the computation out.
fn main() {
    let mut res = 0;
    let pid = unsafe { getpid() };

    let mut tracer = tracer();
    tracer.start_tracing().unwrap_or_else(|e| {
        panic!(format!("Failed to start tracer: {}", e));
    });

    for i in 1..10000 {
        res += i + pid;
    }

    tracer.stop_tracing().unwrap();
    println!("program result: {}", res);
}
