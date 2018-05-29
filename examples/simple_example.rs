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

use std::time::SystemTime;
use hwtracer::Trace;
use hwtracer::backends::TracerBuilder;

/// Prints the addresses of the first `qty` blocks in a trace along with it's name and
/// computation result.
fn print_trace(trace: &Box<Trace>, name: &str, result: u32, qty: usize) {
    let count = trace.iter_blocks().count();
    println!("{}: num_blocks={}, result={}", name, count, result);

    for (i, blk) in trace.iter_blocks().take(qty).enumerate() {
       println!("  block {}: 0x{:x}", i, blk.unwrap().start_vaddr());
    }
    if count > qty {
        println!("  ... {} more", count - qty);
    }
    println!("");
}

/// It's up to you to ensure the compiler doesn't move or optimise out the computation you intend
/// to trace. Here we've used an uninlinable function. See also `test::black_box`.
#[inline(never)]
fn work() -> u32 {
    let mut res = 0;
    for _ in 0..50 {
        // Computation which stops the compiler from eliminating the loop.
        res += SystemTime::now().elapsed().unwrap().subsec_nanos();
    }
    res
}

/// Trace a simple computation loop.
///
/// The results are printed to discourage the compiler from optimising the computation out.
fn main() {
    let mut bldr = TracerBuilder::new();
    println!("Backend configuration: {:?}", bldr.config());
    let mut thr_tracer = bldr.build().unwrap().thread_tracer();

    for i in 1..4 {
        thr_tracer.start_tracing().unwrap_or_else(|e| {
            panic!(format!("Failed to start tracer: {}", e));
        });
        let res = work();
        let trace = thr_tracer.stop_tracing().unwrap();
        let name = format!("trace{}", i);
        print_trace(&trace, &name, res, 10);
    }
}
