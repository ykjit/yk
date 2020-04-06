extern crate hwtracer;
extern crate libc;

use std::time::SystemTime;
use hwtracer::Trace;
use hwtracer::backends::TracerBuilder;

/// Prints the addresses of the first `qty` blocks in a trace along with it's name and
/// computation result.
fn print_trace(trace: &Box<dyn Trace>, name: &str, result: u32, qty: usize) {
    let count = trace.iter_blocks().count();
    println!("{}: num_blocks={}, result={}", name, count, result);

    for (i, blk) in trace.iter_blocks().take(qty).enumerate() {
       println!("  block {}: 0x{:x}", i, blk.unwrap().first_instr());
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
