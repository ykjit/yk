use hwtracer::Trace;
use hwtracer::{
    collect::TraceCollectorBuilder,
    decode::{TraceDecoderBuilder, TraceDecoderKind},
};
use std::time::SystemTime;

/// Prints the addresses of the first `qty` blocks in a trace along with it's name and
/// computation result.
fn print_trace(trace: Box<dyn Trace>, name: &str, result: u32, qty: usize) {
    let dec = TraceDecoderBuilder::new()
        .kind(TraceDecoderKind::LibIPT)
        .build()
        .unwrap();
    let count = dec.iter_blocks(&*trace).count();
    println!("{}: num_blocks={}, result={}", name, count, result);

    for (i, blk) in dec.iter_blocks(&*trace).take(qty).enumerate() {
        if let Some((vaddr, _)) = blk.unwrap().vaddr_range() {
            println!("  block {}: 0x{:x}", i, vaddr);
        } else {
            println!("  block {}: ???", i);
        }
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

/// Collect and decode a trace for a simple computation loop.
///
/// The results are printed to discourage the compiler from optimising the computation out.
fn main() {
    let bldr = TraceCollectorBuilder::new();
    let tc = bldr.build().unwrap();

    for i in 1..4 {
        tc.start_thread_collector().unwrap_or_else(|e| {
            panic!("Failed to start collector: {}", e);
        });
        let res = work();
        let trace = tc.stop_thread_collector().unwrap();
        let name = format!("trace{}", i);
        print_trace(trace, &name, res, 10);
    }
}
