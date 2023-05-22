use std::time::SystemTime;

/// A loop that does some work that we can use to build a trace.
#[inline(never)]
pub fn work_loop(iters: u64) -> u64 {
    let mut res = 0;
    for _ in 0..iters {
        // Computation which stops the compiler from eliminating the loop.
        res += SystemTime::now().elapsed().unwrap().subsec_nanos() as u64;
    }
    res
}
