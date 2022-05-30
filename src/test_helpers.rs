//! Test helpers.
//!
//! Each struct implementing the [ThreadTracer](trait.ThreadTracer.html) trait should include tests
//! calling the following helpers.

#![cfg(test)]

use super::{Block, HWTracerError, ThreadTracer};
use crate::Trace;
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
pub fn trace_closure<F>(tracer: &mut dyn ThreadTracer, f: F) -> Box<dyn Trace>
where
    F: FnOnce() -> u64,
{
    tracer.start_tracing().unwrap();
    let res = f();
    let trace = tracer.stop_tracing().unwrap();
    println!("traced closure with result: {}", res); // To avoid over-optimisation.
    trace
}

// Check that starting and stopping a tracer works.
pub fn test_basic_usage<T>(mut tracer: T)
where
    T: ThreadTracer,
{
    trace_closure(&mut tracer, || work_loop(500));
}

// Check that repeated usage of the same tracer works.
pub fn test_repeated_tracing<T>(mut tracer: T)
where
    T: ThreadTracer,
{
    for _ in 0..10 {
        trace_closure(&mut tracer, || work_loop(500));
    }
}

// Check that starting a tracer twice makes an appropriate error.
pub fn test_already_started<T>(mut tracer: T)
where
    T: ThreadTracer,
{
    tracer.start_tracing().unwrap();
    match tracer.start_tracing() {
        Err(HWTracerError::AlreadyTracing) => (),
        _ => panic!(),
    };
    tracer.stop_tracing().unwrap();
}

// Check that stopping an unstarted tracer makes an appropriate error.
pub fn test_not_started<T>(mut tracer: T)
where
    T: ThreadTracer,
{
    match tracer.stop_tracing() {
        Err(HWTracerError::AlreadyStopped) => (),
        _ => panic!(),
    };
}

// Helper to check an expected list of blocks matches what we actually got.
pub fn test_expected_blocks(trace: Box<dyn Trace>, mut expect_iter: Iter<Block>) {
    let mut got_iter = trace.iter_blocks();
    loop {
        let expect = expect_iter.next();
        let got = got_iter.next();
        if expect.is_none() || got.is_none() {
            break;
        }
        assert_eq!(
            got.unwrap().unwrap().first_instr(),
            expect.unwrap().first_instr()
        );
    }
    // Check that both iterators were the same length.
    assert!(expect_iter.next().is_none());
    assert!(got_iter.next().is_none());
}

// Trace two loops, one 10x larger than the other, then check the proportions match the number
// of block the trace passes through.
#[cfg(perf_pt_test)]
pub fn test_ten_times_as_many_blocks<T>(mut tracer1: T, mut tracer2: T)
where
    T: ThreadTracer,
{
    let trace1 = trace_closure(&mut tracer1, || work_loop(10));
    let trace2 = trace_closure(&mut tracer2, || work_loop(100));

    // Should be roughly 10x more blocks in trace2. It won't be exactly 10x, due to the stuff
    // we trace either side of the loop itself. On a smallish trace, that will be significant.
    let (ct1, ct2) = (trace1.iter_blocks().count(), trace2.iter_blocks().count());
    assert!(ct2 > ct1 * 9);
}
