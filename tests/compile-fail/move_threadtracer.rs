// Check that we never accidentally permit a `ThreadTracer` to be moved to another thread.
extern crate hwtracer;

use hwtracer::backends::TracerBuilder;
use std::thread;

fn main() {
    let thr_tracer = TracerBuilder::new().build().unwrap().thread_tracer();
    thread::spawn(move || thr_tracer).join().unwrap();
    //~^ ERROR the trait bound `hwtracer::ThreadTracer: std::marker::Send` is not satisfied
}
