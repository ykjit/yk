// Check that we never accidentally permit a `ThreadTracer` to be shared between threads.
extern crate hwtracer;

use hwtracer::backends::TracerBuilder;
use std::{sync::Arc, thread};

fn main() {
    let thr_tracer = Arc::new(TracerBuilder::new().build().unwrap().thread_tracer());
    thread::spawn(move || thr_tracer).join().unwrap();
    //~^ ERROR cannot be sent between threads safely [E0277]
    //~| ERROR cannot be shared between threads safely [E0277]
}
