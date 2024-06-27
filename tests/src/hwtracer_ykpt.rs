#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

//! Testing and benchmarking bits for hwtracer's ykpt decoder.
//!
//! Why is this so convoluted? Read on...
//!
//! Ideally tests and benchmarks would be written in pure Rust, however the ykpt
//! decoder relies on the block map section inserted by ykllvm. This means that the tests and
//! benchmarks have to be LTO linked by ykllvm, and be written in C. But the plot thickens, as the
//! kinds of things we want the tests to check are Rust-based, so we will have to call back into
//! Rust somehow.
//!
//! To that end, the test files in `tests/hwtracer_ykpt` are compiled into test binaries (as a
//! langtester suite) and then they call into this file to have assertions checked in Rust code.

use hwtracer::{ThreadTracer, Trace, TracerBuilder};
use std::ffi::c_void;

#[no_mangle]
/// The value returned by this function *must* be passed to [__hwykpt_stop_collector] or memory
/// will leak.
pub extern "C" fn __hwykpt_start_collector() -> *mut Box<dyn ThreadTracer> {
    let t = TracerBuilder::new().build().unwrap();
    let tt = t.start_collector().unwrap();
    // In order to pass the trait object over the FFI, we have to box it twice.
    Box::into_raw(Box::new(tt))
}

#[no_mangle]
/// The value passed to this function *must* have come from [ __hwykpt_start_collector]; doing
/// otherwise leads to undefined behaviour.
pub extern "C" fn __hwykpt_stop_collector(tc: *mut Box<dyn ThreadTracer>) -> *mut c_void {
    let tt: Box<dyn ThreadTracer> = *unsafe { Box::from_raw(tc) };
    let trace = tt.stop_collector().unwrap();
    // We have to return a double-boxed trait object, as the inner Box is a fat pointer that
    // can't be passed across the C ABI.
    Box::into_raw(Box::new(trace)) as *mut c_void
}

use hwtracer::errors::HWTracerError;

/// Decode the specified trace and iterate over the resulting blocks.
///
/// Returns `true` on success, or `false` if there was a temporary error.
///
/// Panics on fatal errors.
///
/// Used for benchmarks.
#[no_mangle]
pub extern "C" fn __hwykpt_decode_trace(trace: *mut Box<dyn Trace>) -> bool {
    let trace: Box<Box<dyn Trace>> = unsafe { Box::from_raw(trace) };
    for b in trace.iter_blocks() {
        match b {
            Ok(_) => (),
            Err(HWTracerError::Temporary(_)) => return false,
            Err(_) => panic!(),
        }
    }
    true
}
