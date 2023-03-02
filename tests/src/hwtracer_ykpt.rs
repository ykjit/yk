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

use hwtracer::decode::{TraceDecoderBuilder, TraceDecoderKind};
use hwtracer::{
    collect::{TraceCollector, TraceCollectorBuilder, TraceCollectorKind},
    Trace,
};
use yktrace::hwt::HWTMapper;

#[no_mangle]
pub extern "C" fn __hwykpt_start_collector() -> *mut TraceCollector {
    let tc = TraceCollectorBuilder::new()
        .kind(TraceCollectorKind::Perf)
        .build()
        .unwrap();
    tc.start_thread_collector().unwrap();
    Box::into_raw(Box::new(tc))
}

#[no_mangle]
pub extern "C" fn __hwykpt_stop_collector(tc: *mut TraceCollector) -> *mut Box<dyn Trace> {
    let tc: Box<TraceCollector> = unsafe { Box::from_raw(tc) };
    let trace = tc.stop_thread_collector().unwrap();
    // We have to return a double-boxed trait object, as the inner Box is a fat pointer that
    // can't be passed across the C ABI.
    Box::into_raw(Box::new(trace))
}

/// Check that the result of mapping `trace` with libipt and ykpt is the same.
#[no_mangle]
pub extern "C" fn __hwykpt_libipt_vs_ykpt(trace: *mut Box<dyn Trace>) {
    let trace: Box<Box<dyn Trace>> = unsafe { Box::from_raw(trace) };

    let ipt_tdec = TraceDecoderBuilder::new()
        .kind(TraceDecoderKind::LibIPT)
        .build()
        .unwrap();
    let mut ipt_itr = ipt_tdec.iter_blocks(&**trace);
    let mut ipt_mapper = HWTMapper::new();
    let ipt_irblocks = ipt_mapper.map_trace(&mut ipt_itr).unwrap();
    let mut ipt_irb_itr = ipt_irblocks.iter();

    let ykpt_tdec = TraceDecoderBuilder::new()
        .kind(TraceDecoderKind::YkPT)
        .build()
        .unwrap();
    let mut ykpt_itr = ykpt_tdec.iter_blocks(&**trace);
    let mut ykpt_mapper = HWTMapper::new();
    let ykpt_irblocks = ykpt_mapper.map_trace(&mut ykpt_itr).unwrap();
    let mut ykpt_irb_itr = ykpt_irblocks.iter();

    loop {
        let next = ipt_irb_itr.next();
        if next.is_none() {
            break;
        }
        let expect = next.unwrap();

        let got = ykpt_irb_itr.next().unwrap();
        assert_eq!(expect, got);
    }
    assert!(ykpt_irb_itr.next().is_none());
}

/// Decode the specified trace and iterate over the resulting blocks.
///
/// Used for benchmarks.
#[no_mangle]
pub extern "C" fn __hwykpt_decode_trace(
    trace: *mut Box<dyn Trace>,
    decoder_kind: TraceDecoderKind,
) {
    let trace: Box<Box<dyn Trace>> = unsafe { Box::from_raw(trace) };

    let ipt_tdec = TraceDecoderBuilder::new()
        .kind(decoder_kind)
        .build()
        .unwrap();

    for b in ipt_tdec.iter_blocks(&**trace) {
        b.unwrap();
    }
}
