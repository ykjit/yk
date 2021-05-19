//! This crate exports the Yorick API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

use std::mem::drop;
use ykrt::{HotThreshold, Location, MT};

#[no_mangle]
pub extern "C" fn yk_mt() -> *const MT {
    MT::global()
}

#[no_mangle]
pub extern "C" fn yk_mt_hot_threshold(mt: *mut MT) -> HotThreshold {
    unsafe { &*mt }.hot_threshold()
}

#[no_mangle]
pub extern "C" fn yk_control_point(mt: *mut MT, loc: *mut Location) {
    if !loc.is_null() {
        unsafe { (&*mt).control_point(Some(&*loc)) };
    } else {
        unsafe { (&*mt).control_point(None) };
    }
}

#[no_mangle]
pub extern "C" fn yk_location_new() -> Location {
    Location::new()
}

#[no_mangle]
pub extern "C" fn yk_location_drop(loc: Location) {
    drop(loc)
}

/// The following module contains exports only used for testing from external C code.
/// These symbols are not shipped as part of the main API.
#[cfg(feature = "c_testing")]
mod c_testing {
    use std::{mem, os::raw::c_char};
    use yktrace::{start_tracing, BlockMap, IRTrace, ThreadTracer, TracingKind};

    const SW_TRACING: usize = 0;
    const HW_TRACING: usize = 1;

    #[no_mangle]
    pub extern "C" fn __yktrace_hwt_mapper_blockmap_new() -> *mut BlockMap {
        Box::into_raw(Box::new(BlockMap::new()))
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_hwt_mapper_blockmap_free(bm: *mut BlockMap) {
        unsafe { Box::from_raw(bm) };
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_hwt_mapper_blockmap_len(bm: *mut BlockMap) -> usize {
        let bm = unsafe { Box::from_raw(bm) };
        let ret = bm.len();
        mem::forget(bm);
        ret
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_start_tracing(kind: usize) -> *mut ThreadTracer {
        let kind: TracingKind = match kind {
            SW_TRACING => TracingKind::SoftwareTracing,
            HW_TRACING => TracingKind::HardwareTracing,
            _ => panic!(),
        };
        Box::into_raw(Box::new(start_tracing(kind)))
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_stop_tracing(tt: *mut ThreadTracer) -> *mut IRTrace {
        let tt = unsafe { Box::from_raw(tt) };
        Box::into_raw(Box::new(tt.stop_tracing().unwrap())) as *mut _
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_irtrace_len(trace: *mut IRTrace) -> usize {
        let trace = unsafe { Box::from_raw(trace) };
        let ret = trace.len();
        mem::forget(trace);
        ret
    }

    /// Fetches the function name (`res_func`) and the block index (`res_bb`) at position `idx` in
    /// `trace`.
    #[no_mangle]
    pub extern "C" fn __yktrace_irtrace_get(
        trace: *mut IRTrace,
        idx: usize,
        res_func: *mut *const c_char,
        res_bb: *mut usize,
    ) {
        let trace = unsafe { Box::from_raw(trace) };
        let blk = trace.get(idx).unwrap();
        unsafe {
            *res_func = blk.func_name().as_ptr();
            *res_bb = blk.bb();
        }
        mem::forget(trace);
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_drop_irtrace(trace: *mut IRTrace) {
        unsafe { Box::from_raw(trace) };
    }

    #[no_mangle]
    pub extern "C" fn __yktrace_irtrace_compile(trace: *mut IRTrace) {
        unsafe { &*trace }.compile();
    }
}
