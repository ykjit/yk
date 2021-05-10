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
pub extern "C" fn yk_new_location() -> Location {
    Location::new()
}

#[no_mangle]
pub extern "C" fn yk_drop_location(loc: *mut Location) {
    drop(loc)
}

/// The following module contains exports only used for testing from external C code.
/// These symbols are not shipped as part of the main API.
#[cfg(feature = "c_testing")]
mod c_testing {
    use yktrace::BlockMap;

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
        Box::leak(bm);
        ret
    }
}
