//! The Rust side of the C header `ykrt.h`.

use crate::{
    location::Location,
    mt::{HotThreshold, MT},
};
use std::mem::drop;

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
