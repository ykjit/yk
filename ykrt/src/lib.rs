//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]
#![feature(assert_matches)]
#![feature(int_roundings)]
#![feature(let_chains)]
#![feature(naked_functions)]
#![feature(ptr_sub_ptr)]
#![allow(clippy::type_complexity)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::comparison_chain)]

pub(crate) mod aotsmp;
pub mod compile;
mod location;
mod log;
pub(crate) mod mt;
pub mod promote;
pub(crate) mod stack;
pub(crate) mod thread_intercept;
pub mod trace;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MTThread, MT};
use std::ffi::{c_char, CStr};

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern "C" fn yk_debug_str(msg: *const c_char) {
    MTThread::with(|mtt| {
        mtt.insert_debug_str(unsafe { CStr::from_ptr(msg).to_str().unwrap().to_owned() });
    });
}
