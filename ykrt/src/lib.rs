//! Interpreter-facing API to the Yk meta-tracer.

#![cfg_attr(test, feature(test))]
#![feature(assert_matches)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::upper_case_acronyms)]

pub(crate) mod aotsmp;
pub mod compile;
mod job_queue;
mod location;
mod log;
pub(crate) mod mt;
pub mod promote;
pub(crate) mod stack;
pub(crate) mod thread_intercept;
pub mod trace;

pub use self::location::Location;
pub use self::mt::{HotThreshold, MT, MTThread};
use std::ffi::{CStr, c_char};

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yk_debug_str(msg: *const c_char) {
    MTThread::with_borrow_mut(|mtt| {
        mtt.insert_debug_str(unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() });
    });
}
