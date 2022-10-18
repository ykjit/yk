//! C-level errors.

use crate::{
    decode::libipt::{hwt_ipt_is_overflow_err, pt_errstr},
    HWTracerError,
};
use libc::c_int;
use std::{
    error::Error,
    ffi::CStr,
    fmt::{self, Display, Formatter},
};

/// An error indicated by a C-level libipt error code.
#[derive(Debug)]
struct LibIPTError(c_int);

impl Display for LibIPTError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Ask libipt for a string representation of the error code.
        let err_str = unsafe { CStr::from_ptr(pt_errstr(self.0)) };
        write!(f, "libipt error: {}", err_str.to_str().unwrap())
    }
}

impl Error for LibIPTError {
    fn description(&self) -> &str {
        "libipt error"
    }

    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}

#[repr(C)]
#[allow(dead_code)] // Only C constructs these.
#[derive(Eq, PartialEq)]
enum PerfPTCErrorKind {
    Unused,
    Unknown,
    Errno,
    IPT,
}

/// Represents an error occurring in C code.
/// Rust code calling C inspects one of these if the return value of a call indicates error.
#[repr(C)]
pub(crate) struct PerfPTCError {
    typ: PerfPTCErrorKind,
    code: c_int,
}

impl PerfPTCError {
    /// Creates a new error struct defaulting to an unknown error.
    pub(crate) fn new() -> Self {
        Self {
            typ: PerfPTCErrorKind::Unused,
            code: 0,
        }
    }
}

impl From<PerfPTCError> for HWTracerError {
    fn from(err: PerfPTCError) -> HWTracerError {
        // If this assert crashes out, then we forgot a hwt_set_cerr() somewhere in C code.
        debug_assert!(err.typ != PerfPTCErrorKind::Unused);
        match err.typ {
            PerfPTCErrorKind::Unused => HWTracerError::Unknown,
            PerfPTCErrorKind::Unknown => HWTracerError::Unknown,
            PerfPTCErrorKind::Errno => HWTracerError::Errno(err.code),
            PerfPTCErrorKind::IPT => {
                // Overflow is a special case with its own error type.
                match unsafe { hwt_ipt_is_overflow_err(err.code) } {
                    true => HWTracerError::HWBufferOverflow,
                    false => HWTracerError::Custom(Box::new(LibIPTError(err.code))),
                }
            }
        }
    }
}
