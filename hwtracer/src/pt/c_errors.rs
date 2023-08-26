//! C-level errors.

use crate::HWTracerError;
use libc::c_int;

#[repr(C)]
#[allow(dead_code)] // Only C constructs these.
#[derive(Eq, PartialEq)]
enum PerfPTCErrorKind {
    Unused,
    Unknown,
    Errno,
    PT,
}

enum PTErrorCode {
    Overflow = 0,
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
            PerfPTCErrorKind::PT => {
                // Overflow is a special case with its own error type.
                match err.code {
                    v if v == PTErrorCode::Overflow as c_int => HWTracerError::HWBufferOverflow,
                    _ => unreachable!(),
                }
            }
        }
    }
}
