//! C-level errors.

use crate::{HWTracerError, TemporaryErrorKind};
use libc::c_int;

#[repr(C)]
#[allow(dead_code)] // Only C constructs these.
#[derive(Eq, PartialEq)]
enum PerfPTCErrorKind {
    Unused = 0,
    Unknown = 1,
    Errno = 2,
    #[cfg(pt)]
    PT = 3,
}

#[cfg(pt)]
#[repr(C)]
enum PTErrorCode {
    Overflow = 0,
}

/// Represents an error occurring in C code.
/// Rust code calling C inspects one of these if the return value of a call indicates error.
#[cfg(pt)]
#[repr(C)]
pub(crate) struct PerfPTCError {
    typ: PerfPTCErrorKind,
    code: c_int,
}

#[cfg(pt)]
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
            PerfPTCErrorKind::Unused => {
                HWTracerError::Unrecoverable("PerfPTCErrorKind::Unused".into())
            }
            PerfPTCErrorKind::Unknown => {
                HWTracerError::Unrecoverable("PerfPTCErrorKind::Unknown".into())
            }
            PerfPTCErrorKind::Errno => {
                HWTracerError::Unrecoverable(format!("c set errnor {}", err.code))
            }
            #[cfg(pt)]
            PerfPTCErrorKind::PT => {
                // Overflow is a special case with its own error type.
                match err.code {
                    v if v == PTErrorCode::Overflow as c_int => {
                        HWTracerError::Temporary(TemporaryErrorKind::TraceBufferOverflow)
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}
