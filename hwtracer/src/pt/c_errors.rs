//! C-level errors.

use crate::{HWTracerError, TemporaryErrorKind};
use libc::c_int;
use thiserror::Error;

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
#[derive(Error, Debug)]
enum PTErrorCode {
    /// We couldn't take AUX data out of the buffer quick enough: the head pointer
    /// caught up with the tail pointer.
    #[error("AUX buffer overflow")]
    AuxOverflow = 0,
    /// The final trace storage buffer was exhausted.
    #[error("Trace buffer capacity too small")]
    TraceCapacity = 1,
    // Perf reported that data buffer samples were lost.
    #[error("Perf event lost")]
    EventLost = 2,
}

impl From<i32> for PTErrorCode {
    fn from(v: i32) -> Self {
        match v {
            v if v == Self::AuxOverflow as i32 => Self::AuxOverflow,
            v if v == Self::TraceCapacity as i32 => Self::TraceCapacity,
            v if v == Self::EventLost as i32 => Self::EventLost,
            _ => unreachable!(),
        }
    }
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
                #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                if err.code == 16 {
                    return HWTracerError::Temporary(TemporaryErrorKind::PerfBusy);
                }

                HWTracerError::Unrecoverable(format!("c set errno {}", err.code))
            }
            #[cfg(pt)]
            PerfPTCErrorKind::PT => HWTracerError::Temporary(
                TemporaryErrorKind::TraceBufferOverflow(PTErrorCode::from(err.code).to_string()),
            ),
        }
    }
}
