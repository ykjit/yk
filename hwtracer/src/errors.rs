use crate::{collect::TraceCollectorKind, decode::TraceDecoderKind};
use libc::{c_int, strerror};
use std::error::Error;
use std::ffi::{self, CStr};
use std::fmt::{self, Display, Formatter};
use std::io;
use std::num::ParseIntError;

#[derive(Debug)]
pub enum HWTracerError {
    /// The trace buffer being used by the hardware overflowed.
    HWBufferOverflow,
    /// The hardware doesn't support a required feature.
    NoHWSupport(String),
    /// This collector was not compiled in to hwtracer.
    CollectorUnavailable(TraceCollectorKind),
    /// This decoder was not compiled into hwtracer.
    DecoderUnavailable(TraceDecoderKind),
    /// Permission denied.
    Permissions(String),
    /// Something went wrong in C code.
    Errno(c_int),
    /// The collector is already collecting.
    AlreadyCollecting,
    /// Trying to stop a not-currently-active collector.
    AlreadyStopped,
    /// Invalid configuration.
    BadConfig(String),
    /// An unknown error. Used sparingly for C code which doesn't set errno.
    Unknown,
    /// Failed to decode trace.
    TraceParseError(String),
    /// Any other error.
    Custom(Box<dyn Error>),
}

impl Display for HWTracerError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            HWTracerError::HWBufferOverflow => write!(f, "Hardware trace buffer overflow"),
            HWTracerError::CollectorUnavailable(ref s) => {
                write!(f, "Trace collector unavailble: {:?}", s)
            }
            HWTracerError::DecoderUnavailable(ref s) => {
                write!(f, "Trace decoder unavailble: {:?}", s)
            }
            HWTracerError::NoHWSupport(ref s) => write!(f, "{}", s),
            HWTracerError::Permissions(ref s) => write!(f, "{}", s),
            HWTracerError::Errno(n) => {
                // Ask libc for a string representation of the error code.
                let err_str = unsafe { CStr::from_ptr(strerror(n)) };
                write!(f, "{}", err_str.to_str().unwrap())
            }
            HWTracerError::AlreadyCollecting => {
                write!(f, "Can't start a collector that's already collecting")
            }
            HWTracerError::AlreadyStopped => write!(f, "Can't stop an inactice collector"),
            HWTracerError::BadConfig(ref s) => write!(f, "{}", s),
            HWTracerError::Custom(ref bx) => write!(f, "{}", bx),
            HWTracerError::TraceParseError(ref s) => write!(f, "failed to parse trace: {}", s),
            HWTracerError::Unknown => write!(f, "Unknown error"),
        }
    }
}

impl Error for HWTracerError {
    fn description(&self) -> &str {
        "hwtracer error"
    }

    fn cause(&self) -> Option<&dyn Error> {
        match *self {
            HWTracerError::HWBufferOverflow => None,
            HWTracerError::CollectorUnavailable(_) => None,
            HWTracerError::DecoderUnavailable(_) => None,
            HWTracerError::NoHWSupport(_) => None,
            HWTracerError::Permissions(_) => None,
            HWTracerError::AlreadyCollecting => None,
            HWTracerError::AlreadyStopped => None,
            HWTracerError::BadConfig(_) => None,
            HWTracerError::Errno(_) => None,
            HWTracerError::Custom(ref bx) => Some(bx.as_ref()),
            HWTracerError::TraceParseError(_) => None,
            HWTracerError::Unknown => None,
        }
    }
}

impl From<io::Error> for HWTracerError {
    fn from(err: io::Error) -> Self {
        HWTracerError::Custom(Box::new(err))
    }
}

impl From<ffi::NulError> for HWTracerError {
    fn from(err: ffi::NulError) -> Self {
        HWTracerError::Custom(Box::new(err))
    }
}

impl From<ParseIntError> for HWTracerError {
    fn from(err: ParseIntError) -> Self {
        HWTracerError::Custom(Box::new(err))
    }
}
