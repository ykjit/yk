use crate::{collect::TraceCollectorKind, decode::TraceDecoderKind};
use libc::{c_int, strerror};
use std::error::Error;
use std::ffi::{self, CStr};
use std::fmt::{self, Display, Formatter};
use std::io;
use std::num::ParseIntError;

#[derive(Debug)]
pub enum HWTracerError {
    HWBufferOverflow, // The trace buffer being used by the hardware overflowed.
    // This is considered a non-fatal error since retrying the tracing
    // may succeed.
    NoHWSupport(String), // The hardware doesn't support a required feature. Not fatal for the
    // same reason as `Permissions`. This may be non-fatal depending
    // upon whether the consumer could (e.g.) try a different collector.
    CollectorUnavailable(TraceCollectorKind), // This collector was not compiled in to hwtracer.
    DecoderUnavailable(TraceDecoderKind),     // This decoder was not compiled in to hwtracer.
    Permissions(String),                      // Permission denied.
    Errno(c_int),                             // Something went wrong in C code.
    AlreadyCollecting,                        // Trying to start an already collecting collector.
    AlreadyStopped,                           // Trying to stop a not-currently-active collector.
    BadConfig(String),                        // Configuration was invalid.
    Custom(Box<dyn Error>), // All other errors can be nested here, however, don't rely on this
    // for performance since the `Box` incurs a runtime cost.
    Unknown, // An unknown error. Used sparingly in C code which doesn't set errno.
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
