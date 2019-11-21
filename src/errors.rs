use std::fmt::{self, Formatter, Display};
use std::error::Error;
use {TracerState, backends::BackendKind};
use libc::{c_int, strerror};
use std::ffi::CStr;

#[derive(Debug)]
pub enum HWTracerError {
    HWBufferOverflow,         // The trace buffer being used by the hardware overflowed.
                              // This is considered a non-fatal error since retrying the tracing
                              // may succeed.
    NoHWSupport(String),      // The hardware doesn't support a required feature. Not fatal for the
                              // same reason as `Permissions`. This may be non-fatal depending
                              // upon whether the consumer could (e.g.) try a different backend.
    BackendUnavailable(BackendKind), // This backend was not compiled in to hwtracer.
    Permissions(String),      // Tracing is not permitted using this backend.
    Errno(c_int),             // Something went wrong in C code.
    TracerState(TracerState), // The tracer is in the wrong state to do the requested task.
    BadConfig(String),        // The tracer configuration was invalid.
    Custom(Box<dyn Error>),       // All other errors can be nested here, however, don't rely on this
                              // for performance since the `Box` incurs a runtime cost.
    Unknown,                  // An unknown error. Used sparingly in C code which doesn't set errno.
}

impl Display for HWTracerError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            HWTracerError::HWBufferOverflow => write!(f, "Hardware trace buffer overflow"),
            HWTracerError::BackendUnavailable(ref s) => write!(f, "Backend unavailble: {:?}", s),
            HWTracerError::NoHWSupport(ref s) => write!(f, "{}", s),
            HWTracerError::Permissions(ref s) => write!(f, "{}", s),
            HWTracerError::Errno(n) => {
                // Ask libc for a string representation of the error code.
                let err_str = unsafe { CStr::from_ptr(strerror(n)) };
                write!(f, "{}", err_str.to_str().unwrap())
            },
            HWTracerError::TracerState(ref s) => write!(f, "Tracer in wrong state: {}", s),
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
            HWTracerError::BackendUnavailable(_) => None,
            HWTracerError::NoHWSupport(_) => None,
            HWTracerError::Permissions(_) => None,
            HWTracerError::TracerState(_) => None,
            HWTracerError::BadConfig(_) => None,
            HWTracerError::Errno(_) => None,
            HWTracerError::Custom(ref bx) => Some(bx.as_ref()),
            HWTracerError::Unknown => None,
        }
    }
}
