use std::{io, ffi, num};
use std::fmt::{self, Formatter, Display};
use ::PERF_PERMS_PATH;

#[derive(Debug)]
pub enum TraceMeError {
    // Wrapped errors from elsewhere.
    FFIIntoString(ffi::IntoStringError),
    FFINul(ffi::NulError),
    IO(io::Error),
    NumParseInt(num::ParseIntError),
    // Our own errors.
    CFailure,
    InvalidFileName(String),
    TracerAlreadyStarted,
    TracerNotStarted,
    TracingNotPermitted,
}

impl From<ffi::IntoStringError> for TraceMeError {
    fn from(err: ffi::IntoStringError) -> Self {
        TraceMeError::FFIIntoString(err)
    }
}

impl From<ffi::NulError> for TraceMeError {
    fn from(err: ffi::NulError) -> Self {
        TraceMeError::FFINul(err)
    }
}

impl From<io::Error> for TraceMeError {
    fn from(err: io::Error) -> Self {
        TraceMeError::IO(err)
    }
}

impl From<num::ParseIntError> for TraceMeError {
    fn from(err: num::ParseIntError) -> Self {
        TraceMeError::NumParseInt(err)
    }
}

impl Display for TraceMeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            &TraceMeError::FFIIntoString(ref e) => write!(f, "{}", e),
            &TraceMeError::FFINul(ref e) => write!(f, "{}", e),
            &TraceMeError::IO(ref e) => write!(f, "{}", e),
            &TraceMeError::NumParseInt(ref e) => write!(f, "{}", e),
            &TraceMeError::CFailure => write!(f, "Calling to C failed"),
            &TraceMeError::InvalidFileName(ref n) => write!(f, "Invalid file name: `{}'", n),
            &TraceMeError::TracerAlreadyStarted => write!(f, "Tracer already started"),
            &TraceMeError::TracerNotStarted => write!(f, "Tracer not started"),
            &TraceMeError::TracingNotPermitted =>
                write!(f, "Tracing not permitted: you must be root or {} must contain -1",
                       PERF_PERMS_PATH),
        }
    }
}
