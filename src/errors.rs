use std::{io, ffi, num};
use std::fmt::{self, Formatter, Display};

#[derive(Debug)]
pub enum TraceMeError {
    // Wrapped errors from elsewhere.
    FFIIntoString(ffi::IntoStringError),
    FFINul(ffi::NulError),
    IO(io::Error),
    NumParseInt(num::ParseIntError),
    // Our own errors.
    CFailure,
    ElfError(String),
    HardwareSupport(String),
    InvalidFileName(String),
    TracerAlreadyStarted,
    TracerNotStarted,
    TracingNotPermitted(String),
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
            &TraceMeError::HardwareSupport(ref m) => write!(f, "Hardware support: {}", m),
            &TraceMeError::CFailure => write!(f, "Calling to C failed"),
            &TraceMeError::ElfError(ref m) => write!(f, "ELF error: {}", m),
            &TraceMeError::InvalidFileName(ref n) => write!(f, "Invalid file name: `{}'", n),
            &TraceMeError::TracerAlreadyStarted => write!(f, "Tracer already started"),
            &TraceMeError::TracerNotStarted => write!(f, "Tracer not started"),
            &TraceMeError::TracingNotPermitted(ref m) => write!(f, "{}", m),
        }
    }
}
