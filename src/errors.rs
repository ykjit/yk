use std::ffi;
use std::fmt::{self, Formatter, Display};

#[derive(Debug)]
pub enum TraceMeError {
    // Wrapped errors from elsewhere.
    FFIIntoString(ffi::IntoStringError),
    FFINul(ffi::NulError),
    // Our own errors.
    CFailure,
    EmptyFileName,
    TracerAlreadyStarted,
    TracerNotStarted,
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

impl Display for TraceMeError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            &TraceMeError::FFIIntoString(ref e) => write!(f, "{}", e),
            &TraceMeError::FFINul(ref e) => write!(f, "{}", e),
            &TraceMeError::CFailure => write!(f, "Calling to C failed"),
            &TraceMeError::EmptyFileName => write!(f, "Empty file name"),
            &TraceMeError::TracerAlreadyStarted => write!(f, "Tracer already started"),
            &TraceMeError::TracerNotStarted => write!(f, "Tracer not started"),
        }
    }
}
