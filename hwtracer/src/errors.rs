use std::fmt::{self, Display, Formatter};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HWTracerError {
    /// hwtracer does not support the requested configuration. Adjust the configuration and retry.
    #[error("Configuration error: {0}")]
    ConfigError(String),
    /// hwtracer has failed in such a way that there is no point retrying the operation.
    #[error("Unrecoverable error: {0}")]
    Unrecoverable(String),
    /// hwtracer has encountered a temporary error: it is possible (though not guaranteed!) that
    /// retrying the same operation again will succeed.
    #[error("Temporary error: {0}")]
    Temporary(TemporaryErrorKind),
}

#[derive(Debug)]
pub enum TemporaryErrorKind {
    /// Memory allocation failed.
    CantAllocate,
    /// The trace buffer has overflowed. Either record a smaller trace or increase the size of the
    /// trace buffer.
    TraceBufferOverflow,
    /// The trace was interrupted.
    TraceInterrupted,
}

impl Display for TemporaryErrorKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            TemporaryErrorKind::CantAllocate => write!(f, "Unable to allocate memory"),
            TemporaryErrorKind::TraceBufferOverflow => write!(f, "Trace buffer overflow"),
            TemporaryErrorKind::TraceInterrupted => write!(f, "Trace interrupted"),
        }
    }
}
