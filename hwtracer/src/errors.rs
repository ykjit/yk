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
    /// Something buffer-related went wrong during tracing.
    TraceBufferOverflow(String),
    /// The trace was interrupted.
    TraceInterrupted,
    /// Perf can't set itself up.
    PerfBusy,
}

impl Display for TemporaryErrorKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            TemporaryErrorKind::CantAllocate => write!(f, "Unable to allocate memory"),
            TemporaryErrorKind::TraceBufferOverflow(s) => write!(f, "{s}"),
            TemporaryErrorKind::TraceInterrupted => write!(f, "Trace interrupted"),
            TemporaryErrorKind::PerfBusy => write!(f, "Perf busy"),
        }
    }
}
