//! Errors that can occur during tracing.

use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// An empty trace was recorded.
    EmptyTrace,
    /// The trace being recorded was too long and tracing was aborted.
    TraceTooLong,
    /// Something went wrong in the compiler's tracing code.
    InternalError,
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::EmptyTrace => write!(f, "Empty trace"),
            InvalidTraceError::TraceTooLong => write!(f, "Trace too long"),
            InvalidTraceError::InternalError => write!(f, "Internal tracing error"),
        }
    }
}
