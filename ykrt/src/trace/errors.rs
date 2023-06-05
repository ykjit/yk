//! Errors that can occur during tracing.

use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// An empty trace was recorded.
    EmptyTrace,
    /// Something went wrong in the compiler's tracing code.
    InternalError,
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::EmptyTrace => write!(f, "Empty trace"),
            InvalidTraceError::InternalError => write!(f, "Internal tracing error"),
        }
    }
}
