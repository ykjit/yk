//! Errors that can occur during tracing.

use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// Nothing was recorded.
    TraceEmpty,
    /// The trace being recorded was too long and tracing was aborted.
    TraceTooLong,
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::TraceEmpty => write!(f, "Trace empty"),
            InvalidTraceError::TraceTooLong => write!(f, "Trace too long"),
        }
    }
}
