//! Errors that can occur during tracing.

use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// The trace being recorded was too long and tracing was aborted.
    TraceTooLong,
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::TraceTooLong => write!(f, "Trace too long"),
        }
    }
}
