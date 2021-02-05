//! XXX needs a top-level doc comment

use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// An empty trace was recorded.
    EmptyTrace,
    /// Something went wrong in the compiler's tracing code.
    InternalError,
    /// There is no SIR for a location in the trace.
    /// The string inside is the binary symbol name in which the location appears.
    NoSir(String)
}

impl InvalidTraceError {
    /// A helper function to create a `InvalidTraceError::NoSir`.
    pub(crate) fn no_sir(symbol_name: &str) -> Self {
        InvalidTraceError::NoSir(String::from(symbol_name))
    }
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::EmptyTrace => write!(f, "Empty trace"),
            InvalidTraceError::InternalError => write!(f, "Internal tracing error"),
            InvalidTraceError::NoSir(symbol_name) => {
                write!(f, "No SIR for location in symbol: {}", symbol_name)
            }
        }
    }
}
