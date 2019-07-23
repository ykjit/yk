// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::debug;
use std::fmt::{self, Display, Formatter};
use ykpack::DefId;

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// There is no SIR for the DefId of a location in the trace.
    /// The second element is the definition path string.
    NoSir(DefId, String),
    /// Something went wrong in the compiler's tracing code
    InternalError
}

impl InvalidTraceError {
    /// A helper function to create a `InvalidTraceError::NoSir`.
    pub fn no_sir(def_id: &DefId) -> Self {
        return InvalidTraceError::NoSir(
            def_id.clone(),
            String::from(debug::def_path(def_id).unwrap_or("<unknown>"))
        );
    }
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::NoSir(def_id, def_path) => {
                write!(f, "No SIR for location: {} ({})", def_id, def_path)
            }
            InvalidTraceError::InternalError => write!(f, "Internal tracing error")
        }
    }
}
