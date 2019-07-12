// Copyright 2019 King's College London.
// Created by the Software Development Team <http://soft-dev.org/>.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::{self, Display, Formatter};
use ykpack::DefId;

#[derive(Debug)]
/// Reasons that a trace can be invalidated.
pub enum InvalidTraceError {
    /// There is no SIR for the DefId of a location in the trace.
    NoSir(DefId),
    /// Something went wrong in the compiler's tracing code
    InternalError
}

impl Display for InvalidTraceError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            InvalidTraceError::NoSir(def_id) => write!(f, "No SIR for location: {}", def_id),
            InvalidTraceError::InternalError => write!(f, "Internal tracing error")
        }
    }
}
