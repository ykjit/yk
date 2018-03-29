// Copyright (c) 2017-2018 King's College London
// created by the Software Development Team <http://soft-dev.org/>
//
// The Universal Permissive License (UPL), Version 1.0
//
// Subject to the condition set forth below, permission is hereby granted to any
// person obtaining a copy of this software, associated documentation and/or
// data (collectively the "Software"), free of charge and under any and all
// copyright rights in the Software, and any and all patent rights owned or
// freely licensable by each licensor hereunder covering either (i) the
// unmodified Software as contributed to or provided by such licensor, or (ii)
// the Larger Works (as defined below), to deal in both
//
// (a) the Software, and
// (b) any piece of software and/or hardware listed in the lrgrwrks.txt file
// if one is included with the Software (each a "Larger Work" to which the Software
// is contributed by such licensors),
//
// without restriction, including without limitation the rights to copy, create
// derivative works of, display, perform, and distribute the Software and make,
// use, sell, offer for sale, import, export, have made, and have sold the
// Software and the Larger Work(s), and to sublicense the foregoing rights on
// either these or other terms.
//
// This license is subject to the following condition: The above copyright
// notice and either this complete permission notice or at a minimum a reference
// to the UPL must be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::fmt::{self, Formatter, Display};
use std::error::Error;
use TracerState;

#[derive(Debug)]
pub enum HWTracerError {
    HWBufferOverflow,         // The trace buffer being used by the hardware overflowed.
                              // This is considered a non-fatal error since retrying the tracing
                              // may succeed.
    NoHWSupport(String),      // The hardware doesn't support a required feature. Not fatal for the
                              // same reason as `Permissions`. This may be non-fatal depending
                              // upon whether the consumer could (e.g.) try a different backend.
    Permissions(String),      // Tracing is not permitted using this backend.
    CFailure,                 // Something went wrong in C code.
                              // ^ XXX will be replaced with an errno mechanism.
    TracerState(TracerState), // The tracer is in the wrong state to do the requested task.
    Custom(Box<Error>),       // All other errors can be nested here, however, don't rely on this
                              // for performance since the `Box` incurs a runtime cost.
}

impl Display for HWTracerError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            HWTracerError::HWBufferOverflow => write!(f, "Hardware trace buffer overflow"),
            HWTracerError::NoHWSupport(ref s) => write!(f, "{}", s),
            HWTracerError::Permissions(ref s) => write!(f, "{}", s),
            HWTracerError::CFailure => write!(f, "C failure"),
            HWTracerError::TracerState(ref s) => write!(f, "Tracer in wrong state: {}", s),
            HWTracerError::Custom(ref bx) => write!(f, "{}", bx),
        }
    }
}

impl Error for HWTracerError {
    fn description(&self) -> &str {
        "hwtracer error"
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            HWTracerError::HWBufferOverflow => None,
            HWTracerError::NoHWSupport(_) => None,
            HWTracerError::Permissions(_) => None,
            HWTracerError::TracerState(_) => None,
            HWTracerError::CFailure => None,
            HWTracerError::Custom(ref bx) => Some(bx.as_ref()),
        }
    }
}
