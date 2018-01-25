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

use std::{io, ffi, num};
use std::fmt::{self, Formatter, Display};

#[derive(Debug)]
pub enum HWTracerError {
    // Wrapped errors from elsewhere.
    FFIIntoString(ffi::IntoStringError),
    FFINul(ffi::NulError),
    IO(io::Error),
    NumParseInt(num::ParseIntError),
    // Our own errors.
    CFailure,
    ElfError(String),
    HardwareSupport(String),
    InvalidFileName(String),
    TracerAlreadyStarted,
    TracerNotStarted,
    TracingNotPermitted(String),
}

impl From<ffi::IntoStringError> for HWTracerError {
    fn from(err: ffi::IntoStringError) -> Self {
        HWTracerError::FFIIntoString(err)
    }
}

impl From<ffi::NulError> for HWTracerError {
    fn from(err: ffi::NulError) -> Self {
        HWTracerError::FFINul(err)
    }
}

impl From<io::Error> for HWTracerError {
    fn from(err: io::Error) -> Self {
        HWTracerError::IO(err)
    }
}

impl From<num::ParseIntError> for HWTracerError {
    fn from(err: num::ParseIntError) -> Self {
        HWTracerError::NumParseInt(err)
    }
}

impl Display for HWTracerError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            HWTracerError::FFIIntoString(ref e) => write!(f, "{}", e),
            HWTracerError::FFINul(ref e) => write!(f, "{}", e),
            HWTracerError::IO(ref e) => write!(f, "{}", e),
            HWTracerError::NumParseInt(ref e) => write!(f, "{}", e),
            HWTracerError::HardwareSupport(ref m) => write!(f, "Hardware support: {}", m),
            HWTracerError::CFailure => write!(f, "Calling to C failed"),
            HWTracerError::ElfError(ref m) => write!(f, "ELF error: {}", m),
            HWTracerError::InvalidFileName(ref n) => write!(f, "Invalid file name: `{}'", n),
            HWTracerError::TracerAlreadyStarted => write!(f, "Tracer already started"),
            HWTracerError::TracerNotStarted => write!(f, "Tracer not started"),
            HWTracerError::TracingNotPermitted(ref m) => write!(f, "{}", m),
        }
    }
}
