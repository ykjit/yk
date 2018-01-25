// Copyright (c) 2018 King's College London
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

use libc::{c_int, uintptr_t};
#[cfg(target_os = "linux")]
use libc::pid_t;
use ::errors::HWTracerError;

// FFI prototypes.
extern "C" {
    fn hwtracer_exec_base(addr: *const uintptr_t) -> c_int;
    #[cfg(target_os = "linux")]
    fn hwtracer_linux_gettid() -> pid_t;
}

/// Get the relocated virtual start address of the executable code for the current program.
pub fn exec_base() -> Result<usize, HWTracerError> {
    let mut addr: uintptr_t = 0;
    match unsafe { hwtracer_exec_base(&mut addr as *const uintptr_t) } {
        0 => Err(HWTracerError::ElfError(String::from("Failed to get executable base address"))),
        1 => Ok(addr as usize),
        _ => unreachable!(),
    }
}

/// Get the thread ID of the current thread.
#[cfg(target_os = "linux")]
pub fn linux_gettid() -> pid_t {
    unsafe { hwtracer_linux_gettid() }
}

#[cfg(all(test, target_os = "linux"))]
mod test_linux {
    use super::exec_base;

    use libc::getpid;
    use std::path::PathBuf;
    use std::fs::File;
    use std::io::BufReader;
    use std::io::BufRead;
    use std::env::current_exe;

    /// Check we can get the exec_base() address and that the procfs map file agrees.
    #[test]
    fn test_linux_exec_base() {
        let got_base = exec_base().unwrap();

        let mut path = PathBuf::from("/proc");
        path.push(unsafe {getpid()}.to_string());
        path.push("maps");
        let file = File::open(path).unwrap();

        // Search for the executable base address.
        let exe = current_exe().unwrap();
        let mut expect_base: usize = 0;
        let rdr = BufReader::new(file);
        for line in rdr.lines() {
            let line = line.unwrap();
            let mut parts = line.split_whitespace();
            let addrs = parts.next().unwrap();
            let flags = parts.next().unwrap();
            let path = parts.nth(3).unwrap();

            if path == exe.to_str().unwrap() && flags == "r-xp" {
                let mut addrs_parts = addrs.split('-');
                expect_base = usize::from_str_radix(addrs_parts.next().unwrap(), 16).unwrap();
                break;
            }
        }
        assert_eq!(got_base, expect_base);
    }
}

#[cfg(test)]
mod test {
    use super::exec_base;

    /// Check we can get the exec_base() address.
    #[test]
    fn test_exec_base() {
        let got_base = exec_base().unwrap();
        assert_ne!(got_base, 0); // Code will never be mapped at 0.
    }
}
