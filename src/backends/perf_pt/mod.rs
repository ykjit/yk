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

use libc::{pid_t, c_void, size_t, c_int, geteuid, free};
use errors::HWTracerError;
use std::fs::File;
use std::io::Read;
use Tracer;
use util::linux_gettid;
use std::ptr;
#[cfg(debug_assertions)]
use std::ops::Drop;
use {TracerState, Trace};

// The sysfs path used to set perf permissions.
const PERF_PERMS_PATH: &str = "/proc/sys/kernel/perf_event_paranoid";

// FFI prototypes.
extern "C" {
    fn perf_pt_init_tracer(conf: *const PerfPTConf) -> *const c_void;
    fn perf_pt_start_tracer(tr_ctx: *const c_void) -> c_int;
    fn perf_pt_stop_tracer(tr_ctx: *const c_void, buf: *const *const u8, len: &u64) -> c_int;
    fn perf_pt_free_tracer(tr_ctx: *const c_void) -> c_int;
}

/// A raw Intel PT trace, obtained via Linux perf.
pub struct PerfPTTrace {
    buf: *const u8,
    #[allow(dead_code)]
    len: u64,
}

impl PerfPTTrace {
    /// Makes a new trace from a raw pointer and a size.
    ///
    /// The `buf` argument is assumed to have been allocated on the heap using malloc(3). `len`
    /// must be less than or equal to the allocated size.
    ///
    /// The allocation is automatically freed by Rust when the struct falls out of scope.
    fn from_buf(buf: *const u8, len: u64) -> Self {
        Self {buf: buf, len: len}
    }
}

impl Trace for PerfPTTrace {
    /// Write the raw trace packets into the specified file.
    ///
    /// This can be useful for developers who want to use (e.g.) the pt utility to inspect the raw
    /// packet stream.
    #[cfg(debug_assertions)]
    fn to_file(&self, filename: &str) {
        use std::slice;
        use std::fs::File;
        use std::io::prelude::*;

        let mut f = File::create(filename).unwrap();
        let slice = unsafe { slice::from_raw_parts(self.buf, self.len as usize) };
        f.write(slice).unwrap();
    }
}


/// Once a PerfPTTrace is brought into existence, we say the instance owns the C-level allocation.
/// When the it falls out of scope, free up the memory.
impl Drop for PerfPTTrace {
    fn drop(&mut self) {
        if self.buf != ptr::null() {
            unsafe { free(self.buf as *mut c_void) };
        }
    }
}


// Struct used to communicate a tracing configuration to the C code. Must
// stay in sync with the C code.
#[repr(C)]
pub struct PerfPTConf {
    /// Thread ID to trace.
    target_tid: pid_t,
    /// Data buffer size, in pages. Must be a power of 2.
    data_bufsize: size_t,
    /// AUX buffer size, in pages. Must be a power of 2.
    aux_bufsize: size_t,
}

/// Configures a PerfPTTracer.
impl PerfPTConf {
    /// Creates a new configuration with defaults.
    pub fn new() -> Self {
        Self {
            target_tid: linux_gettid(),
            data_bufsize: 64,
            aux_bufsize: 1024,
        }
    }

    /// Select which thread to trace.
    ///
    /// By default, the current thread is traced.
    ///
    /// The `tid` argument is a Linux thread ID. Note that Linux re-uses the `pid_t` type, but that
    /// PIDs are distinct from TIDs.
    pub fn target_tid(mut self, pid: pid_t) -> Self {
        self.target_tid = pid;
        self
    }

    /// Set the PT data buffer size (in pages).
    pub fn data_bufsize(mut self, size: usize) -> Self {
        self.data_bufsize = size as size_t;
        self
    }

    /// Set the PT AUX buffer size (in pages).
    pub fn aux_bufsize(mut self, size: usize) -> Self {
        self.aux_bufsize = size as size_t;
        self
    }

}

/// A tracer that uses the Linux Perf interface to Intel Processor Trace.
pub struct PerfPTTracer {
    /// Opaque C pointer representing the tracer context.
    tracer_ctx: *const c_void,
    /// The state of the tracer.
    state: TracerState,
}

impl PerfPTTracer {
    /// Create a new tracer using the specified `PerfPTConf`.
    ///
    /// Returns `Err` if the CPU doesn't support Intel Processor Trace.
    ///
    /// # Example
    ///
    /// ```
    /// use hwtracer::backends::PerfPTTracer;
    ///
    /// let config = PerfPTTracer::config().data_bufsize(1024).target_tid(12345);
    /// let res = PerfPTTracer::new(config);
    /// if res.is_ok() {
    ///     let tracer = res.unwrap();
    ///     // Use the tracer...
    /// } else {
    ///     // CPU doesn't support Intel Processor Trace.
    /// }
    /// ```
    pub fn new(config: PerfPTConf) -> Result<Self, HWTracerError> {
        PerfPTTracer::check_perf_perms()?;
        if !Self::pt_supported() {
            return Err(HWTracerError::HardwareSupport("Intel PT not supported by CPU".into()));
        }

        let ctx = unsafe { perf_pt_init_tracer(&config as *const PerfPTConf) };
        if ctx == ptr::null() {
            return Err(HWTracerError::CFailure);
        }

        Ok(Self {
            tracer_ctx: ctx,
            state: TracerState::Stopped,
        })
    }

    /// Utility function to get a default config.
    pub fn config() -> PerfPTConf {
        PerfPTConf::new()
    }

    fn check_perf_perms() -> Result<(), HWTracerError> {
        if unsafe { geteuid() } == 0 {
            // Root can always trace.
            return Ok(());
        }

        let mut f = File::open(&PERF_PERMS_PATH)?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        let perm = buf.trim().parse::<i8>()?;
        if perm != -1 {
            let msg = format!("Tracing not permitted: you must be root or {} must contain -1",
                           PERF_PERMS_PATH);
            return Err(HWTracerError::TracingNotPermitted(msg));
        }

        Ok(())
    }

    /// Checks if the CPU supports Intel Processor Trace.
    fn pt_supported() -> bool {
        const LEAF: u32 = 0x07;
        const SUBPAGE: u32 = 0x0;
        const EBX_BIT: u32 = 1 << 25;
        let ebx_out: u32;

        unsafe {
            asm!(r"
                  mov $1, %eax;
                  mov $2, %ecx;
                  cpuid;"
                : "={ebx}" (ebx_out)
                : "i" (LEAF), "i" (SUBPAGE)
                : "eax", "ecx", "edx"
                : "volatile");
        }

        if ebx_out & (EBX_BIT) != 0 {
            return true;
        }
        false
    }

    fn err_if_destroyed(&self) -> Result<(), HWTracerError> {
        if self.state == TracerState::Destroyed {
            return Err(HWTracerError::TracerDestroyed);
        }
        Ok(())
    }
}

impl Tracer for PerfPTTracer {
    fn start_tracing(&mut self) -> Result<(), HWTracerError> {
        self.err_if_destroyed()?;
        if self.state == TracerState::Started {
            return Err(HWTracerError::TracerAlreadyStarted);
        }

        if unsafe { perf_pt_start_tracer(self.tracer_ctx) } == -1 {
            return Err(HWTracerError::CFailure);
        }
        self.state = TracerState::Started;
        Ok(())
    }

    fn stop_tracing(&mut self) -> Result<Box<Trace>, HWTracerError> {
        self.err_if_destroyed()?;
        if self.state == TracerState::Stopped {
            return Err(HWTracerError::TracerNotStarted);
        }

        let buf = ptr::null() as *const u8;
        let len = 0;
        let rc = unsafe {
            perf_pt_stop_tracer(self.tracer_ctx, &buf, &len)
        };
        self.state = TracerState::Stopped;
        if rc == -1 {
            return Err(HWTracerError::CFailure);
        }
        let trace = PerfPTTrace::from_buf(buf, len);
        Ok(Box::new(trace) as Box<Trace>)
    }

    fn destroy(&mut self) -> Result<(), HWTracerError> {
        self.err_if_destroyed()?;
        self.state = TracerState::Destroyed;
        let res = unsafe { perf_pt_free_tracer(self.tracer_ctx) };
        if res != 0 {
            return Err(HWTracerError::CFailure);
        }
        Ok(())
    }
}

#[cfg(debug_assertions)]
impl Drop for PerfPTTracer {
    fn drop(&mut self) {
        if self.state != TracerState::Destroyed {
            panic!("PerfPTTracer dropped with no call to destroy()");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PerfPTTracer;
    use ::test_helpers;

    fn run_test_helper<F>(f: F) where F: Fn(PerfPTTracer) {
        let res = PerfPTTracer::new(PerfPTTracer::config());
        // Only run the test if the CPU supports Intel PT.
        if let Ok(tracer) = res {
            f(tracer);
        }
    }

    #[test]
    fn test_basic_usage() {
        run_test_helper(test_helpers::test_basic_usage);
    }

    #[test]
    fn test_repeated_tracing() {
        run_test_helper(test_helpers::test_repeated_tracing);
    }

    #[test]
    fn test_already_started() {
        run_test_helper(test_helpers::test_already_started);
    }

    #[test]
    fn test_not_started() {
        run_test_helper(test_helpers::test_not_started);
    }

    #[test]
    fn test_use_tracer_after_destroy1() {
        run_test_helper(test_helpers::test_use_tracer_after_destroy1);
    }

    #[test]
    fn test_use_tracer_after_destroy2() {
        run_test_helper(test_helpers::test_use_tracer_after_destroy1);
    }

    #[test]
    fn test_use_tracer_after_destroy3() {
        run_test_helper(test_helpers::test_use_tracer_after_destroy1);
    }

    #[cfg(debug_assertions)]
    #[should_panic]
    #[test]
    fn test_drop_without_destroy() {
        run_test_helper(test_helpers::test_drop_without_destroy);
    }

    /// Test writing a trace to file.
    #[cfg(debug_assertions)]
    #[test]
    fn test_to_file() {
        use std::fs::File;
        use std::slice;
        use std::io::prelude::*;
        use libc::malloc;
        use super::PerfPTTrace;
        use Trace;

        // Allocate and fill a buffer to make a "trace" from.
        let size = 33;
        let buf = unsafe { malloc(size) as *mut u8 };
        let sl = unsafe { slice::from_raw_parts_mut(buf, size) };
        for (i, byte) in sl.iter_mut().enumerate() {
            *byte = i as u8;
        }

        // Make the trace and write it to a file.
        let filename = String::from("test_to_file.ptt");
        let trace = PerfPTTrace::from_buf(buf, size as u64);
        trace.to_file(&filename);

        // Check the resulting file makes sense.
        let file = File::open(&filename).unwrap();
        let mut total_bytes = 0;
        for (i, byte) in file.bytes().enumerate() {
            assert_eq!(i as u8, byte.unwrap());
            total_bytes += 1;
        }
        assert_eq!(total_bytes, size);
    }
}
