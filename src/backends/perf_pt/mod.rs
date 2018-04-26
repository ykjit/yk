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

use libc::{pid_t, c_void, size_t, geteuid, malloc, free, c_char, c_int};
use std::fs::File;
use std::iter::Iterator;
use std::io::{self, Read};
use std::ffi::{self, CString, CStr};
use std::os::unix::io::AsRawFd;
use std::ptr;
#[cfg(debug_assertions)]
use std::ops::Drop;
use util::linux_gettid;
use tempfile::NamedTempFile;
use {Tracer, TracerState, Trace, Block};
use errors::HWTracerError;
use std::num::ParseIntError;
use std::error::Error;
use std::fmt::{self, Formatter, Display};

// The sysfs path used to set perf permissions.
const PERF_PERMS_PATH: &str = "/proc/sys/kernel/perf_event_paranoid";

/// An error indicated by a C-level libipt error code.
#[derive(Debug)]
struct LibIPTError(c_int);

impl Display for LibIPTError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Ask libipt for a string representation of the error code.
        let err_str = unsafe { CStr::from_ptr(pt_errstr(self.0)) };
        write!(f, "libipt error: {}", err_str.to_str().unwrap())
    }
}

impl Error for LibIPTError {
    fn description(&self) -> &str {
        "libipt error"
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}


#[repr(C)]
#[allow(dead_code)] // Only C constructs these.
#[derive(PartialEq)]
enum PerfPTCErrorKind {
    Unused,
    Unknown,
    Errno,
    IPT,
}

/// Represents an error occurring in the C code in this backend.
/// Rust code calling C inspects one of these if the return value of a call indicates error.
#[repr(C)]
struct PerfPTCError {
    typ: PerfPTCErrorKind,
    code: c_int,
}

impl PerfPTCError {
    // Creates a new error struct defaulting to an unknown error.
    fn new() -> Self {
        Self {
            typ: PerfPTCErrorKind::Unused,
            code: 0,
        }
    }
}

impl From<PerfPTCError> for HWTracerError {
    fn from(err: PerfPTCError) -> HWTracerError {
        // If this assert crashes out, then we forgot a perf_pt_set_err() somewhere in C code.
        debug_assert!(err.typ != PerfPTCErrorKind::Unused);
        match err.typ {
            PerfPTCErrorKind::Unused => HWTracerError::Unknown,
            PerfPTCErrorKind::Unknown => HWTracerError::Unknown,
            PerfPTCErrorKind::Errno => HWTracerError::Errno(err.code),
            PerfPTCErrorKind::IPT => {
                // Overflow is a special case with its own error type.
                match unsafe {perf_pt_is_overflow_err(err.code)} {
                    true => HWTracerError::HWBufferOverflow,
                    false => HWTracerError::Custom(Box::new(LibIPTError(err.code))),
                }
            }
        }
    }
}

// FFI prototypes.
//
// XXX Rust bug. link_args always reported unused.
// https://github.com/rust-lang/rust/issues/29596#issuecomment-310288094
//
// XXX Cargo bug(?).
// Linker flags in build.rs ignored for the testing target. We must use `link_args` instead.
#[allow(unused_attributes)]
#[link_args="-lipt"]
extern "C" {
    // collect.c
    fn perf_pt_init_tracer(conf: *const PerfPTConf, err: *mut PerfPTCError) -> *mut c_void;
    fn perf_pt_start_tracer(tr_ctx: *mut c_void, trace: *mut PerfPTTrace, err: *mut PerfPTCError) -> bool;
    fn perf_pt_stop_tracer(tr_ctx: *mut c_void, err: *mut PerfPTCError) -> bool;
    fn perf_pt_free_tracer(tr_ctx: *mut c_void, err: *mut PerfPTCError) -> bool;
    // decode.c
    fn perf_pt_init_block_decoder(buf: *const c_void, len: u64, vdso_fd: c_int,
                                  vdso_filename: *const c_char,
                                  decoder_status: *mut c_int,
                                  err: *mut PerfPTCError) -> *mut c_void;
    fn perf_pt_next_block(decoder: *mut c_void, decoder_status: *mut c_int,
                          addr: *mut u64, err: *mut PerfPTCError) -> bool;
    fn perf_pt_free_block_decoder(decoder: *mut c_void);
    // util.c
    fn perf_pt_is_overflow_err(err: c_int) -> bool;
    // libipt
    fn pt_errstr(error_code: c_int) -> *const c_char;
}

// Iterate over the blocks of a PerfPTTrace.
struct PerfPTBlockIterator<'t> {
    decoder: *mut c_void,   // C-level libipt block decoder.
    decoder_status: c_int,  // Stores the current libipt-level status of the above decoder.
    #[allow(dead_code)]     // Rust doesn't know that this exists only to keep the file long enough.
    vdso_tempfile: Option<NamedTempFile>, // VDSO code stored temporarily.
    trace: &'t PerfPTTrace, // The trace we are iterating.
    errored: bool,          // Set to true when an error occurs, thus invalidating the iterator.
}

impl From<io::Error> for HWTracerError {
    fn from(err: io::Error) -> Self {
        HWTracerError::Custom(Box::new(err))
    }
}

impl From<ffi::NulError> for HWTracerError {
    fn from(err: ffi::NulError) -> Self {
        HWTracerError::Custom(Box::new(err))
    }
}

impl From<ParseIntError> for HWTracerError {
    fn from(err: ParseIntError) -> Self {
        HWTracerError::Custom(Box::new(err))
    }
}

impl<'t> PerfPTBlockIterator<'t> {
    // Initialise the block decoder.
    fn init_decoder(&mut self) -> Result<(), HWTracerError> {
        // Make a temp file for the C code to write the VDSO code into.
        //
        // We have to do this because libipt lazily reads the code from the files you load into the
        // image. We store it into `self` to ensure the file lives as long as the iterator.
        let vdso_tempfile = NamedTempFile::new()?;
        // File name of a NamedTempFile should always be valid UTF-8, unwrap() below can't fail.
        let vdso_filename = CString::new(vdso_tempfile.path().to_str().unwrap())?;
        let mut cerr = PerfPTCError::new();
        let decoder = unsafe {
            perf_pt_init_block_decoder(self.trace.buf as *const c_void, self.trace.len,
                                 vdso_tempfile.as_raw_fd(), vdso_filename.as_ptr(),
                                 &mut self.decoder_status, &mut cerr)
        };
        if decoder.is_null() {
            Err(cerr)?
        }

        vdso_tempfile.sync_all()?;
        self.decoder = decoder;
        self.vdso_tempfile = Some(vdso_tempfile);
        Ok(())
    }
}

impl<'t> Drop for PerfPTBlockIterator<'t> {
    fn drop(&mut self) {
        unsafe { perf_pt_free_block_decoder(self.decoder) };
    }
}

impl<'t> Iterator for PerfPTBlockIterator<'t> {
    type Item = Result<Block, HWTracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        // There was an error in a previous iteration.
        if self.errored {
            return None;
        }

        // Lazily initialise the block decoder.
        if self.decoder.is_null() {
            if let Err(e) = self.init_decoder() {
                return Some(Err(e));
            }
        }

        let mut addr = 0;
        let mut cerr = PerfPTCError::new();
        let rv = unsafe {
            perf_pt_next_block(self.decoder, &mut self.decoder_status, &mut addr, &mut cerr)
        };
        if !rv {
            self.errored = true; // This iterator is unusable now.
            return Some(Err(HWTracerError::from(cerr)));
        }
        if addr == 0 {
            None // End of packet stream.
        } else {
            Some(Ok(Block::new(addr)))
        }
    }
}

/// A raw Intel PT trace, obtained via Linux perf.
#[repr(C)]
#[derive(Debug)]
pub struct PerfPTTrace {
    // The trace buffer.
    buf: *mut u8,
    // The length of the trace (in bytes).
    len: u64,
    // `buf`'s allocation size (in bytes), <= `len`.
    capacity: u64,
}

impl PerfPTTrace {
    /// Makes a new trace, initially allocating the specified number of bytes for the PT trace
    /// packet buffer.
    ///
    /// The allocation is automatically freed by Rust when the struct falls out of scope.
    fn new(capacity: size_t) -> Result<Self, HWTracerError> {
        let buf = unsafe { malloc(capacity) as *mut u8 };
        if buf.is_null() {
            return Err(HWTracerError::Unknown);
        }
        Ok(Self {buf: buf, len: 0, capacity: capacity as u64})
    }
}

impl Trace for PerfPTTrace {
    /// Write the raw trace packets into the specified file.
    #[cfg(test)]
    fn to_file(&self, file: &mut File) {
        use std::slice;
        use std::io::prelude::*;

        let slice = unsafe { slice::from_raw_parts(self.buf as *const u8, self.len as usize) };
        file.write_all(slice).unwrap();
    }

    fn iter_blocks<'t: 'i, 'i>(&'t self) -> Box<Iterator<Item=Result<Block, HWTracerError>> + 'i> {
        let itr = PerfPTBlockIterator {
            decoder: ptr::null_mut(),
            decoder_status: 0,
            vdso_tempfile: None,
            trace: self,
            errored: false,
        };
        Box::new(itr)
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.capacity as usize
    }
}

impl Drop for PerfPTTrace {
    fn drop(&mut self) {
        if !self.buf.is_null() {
            unsafe { free(self.buf as *mut c_void) };
        }
    }
}

/// Configures a [`PerfPTTracer`](struct.PerfPTTracer.html).
///
// Must stay in sync with the C code.
#[repr(C)]
pub struct PerfPTConf {
    // Thread ID to trace.
    target_tid: pid_t,
    // Data buffer size, in pages. Must be a power of 2.
    data_bufsize: size_t,
    // AUX buffer size, in pages. Must be a power of 2.
    aux_bufsize: size_t,
    // The initial trace buffer size (in bytes) for new traces.
    new_trace_bufsize: size_t,
}

impl PerfPTConf {
    /// Creates a new configuration with defaults.
    pub fn new() -> Self {
        Self {
            target_tid: linux_gettid(),
            data_bufsize: 64,
            aux_bufsize: 1024,
            new_trace_bufsize: 1024 * 1024, // 1MiB
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

    /// Set the initial trace buffer size (in bytes) for new
    /// [`PerfPTTrace`](struct.PerfPTTrace.html) instances.
    pub fn new_trace_bufsize(mut self, size: usize) -> Self {
        self.new_trace_bufsize = size as size_t;
        self
    }
}

/// A tracer that uses the Linux Perf interface to Intel Processor Trace.
pub struct PerfPTTracer {
    // The configuration for this tracer.
    config: PerfPTConf,
    // Opaque C pointer representing the tracer context.
    tracer_ctx: *mut c_void,
    // The state of the tracer.
    state: TracerState,
    // The trace currently being collected, or `None`.
    trace: Option<Box<PerfPTTrace>>,
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
    /// use hwtracer::Tracer;
    ///
    /// let config = PerfPTTracer::config().data_bufsize(1024).target_tid(12345);
    /// let res = PerfPTTracer::new(config);
    /// if res.is_ok() {
    ///     let mut tracer = res.unwrap();
    ///     // Use the tracer...
    ///     tracer.destroy().unwrap();
    /// } else {
    ///     // CPU doesn't support Intel Processor Trace.
    /// }
    /// ```
    pub fn new(config: PerfPTConf) -> Result<Self, HWTracerError> {
        PerfPTTracer::check_perf_perms()?;
        if !Self::pt_supported() {
            return Err(HWTracerError::NoHWSupport("Intel PT not supported by CPU".into()));
        }

        // Check for inavlid configuration.
        fn power_of_2(v: size_t) -> bool {
            !(v <= 0) && ((v & (v - 1)) == 0)
        }
        if !power_of_2(config.data_bufsize) {
            return Err(HWTracerError::BadConfig(String::from("data_bufsize must be a positive power of 2")));
        }
        if !power_of_2(config.aux_bufsize) {
            return Err(HWTracerError::BadConfig(String::from("aux_bufsize must be a positive power of 2")));
        }


        Ok(Self {
            config: config,
            tracer_ctx: ptr::null_mut(),
            state: TracerState::Stopped,
            trace: None,
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
            return Err(HWTracerError::Permissions(msg));
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
        ebx_out & EBX_BIT != 0
    }

    fn err_if_destroyed(&self) -> Result<(), HWTracerError> {
        if self.state == TracerState::Destroyed {
            return Err(TracerState::Destroyed.as_error());
        }
        Ok(())
    }
}

impl Tracer for PerfPTTracer {
    fn start_tracing(&mut self) -> Result<(), HWTracerError> {
        self.err_if_destroyed()?;
        if self.state == TracerState::Started {
            return Err(TracerState::Started.as_error());
        }

        // At the time of writing, we have to use a fresh Perf file descriptor to ensure traces
        // start with a `PSB+` packet sequence. This is required for correct instruction-level and
        // block-level decoding. Therefore we have to re-initialise for each new tracing session.
        let mut cerr = PerfPTCError::new();
        self.tracer_ctx = unsafe {
            perf_pt_init_tracer(&self.config as *const PerfPTConf, &mut cerr)
        };
        if self.tracer_ctx.is_null() {
            Err(cerr)?
        }

        // It is essential we box the trace now to stop it from moving. If it were to move, then
        // the reference which we pass to C here would become invalid. The interface to
        // `stop_tracing` needs to return a Box<Tracer> anyway, so it's no big deal.
        //
        // Note that the C code will mutate the trace's members directly.
        let mut trace = Box::new(PerfPTTrace::new(self.config.new_trace_bufsize)?);
        let mut cerr = PerfPTCError::new();
        if !unsafe { perf_pt_start_tracer(self.tracer_ctx, &mut *trace, &mut cerr) } {
            Err(cerr)?
        }
        self.state = TracerState::Started;
        self.trace = Some(trace);
        Ok(())
    }

    fn stop_tracing(&mut self) -> Result<Box<Trace>, HWTracerError> {
        self.err_if_destroyed()?;
        if self.state == TracerState::Stopped {
            return Err(TracerState::Stopped.as_error());
        }
        let mut cerr = PerfPTCError::new();
        let rc = unsafe { perf_pt_stop_tracer(self.tracer_ctx, &mut cerr) };
        self.state = TracerState::Stopped;
        if !rc {
            Err(cerr)?
        }

        let mut cerr = PerfPTCError::new();
        if !unsafe { perf_pt_free_tracer(self.tracer_ctx, &mut cerr) } {
            Err(cerr)?
        }
        self.tracer_ctx = ptr::null_mut();

        let ret = self.trace.take().unwrap();
        self.trace = None;
        Ok(ret as Box<Trace>)
    }

    fn destroy(&mut self) -> Result<(), HWTracerError> {
        self.err_if_destroyed()?;
        self.state = TracerState::Destroyed;
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

// Called by C to store a ptxed argument into a Rust Vec.
#[cfg(test)]
#[no_mangle]
pub unsafe extern "C" fn push_ptxed_arg(args: &mut Vec<String>, new_arg: *const c_char) {
    use std::ffi::CStr;
    let new_arg = CStr::from_ptr(new_arg).to_owned().to_str().unwrap().to_owned();
    args.push(new_arg);
}

#[cfg(all(perf_pt_test,test))]
mod tests {
    use super::{PerfPTTracer, Trace, NamedTempFile, c_int, c_char, AsRawFd, CString, c_void,
                PerfPTTrace, PerfPTBlockIterator, ptr, HWTracerError, Tracer};
    use ::{test_helpers, Block};
    use std::process::Command;

    // Makes a `PerfPTTracer` with the default config.
    pub fn default_tracer() -> PerfPTTracer {
        PerfPTTracer::new(PerfPTTracer::config()).unwrap()
    }

    // Arguments for calling perf_pt_append_self_ptxed_raw_args.
    #[repr(C)]
    struct AppendSelfPtxedArgs {
        vdso_fd: c_int,
        vdso_filename: *const c_char,
        ptxed_args: *mut Vec<String>,
    }

    extern "C" {
        fn perf_pt_append_self_ptxed_raw_args(args: *mut c_void) -> bool;
    }

    // Gets the ptxed arguments required to decode a trace for the current process.
    //
    // Returns a vector of arguments and a handle to a temproary file containing the VDSO code. The
    // caller must make sure that this file lives long enough for ptxed to run (temp files are
    // removed when they fall out of scope).
    //
    // See https://github.com/01org/processor-trace/issues/43 for why we append the --iscache-limit
    // argument. Depending on the outcome of that issue, we may be able to remove this later (we
    // should if we can, because it slows tests down).
    pub fn self_ptxed_args(trace_filename: &str) -> (Vec<String>, NamedTempFile) {
        let ptxed_args = vec![
            "--cpu", "auto", "--block-decoder", "--block:end-on-call", "--block:end-on-jump",
            "--block:show-blocks", "--iscache-limit", "0", "--pt", trace_filename];
        let mut ptxed_args = ptxed_args.into_iter().map(|e| String::from(e)).collect();

        // Make a temp file for the VDSO to live in.
        let vdso_tempfile = NamedTempFile::new().unwrap();
        let vdso_filename = CString::new(vdso_tempfile.path().to_str().unwrap()).unwrap();

        // Call C to fill in the rest of the arguments and dump the VDSO to disk.
        let mut call_args = AppendSelfPtxedArgs {
            vdso_fd: vdso_tempfile.as_raw_fd(),
            vdso_filename: vdso_filename.as_ptr(),
            ptxed_args: &mut ptxed_args as *mut Vec<String>,
        };

        let rv = unsafe {
            perf_pt_append_self_ptxed_raw_args(&mut call_args as *mut _ as *mut c_void)
        };
        assert!(rv);
        (ptxed_args, vdso_tempfile)
    }

    // Given a trace, use ptxed to get a vector of block start vaddrs.
    fn get_expected_blocks(trace: &Box<Trace>) -> Vec<Block> {
        // Write the trace out to a temp file so ptxed can decode it.
        let mut fh = NamedTempFile::new().unwrap();
        trace.to_file(&mut fh);
        let (args, _vdso_tempfile) = self_ptxed_args(fh.path().to_str().unwrap());

        let out = Command::new(env!("PTXED"))
                          .args(&args)
                          .output()
                          .unwrap();
        let outstr = String::from_utf8(out.stdout).unwrap();
        if !out.status.success() {
            let errstr = String::from_utf8(out.stderr).unwrap();
            panic!("ptxed failed:\nInvocation----------\n{:?}\n \
                   Stdout\n------\n{}Stderr\n------\n{}",
                   args, outstr, errstr);
        }

        let mut block_start = false;
        let mut block_vaddrs = Vec::new();
        for line in outstr.lines() {
            if line.contains("error") {
                panic!("error line in ptxed output:\n{}", line);
            }
            if block_start {
                // We are expecting an instruction at the start of a block.
                let vaddr_s = line.split_whitespace().next().unwrap();
                let vaddr = u64::from_str_radix(vaddr_s, 16).unwrap();
                block_vaddrs.push(Block::new(vaddr));
                block_start = false;
            } else if line.trim() == "[block]" {
                // The next line will be a block start.
                block_start = true;
                continue;
            }
        }

        block_vaddrs
    }

    // Trace a closure and then decode it and check the block iterator agrees with ptxed.
    fn trace_and_check_blocks<T, F>(mut tracer: T, f: F) where T: Tracer, F: FnOnce() -> u64 {
        let trace = test_helpers::trace_closure(&mut tracer, f);
        let expects = get_expected_blocks(&trace);
        test_helpers::test_expected_blocks(tracer, trace, expects.iter());
    }

    #[test]
    fn test_basic_usage() {
        test_helpers::test_basic_usage(default_tracer());
    }

    #[test]
    fn test_repeated_tracing() {
        test_helpers::test_repeated_tracing(default_tracer());
    }

    #[test]
    fn test_already_started() {
        test_helpers::test_already_started(default_tracer());
    }

    #[test]
    fn test_not_started() {
        test_helpers::test_not_started(default_tracer());
    }

    #[test]
    fn test_use_tracer_after_destroy1() {
        test_helpers::test_use_tracer_after_destroy1(default_tracer());
    }

    #[test]
    fn test_use_tracer_after_destroy2() {
        test_helpers::test_use_tracer_after_destroy2(default_tracer());
    }

    #[test]
    fn test_use_tracer_after_destroy3() {
        test_helpers::test_use_tracer_after_destroy3(default_tracer());
    }

    #[cfg(debug_assertions)]
    #[should_panic]
    #[test]
    fn test_drop_without_destroy() {
        if PerfPTTracer::pt_supported() {
            let tracer = PerfPTTracer::new(PerfPTTracer::config()).unwrap();
            test_helpers::test_drop_without_destroy(tracer);
        } else {
            panic!("ok"); // Because this test expects a panic.
        }
    }

    // Test writing a trace to file.
    #[cfg(debug_assertions)]
    #[test]
    fn test_to_file() {
        use std::fs::File;
        use std::slice;
        use std::io::prelude::*;
        use Trace;

        // Allocate and fill a buffer to make a "trace" from.
        let capacity = 1024;
        let mut trace = PerfPTTrace::new(capacity).unwrap();
        trace.len = capacity as u64;
        let sl = unsafe { slice::from_raw_parts_mut(trace.buf as *mut u8, capacity) };
        for (i, byte) in sl.iter_mut().enumerate() {
            *byte = i as u8;
        }

        // Make the trace and write it to a file.
        let mut fh = NamedTempFile::new().unwrap();
        trace.to_file(&mut fh);
        fh.sync_all().unwrap();

        // Check the resulting file makes sense.
        let file = File::open(fh.path().to_str().unwrap()).unwrap();
        let mut total_bytes = 0;
        for (i, byte) in file.bytes().enumerate() {
            assert_eq!(i as u8, byte.unwrap());
            total_bytes += 1;
        }
        assert_eq!(total_bytes, capacity);
    }

    // Check that our block decoder agrees with the reference implementation in ptxed.
    #[test]
    fn test_block_iterator1() {
        let tracer = default_tracer();
        trace_and_check_blocks(tracer, || test_helpers::work_loop(10));
    }

    // Check that our block decoder agrees ptxed on a (likely) empty trace;
    #[test]
    fn test_block_iterator2() {
        let tracer = default_tracer();
        trace_and_check_blocks(tracer, || test_helpers::work_loop(0));
    }

    // Check that our block decoder deals with traces involving the VDSO correctly.
    #[test]
    fn test_block_iterator3() {
        use libc::{timespec, CLOCK_MONOTONIC, clock_gettime};

        let tracer = default_tracer();
        trace_and_check_blocks(tracer, || {
            let mut res = 0;
            let mut tv = timespec { tv_sec: 0, tv_nsec: 0 };
            for _ in 1..100 {
                // clock_gettime(2) is in the Linux VDSO.
                let rv = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut tv) };
                assert_eq!(rv, 0);
                res += tv.tv_sec;
            }
            res as u64
        });
    }

    // Check that a shorter trace yields fewer blocks.
    #[test]
    fn test_block_iterator4() {
        let tracer1 = default_tracer();
        let tracer2 = default_tracer();
        test_helpers::test_ten_times_as_many_blocks(tracer1, tracer2);
    }

    // Check that our block decoder agrees ptxed on long trace.
    // XXX We use an even higher iteration count once our decoder uses a libipt image cache.
    #[ignore] // Decoding long traces is slow.
    #[test]
    fn test_block_iterator5() {
        let tracer = default_tracer();
        trace_and_check_blocks(tracer, || test_helpers::work_loop(3000));
    }

    // Check that a long trace causes the trace buffer to reallocate.
    #[test]
    fn test_relloc_trace_buf1() {
        let start_bufsize = 512;
        let config = PerfPTTracer::config().new_trace_bufsize(start_bufsize);
        let mut tracer = PerfPTTracer::new(config).unwrap();
        use Tracer;

        tracer.start_tracing().unwrap();
        let res = test_helpers::work_loop(10000);
        let trace = tracer.stop_tracing().unwrap();

        println!("res: {}", res); // Stop over-optimisation.
        assert!(trace.capacity() > start_bufsize);
        println!("CAP: {}", trace.capacity());
        tracer.destroy().unwrap();
    }

    // Check that a block iterator returns none after an error.
    #[test]
    fn test_error_stops_block_iter1() {
        // A zero-sized trace will lead to an error.
        let trace = PerfPTTrace::new(0).unwrap();
        let mut itr = PerfPTBlockIterator {
            decoder: ptr::null_mut(),
            decoder_status: 0,
            vdso_tempfile: None,
            trace: &trace,
            errored: false,
        };

        // First we expect a libipt error.
        match itr.next() {
            Some(Err(HWTracerError::Custom(e))) => assert_eq!(e.description(), "libipt error"),
            _ => panic!(),
        }
        // And now the iterator is invalid, and should return None.
        for _ in 0..128 {
            assert!(itr.next().is_none());
        }
    }

    #[test]
    fn test_config_bad_data_bufsize() {
        match PerfPTTracer::new(PerfPTTracer::config().data_bufsize(3)) {
            Err(HWTracerError::BadConfig(s)) => {
                assert_eq!(s, "data_bufsize must be a positive power of 2");
            },
            _ => panic!(),
        }
    }

    #[test]
    fn test_config_bad_aux_bufsize() {
        match PerfPTTracer::new(PerfPTTracer::config().aux_bufsize(3)) {
            Err(HWTracerError::BadConfig(s)) => {
                assert_eq!(s, "aux_bufsize must be a positive power of 2");
            },
            _ => panic!(),
        }
    }
}
