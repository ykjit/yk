use super::PerfPTConfig;
use crate::errors::HWTracerError;
use crate::{Block, ThreadTracer, Trace, Tracer};
use libc::{c_char, c_int, c_void, free, geteuid, malloc, size_t};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, Read};
use std::iter::Iterator;
use std::num::ParseIntError;
#[cfg(debug_assertions)]
use std::ops::Drop;
use std::os::unix::io::AsRawFd;
use std::ptr;
use std::{
    env,
    ffi::{self, CStr, CString},
};
use tempfile::NamedTempFile;

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

    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}

#[repr(C)]
#[allow(dead_code)] // Only C constructs these.
#[derive(Eq, PartialEq)]
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
                match unsafe { perf_pt_is_overflow_err(err.code) } {
                    true => HWTracerError::HWBufferOverflow,
                    false => HWTracerError::Custom(Box::new(LibIPTError(err.code))),
                }
            }
        }
    }
}

// FFI prototypes.
extern "C" {
    // collect.c
    fn perf_pt_init_tracer(conf: *const PerfPTConfig, err: *mut PerfPTCError) -> *mut c_void;
    fn perf_pt_start_tracer(
        tr_ctx: *mut c_void,
        trace: *mut PerfPTTrace,
        err: *mut PerfPTCError,
    ) -> bool;
    fn perf_pt_stop_tracer(tr_ctx: *mut c_void, err: *mut PerfPTCError) -> bool;
    fn perf_pt_free_tracer(tr_ctx: *mut c_void, err: *mut PerfPTCError) -> bool;
    // decode.c
    fn perf_pt_init_block_decoder(
        buf: *const c_void,
        len: u64,
        vdso_fd: c_int,
        vdso_filename: *const c_char,
        decoder_status: *mut c_int,
        err: *mut PerfPTCError,
        current_exe: *const c_char,
    ) -> *mut c_void;
    fn perf_pt_next_block(
        decoder: *mut c_void,
        decoder_status: *mut c_int,
        addr: *mut u64,
        len: *mut u64,
        err: *mut PerfPTCError,
    ) -> bool;
    fn perf_pt_free_block_decoder(decoder: *mut c_void);
    // util.c
    fn perf_pt_is_overflow_err(err: c_int) -> bool;
    // libipt
    fn pt_errstr(error_code: c_int) -> *const c_char;
}

// Iterate over the blocks of a PerfPTTrace.
struct PerfPTBlockIterator<'t> {
    decoder: *mut c_void,  // C-level libipt block decoder.
    decoder_status: c_int, // Stores the current libipt-level status of the above decoder.
    #[allow(dead_code)] // Rust doesn't know that this exists only to keep the file long enough.
    vdso_tempfile: Option<NamedTempFile>, // VDSO code stored temporarily.
    trace: &'t PerfPTTrace, // The trace we are iterating.
    errored: bool,         // Set to true when an error occurs, thus invalidating the iterator.
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
            perf_pt_init_block_decoder(
                self.trace.buf.0 as *const c_void,
                self.trace.len,
                vdso_tempfile.as_raw_fd(),
                vdso_filename.as_ptr(),
                &mut self.decoder_status,
                &mut cerr,
                // FIXME: current_exe() isn't reliable. We should find another way to do this.
                CString::new(env::current_exe().unwrap().to_str().unwrap())
                    .unwrap()
                    .as_c_str()
                    .as_ptr() as *const c_char,
            )
        };
        if decoder.is_null() {
            return Err(cerr.into());
        }

        vdso_tempfile.as_file().sync_all()?;
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
                self.errored = true;
                return Some(Err(e));
            }
        }

        let mut first_instr = 0;
        let mut last_instr = 0;
        let mut cerr = PerfPTCError::new();
        let rv = unsafe {
            perf_pt_next_block(
                self.decoder,
                &mut self.decoder_status,
                &mut first_instr,
                &mut last_instr,
                &mut cerr,
            )
        };
        if !rv {
            self.errored = true; // This iterator is unusable now.
            return Some(Err(HWTracerError::from(cerr)));
        }
        if first_instr == 0 {
            None // End of packet stream.
        } else {
            Some(Ok(Block::new(first_instr, last_instr)))
        }
    }
}

/// A wrapper around a manually malloc/free'd buffer for holding an Intel PT trace. We've split
/// this out from PerfPTTrace so that we can mark just this raw pointer as `unsafe Send`.
#[repr(C)]
#[derive(Debug)]
struct PerfPTTraceBuf(*mut u8);

/// We need to be able to transfer `PerfPTTraceBuf`s between threads to allow background
/// compilation. However, `PerfPTTraceBuf` wraps a raw pointer, which is not `Send`, so nor is
/// `PerfPTTraceBuf`. As long as we take great care to never: a) give out copies of the pointer to
/// the wider world, or b) dereference the pointer when we shouldn't, then we can manually (and
/// unsafely) mark the struct as being Send.
unsafe impl Send for PerfPTTrace {}

/// An Intel PT trace, obtained via Linux perf.
#[repr(C)]
#[derive(Debug)]
pub struct PerfPTTrace {
    // The trace buffer.
    buf: PerfPTTraceBuf,
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
        Ok(Self {
            buf: PerfPTTraceBuf(buf),
            len: 0,
            capacity: capacity as u64,
        })
    }
}

impl Trace for PerfPTTrace {
    /// Write the raw trace packets into the specified file.
    #[cfg(test)]
    fn to_file(&self, file: &mut File) {
        use std::io::prelude::*;
        use std::slice;

        let slice = unsafe { slice::from_raw_parts(self.buf.0 as *const u8, self.len as usize) };
        file.write_all(slice).unwrap();
    }

    fn iter_blocks<'t: 'i, 'i>(
        &'t self,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + 'i> {
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
        if !self.buf.0.is_null() {
            unsafe { free(self.buf.0 as *mut c_void) };
        }
    }
}

#[derive(Debug)]
pub struct PerfPTTracer {
    config: PerfPTConfig,
}

impl PerfPTTracer {
    pub(super) fn new(config: PerfPTConfig) -> Result<Self, HWTracerError>
    where
        Self: Sized,
    {
        // Check for inavlid configuration.
        fn power_of_2(v: size_t) -> bool {
            (v & (v - 1)) == 0
        }
        if !power_of_2(config.data_bufsize) {
            return Err(HWTracerError::BadConfig(String::from(
                "data_bufsize must be a positive power of 2",
            )));
        }
        if !power_of_2(config.aux_bufsize) {
            return Err(HWTracerError::BadConfig(String::from(
                "aux_bufsize must be a positive power of 2",
            )));
        }

        Self::check_perf_perms()?;
        Ok(Self { config })
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
            let msg = format!(
                "Tracing not permitted: you must be root or {} must contain -1",
                PERF_PERMS_PATH
            );
            return Err(HWTracerError::Permissions(msg));
        }

        Ok(())
    }
}

impl Tracer for PerfPTTracer {
    fn thread_tracer(&self) -> Box<dyn ThreadTracer> {
        Box::new(PerfPTThreadTracer::new(self.config.clone()))
    }
}

/// A tracer that uses the Linux Perf interface to Intel Processor Trace.
pub struct PerfPTThreadTracer {
    // The configuration for this tracer.
    config: PerfPTConfig,
    // Opaque C pointer representing the tracer context.
    tracer_ctx: *mut c_void,
    // The state of the tracer.
    is_tracing: bool,
    // The trace currently being collected, or `None`.
    trace: Option<Box<PerfPTTrace>>,
}

impl PerfPTThreadTracer {
    fn new(config: PerfPTConfig) -> Self {
        Self {
            config,
            tracer_ctx: ptr::null_mut(),
            is_tracing: false,
            trace: None,
        }
    }
}

impl Default for PerfPTThreadTracer {
    fn default() -> Self {
        PerfPTThreadTracer::new(PerfPTConfig::default())
    }
}

impl Drop for PerfPTThreadTracer {
    fn drop(&mut self) {
        if self.is_tracing {
            // If we haven't stopped the tracer already, stop it now.
            self.stop_tracing().unwrap();
        }
    }
}

impl ThreadTracer for PerfPTThreadTracer {
    fn start_tracing(&mut self) -> Result<(), HWTracerError> {
        if self.is_tracing {
            return Err(HWTracerError::AlreadyTracing);
        }

        // At the time of writing, we have to use a fresh Perf file descriptor to ensure traces
        // start with a `PSB+` packet sequence. This is required for correct instruction-level and
        // block-level decoding. Therefore we have to re-initialise for each new tracing session.
        let mut cerr = PerfPTCError::new();
        self.tracer_ctx =
            unsafe { perf_pt_init_tracer(&self.config as *const PerfPTConfig, &mut cerr) };
        if self.tracer_ctx.is_null() {
            return Err(cerr.into());
        }

        // It is essential we box the trace now to stop it from moving. If it were to move, then
        // the reference which we pass to C here would become invalid. The interface to
        // `stop_tracing` needs to return a Box<Tracer> anyway, so it's no big deal.
        //
        // Note that the C code will mutate the trace's members directly.
        let mut trace = Box::new(PerfPTTrace::new(self.config.initial_trace_bufsize)?);
        let mut cerr = PerfPTCError::new();
        if !unsafe { perf_pt_start_tracer(self.tracer_ctx, &mut *trace, &mut cerr) } {
            return Err(cerr.into());
        }
        self.is_tracing = true;
        self.trace = Some(trace);
        Ok(())
    }

    fn stop_tracing(&mut self) -> Result<Box<dyn Trace>, HWTracerError> {
        if !self.is_tracing {
            return Err(HWTracerError::AlreadyStopped);
        }
        let mut cerr = PerfPTCError::new();
        let rc = unsafe { perf_pt_stop_tracer(self.tracer_ctx, &mut cerr) };
        self.is_tracing = false;
        if !rc {
            return Err(cerr.into());
        }

        let mut cerr = PerfPTCError::new();
        if !unsafe { perf_pt_free_tracer(self.tracer_ctx, &mut cerr) } {
            return Err(cerr.into());
        }
        self.tracer_ctx = ptr::null_mut();

        let ret = self.trace.take().unwrap();
        self.trace = None;
        Ok(ret as Box<dyn Trace>)
    }
}

// Called by C to store a ptxed argument into a Rust Vec.
#[cfg(test)]
#[no_mangle]
pub unsafe extern "C" fn push_ptxed_arg(args: &mut Vec<String>, new_arg: *const c_char) {
    let new_arg = CStr::from_ptr(new_arg)
        .to_owned()
        .to_str()
        .unwrap()
        .to_owned();
    args.push(new_arg);
}

#[cfg(all(perf_pt_test, test))]
mod tests {
    use super::PerfPTCError;
    use super::{
        c_int, ptr, size_t, AsRawFd, HWTracerError, NamedTempFile, PerfPTBlockIterator,
        PerfPTConfig, PerfPTThreadTracer, PerfPTTrace, ThreadTracer, Trace,
    };
    use crate::backends::{BackendConfig, TracerBuilder};
    use crate::{test_helpers, Block};
    use phdrs::{PF_X, PT_LOAD};
    use std::convert::TryFrom;
    use std::env;
    use std::process::Command;

    extern "C" {
        fn dump_vdso(fd: c_int, vaddr: u64, len: size_t, err: &PerfPTCError) -> bool;
    }

    const VDSO_FILENAME: &str = "linux-vdso.so.1";

    // Gets the ptxed arguments required to decode a trace for the current process.
    //
    // Returns a vector of arguments and a handle to a temproary file containing the VDSO code. The
    // caller must make sure that this file lives long enough for ptxed to run (temp files are
    // removed when they fall out of scope).
    fn self_ptxed_args(trace_filename: &str) -> (Vec<String>, NamedTempFile) {
        let ptxed_args = vec![
            "--cpu",
            "auto",
            "--block-decoder",
            "--block:end-on-call",
            "--block:end-on-jump",
            "--block:show-blocks",
            "--pt",
            trace_filename,
        ];
        let mut ptxed_args: Vec<String> = ptxed_args.into_iter().map(|e| String::from(e)).collect();

        // Make a temp file to dump the VDSO code into. This is necessary because ptxed cannot read
        // code from a virtual address: it can only load from file.
        let vdso_tempfile = NamedTempFile::new().unwrap();

        let exe = env::current_exe().unwrap();
        for obj in phdrs::objects() {
            let obj_name = obj.name().to_str().unwrap();
            let mut filename = if cfg!(target_os = "linux") && obj_name == "" {
                exe.to_str().unwrap()
            } else {
                obj_name
            };

            for hdr in obj.iter_phdrs() {
                if hdr.type_() != PT_LOAD || hdr.flags() & PF_X.0 == 0 {
                    continue; // Only look at loadable and executable segments.
                }

                let vaddr = obj.addr() + hdr.vaddr();
                let offset;

                if filename == VDSO_FILENAME {
                    let cerr = PerfPTCError::new();
                    if !unsafe {
                        dump_vdso(
                            vdso_tempfile.as_raw_fd(),
                            vaddr,
                            size_t::try_from(hdr.memsz()).unwrap(),
                            &cerr,
                        )
                    } {
                        panic!("failed to dump vdso");
                    }
                    filename = vdso_tempfile.path().to_str().unwrap();
                    offset = 0;
                } else {
                    offset = hdr.offset();
                }

                let raw_arg = format!(
                    "{}:0x{:x}-0x{:x}:0x{:x}",
                    filename,
                    offset,
                    hdr.offset() + hdr.filesz(),
                    vaddr
                );
                ptxed_args.push("--raw".to_owned());
                ptxed_args.push(raw_arg);
            }
        }

        (ptxed_args, vdso_tempfile)
    }

    /*
     * Determine if the given x86_64 assembler mnemonic should terminate a block.
     *
     * Mnemonic assumed to be lower case.
     */
    fn instr_terminates_block(instr: &str) -> bool {
        assert!(instr.find(|c: char| !c.is_lowercase()).is_none());
        match instr {
            // JMP or Jcc are the only instructions beginning with 'j'.
            m if m.starts_with("j") => true,
            "call" | "ret" | "loop" | "loope" | "loopne" | "syscall" | "sysenter" | "sysexit"
            | "sysret" | "xabort" => true,
            _ => false,
        }
    }

    // Given a trace, use ptxed to get a vector of block start vaddrs.
    fn get_expected_blocks(trace: &Box<dyn Trace>) -> Vec<Block> {
        // Write the trace out to a temp file so ptxed can decode it.
        let mut tmpf = NamedTempFile::new().unwrap();
        trace.to_file(&mut tmpf.as_file_mut());
        let (args, _vdso_tempfile) = self_ptxed_args(tmpf.path().to_str().unwrap());

        let out = Command::new(env!("PTXED")).args(&args).output().unwrap();
        let outstr = String::from_utf8(out.stdout).unwrap();
        if !out.status.success() {
            let errstr = String::from_utf8(out.stderr).unwrap();
            panic!(
                "ptxed failed:\nInvocation----------\n{:?}\n \
                   Stdout\n------\n{}Stderr\n------\n{}",
                args, outstr, errstr
            );
        }

        let mut block_start = false;
        let mut block_vaddrs = Vec::new();
        let mut last_instr: Option<&str> = None;
        for line in outstr.lines() {
            let line = line.trim();
            if line.contains("error") {
                panic!("error line in ptxed output:\n{}", line);
            } else if line.starts_with("[") {
                // It's a special line, e.g. [enabled], [disabled], [block]...
                if line == "[block]"
                    && (last_instr.is_none() || instr_terminates_block(last_instr.unwrap()))
                {
                    // The next insruction we see will be the start of a block.
                    block_start = true;
                }
            } else {
                // It's a regular instruction line.
                if block_start {
                    // This instruction is the start of a block.
                    let vaddr_s = line.split_whitespace().next().unwrap();
                    let vaddr = u64::from_str_radix(vaddr_s, 16).unwrap();
                    block_vaddrs.push(Block::new(vaddr, 0));
                    block_start = false;
                }
                last_instr = Some(line.split_whitespace().nth(1).unwrap());
            }
        }

        block_vaddrs
    }

    // Trace a closure and then decode it and check the block iterator agrees with ptxed.
    fn trace_and_check_blocks<T, F>(mut tracer: T, f: F)
    where
        T: ThreadTracer,
        F: FnOnce() -> u64,
    {
        let trace = test_helpers::trace_closure(&mut tracer, f);
        let expects = get_expected_blocks(&trace);
        test_helpers::test_expected_blocks(trace, expects.iter());
    }

    #[test]
    fn test_basic_usage() {
        test_helpers::test_basic_usage(PerfPTThreadTracer::default());
    }

    #[test]
    fn test_repeated_tracing() {
        test_helpers::test_repeated_tracing(PerfPTThreadTracer::default());
    }

    #[test]
    fn test_already_started() {
        test_helpers::test_already_started(PerfPTThreadTracer::default());
    }

    #[test]
    fn test_not_started() {
        test_helpers::test_not_started(PerfPTThreadTracer::default());
    }

    // Test writing a trace to file.
    #[cfg(debug_assertions)]
    #[test]
    fn test_to_file() {
        use std::fs::File;
        use std::io::prelude::*;
        use std::slice;
        use Trace;

        // Allocate and fill a buffer to make a "trace" from.
        let capacity = 1024;
        let mut trace = PerfPTTrace::new(capacity).unwrap();
        trace.len = capacity as u64;
        let sl = unsafe { slice::from_raw_parts_mut(trace.buf.0 as *mut u8, capacity) };
        for (i, byte) in sl.iter_mut().enumerate() {
            *byte = i as u8;
        }

        // Make the trace and write it to a file.
        let mut tmpf = NamedTempFile::new().unwrap();
        trace.to_file(&mut tmpf.as_file_mut());
        tmpf.as_file().sync_all().unwrap();

        // Check the resulting file makes sense.
        let file = File::open(tmpf.path().to_str().unwrap()).unwrap();
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
        let tracer = PerfPTThreadTracer::default();
        trace_and_check_blocks(tracer, || test_helpers::work_loop(10));
    }

    // Check that our block decoder agrees ptxed on a (likely) empty trace;
    #[test]
    fn test_block_iterator2() {
        let tracer = PerfPTThreadTracer::default();
        trace_and_check_blocks(tracer, || test_helpers::work_loop(0));
    }

    // Check that our block decoder deals with traces involving the VDSO correctly.
    #[test]
    fn test_block_iterator3() {
        use libc::{clock_gettime, timespec, CLOCK_MONOTONIC};

        let tracer = PerfPTThreadTracer::default();
        trace_and_check_blocks(tracer, || {
            let mut res = 0;
            let mut tv = timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
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
        let tracer1 = PerfPTThreadTracer::default();
        let tracer2 = PerfPTThreadTracer::default();
        test_helpers::test_ten_times_as_many_blocks(tracer1, tracer2);
    }

    // Check that our block decoder agrees ptxed on long trace.
    // XXX We use an even higher iteration count once our decoder uses a libipt image cache.
    #[ignore] // Decoding long traces is slow.
    #[test]
    fn test_block_iterator5() {
        let tracer = PerfPTThreadTracer::default();
        trace_and_check_blocks(tracer, || test_helpers::work_loop(3000));
    }

    // Check that a long trace causes the trace buffer to reallocate.
    #[test]
    fn test_relloc_trace_buf1() {
        let start_bufsize = 512;
        let mut config = PerfPTConfig::default();
        config.initial_trace_bufsize = start_bufsize;
        let mut tracer = PerfPTThreadTracer::new(config);

        tracer.start_tracing().unwrap();
        let res = test_helpers::work_loop(10000);
        let trace = tracer.stop_tracing().unwrap();

        println!("res: {}", res); // Stop over-optimisation.
        assert!(trace.capacity() > start_bufsize);
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
            Some(Err(HWTracerError::Custom(e))) => {
                assert!(e.to_string().starts_with("libipt error: "))
            }
            _ => panic!(),
        }
        // And now the iterator is invalid, and should return None.
        for _ in 0..128 {
            assert!(itr.next().is_none());
        }
    }

    #[test]
    fn test_config_bad_data_bufsize() {
        let mut bldr = TracerBuilder::new().perf_pt();
        match bldr.config() {
            BackendConfig::PerfPT(ref mut ppt_conf) => ppt_conf.data_bufsize = 3,
            _ => panic!(),
        }
        match bldr.build() {
            Err(HWTracerError::BadConfig(s)) => {
                assert_eq!(s, "data_bufsize must be a positive power of 2");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_config_bad_aux_bufsize() {
        let mut bldr = TracerBuilder::new().perf_pt();
        match bldr.config() {
            BackendConfig::PerfPT(ref mut ppt_conf) => ppt_conf.aux_bufsize = 3,
            _ => panic!(),
        }
        match bldr.build() {
            Err(HWTracerError::BadConfig(s)) => {
                assert_eq!(s, "aux_bufsize must be a positive power of 2");
            }
            _ => panic!(),
        }
    }
}
