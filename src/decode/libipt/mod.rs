//! The libipt trace decoder.

use crate::{c_errors::PerfPTCError, decode::TraceDecoder, errors::HWTracerError, Block, Trace};
use libc::{c_char, c_int, c_void};
use std::{convert::TryFrom, env, ffi::CString, os::fd::AsRawFd, ptr};
use tempfile::NamedTempFile;

extern "C" {
    // decode.c
    fn hwt_ipt_init_block_decoder(
        buf: *const c_void,
        len: u64,
        vdso_fd: c_int,
        vdso_filename: *const c_char,
        decoder_status: *mut c_int,
        err: *mut PerfPTCError,
        current_exe: *const c_char,
    ) -> *mut c_void;
    fn hwt_ipt_next_block(
        decoder: *mut c_void,
        decoder_status: *mut c_int,
        addr: *mut u64,
        len: *mut u64,
        err: *mut PerfPTCError,
    ) -> bool;
    fn hwt_ipt_free_block_decoder(decoder: *mut c_void);
    // util.c
    pub(crate) fn hwt_ipt_is_overflow_err(err: c_int) -> bool;
    // libipt
    pub(crate) fn pt_errstr(error_code: c_int) -> *const c_char;
}

pub(crate) struct LibIPTTraceDecoder {}

impl TraceDecoder for LibIPTTraceDecoder {
    fn new() -> Self {
        Self {}
    }

    fn iter_blocks<'t>(
        &'t self,
        trace: &'t dyn Trace,
    ) -> Box<dyn Iterator<Item = Result<Block, HWTracerError>> + '_> {
        let itr = LibIPTBlockIterator {
            decoder: ptr::null_mut(),
            decoder_status: 0,
            vdso_tempfile: None,
            trace,
            errored: false,
        };
        Box::new(itr)
    }
}

/// Iterate over the blocks of an Intel PT trace using libipt.
struct LibIPTBlockIterator<'t> {
    /// C-level libipt block decoder.
    decoder: *mut c_void,
    /// Stores the current libipt-level status of the above decoder.
    decoder_status: c_int,
    /// VDSO code (stored temporarily).
    #[allow(dead_code)]
    // Rust doesn't know that this exists only to keep the file long enough.
    vdso_tempfile: Option<NamedTempFile>,
    /// The trace we are iterating over.
    trace: &'t dyn Trace,
    /// Error state. Set to true when an error occurs, thus invalidating the iterator.
    errored: bool,
}

impl<'t> LibIPTBlockIterator<'t> {
    /// Initialise the block decoder.
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
            hwt_ipt_init_block_decoder(
                self.trace.bytes().as_ptr() as *const c_void,
                u64::try_from(self.trace.len()).unwrap(),
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

impl<'t> Drop for LibIPTBlockIterator<'t> {
    fn drop(&mut self) {
        unsafe { hwt_ipt_free_block_decoder(self.decoder) };
    }
}

impl<'t> Iterator for LibIPTBlockIterator<'t> {
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
            hwt_ipt_next_block(
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

#[cfg(test)]
mod tests {
    use super::{LibIPTBlockIterator, PerfPTCError};
    use crate::{
        collect::{
            perf::PerfTrace, test_helpers::trace_closure, ThreadTraceCollector,
            TraceCollectorBuilder,
        },
        decode::{test_helpers, TraceDecoderKind},
        errors::HWTracerError,
        test_helpers::work_loop,
        Block, Trace,
    };
    use libc::{c_int, size_t, PF_X, PT_LOAD};
    use std::{convert::TryFrom, env, os::fd::AsRawFd, process::Command, ptr};
    use tempfile::NamedTempFile;

    extern "C" {
        fn hwt_ipt_dump_vdso(fd: c_int, vaddr: u64, len: size_t, err: &PerfPTCError) -> bool;
    }

    const VDSO_FILENAME: &str = "linux-vdso.so.1";

    /// Gets the ptxed arguments required to decode a trace for the current process.
    ///
    /// Returns a vector of arguments and a handle to a temproary file containing the VDSO code.
    /// The caller must make sure that this file lives long enough for ptxed to run (temp files are
    /// removed when they fall out of scope).
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
                if hdr.type_() != PT_LOAD || hdr.flags() & PF_X == 0 {
                    continue; // Only look at loadable and executable segments.
                }

                let vaddr = obj.addr() + hdr.vaddr();
                let offset;

                if filename == VDSO_FILENAME {
                    let cerr = PerfPTCError::new();
                    if !unsafe {
                        hwt_ipt_dump_vdso(
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

    /// Determine if the given x86_64 assembler mnemonic should terminate a block.
    ///
    /// Mnemonic assumed to be lower case.
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

    /// Given a trace, use ptxed to get a vector of block start vaddrs.
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

    /// Trace a closure and then decode it and check the block iterator agrees with ptxed.
    fn trace_and_check_blocks<F>(tracer: &mut dyn ThreadTraceCollector, f: F)
    where
        F: FnOnce() -> u64,
    {
        let trace = trace_closure(tracer, f);
        let expects = get_expected_blocks(&trace);
        test_helpers::test_expected_blocks(trace, TraceDecoderKind::LibIPT, expects.iter());
    }

    /// Check that the block decoder agrees with the reference implementation in ptxed.
    #[test]
    fn versus_ptxed_short_trace() {
        let tracer = TraceCollectorBuilder::new().build().unwrap();
        trace_and_check_blocks(&mut *unsafe { tracer.thread_collector() }, || work_loop(10));
    }

    /// Check that the block decoder agrees ptxed on a (likely) empty trace;
    #[test]
    fn versus_ptxed_empty_trace() {
        let tracer = TraceCollectorBuilder::new().build().unwrap();
        trace_and_check_blocks(&mut *unsafe { tracer.thread_collector() }, || work_loop(0));
    }

    /// Check that our block decoder deals with traces involving the VDSO correctly.
    #[test]
    fn versus_ptxed_vdso() {
        use libc::{clock_gettime, timespec, CLOCK_MONOTONIC};

        let tracer = TraceCollectorBuilder::new().build().unwrap();
        trace_and_check_blocks(&mut *unsafe { tracer.thread_collector() }, || {
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

    /// Check that the block decoder agrees with ptxed on long trace.
    #[test]
    fn versus_ptxed_long_trace() {
        let tracer = TraceCollectorBuilder::new().build().unwrap();
        trace_and_check_blocks(&mut *unsafe { tracer.thread_collector() }, || {
            work_loop(3000)
        });
    }

    /// Check that a block iterator returns none after an error.
    #[test]
    fn error_stops_block_iter() {
        // A zero-sized trace will lead to an error.
        let trace = PerfTrace::new(0).unwrap();
        let mut itr = LibIPTBlockIterator {
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
    fn ten_times_as_many_blocks() {
        let col = TraceCollectorBuilder::new().build().unwrap();
        test_helpers::ten_times_as_many_blocks(
            &mut *unsafe { col.thread_collector() },
            TraceDecoderKind::LibIPT,
        );
    }
}
