//! Linux perf trace profiling support.
//!
//! This module implements the ability to profile our JITted code using perf. In `perf report` and
//! firefox profiler, our traces will be properly recognised and have human-readable symbol names
//! containing the trace ID. Perf's "annotate" feature will also work.
//!
//! This is how it works:
//!
//!  - The process is run under `perf record`.
//!
//!  - The code in this module creates a `jitdump-<pid>.dump` file is created for the process. The
//!    filename must be of this form, or perf won't find it.
//!
//!  - The file is mmap(2)'d into the address space with PROT_EXEC permissions and held mapped
//!    until the process exits. This is how perf is notified of jitdump files that it should look
//!    at later in the `perf inject` stage. We never use the mapping ourselves. The mapping is
//!    called the "marker".
//!
//!  - Each time new JITted code is introduced, a record is written to the dump file specifying
//!    where the code starts, its length, the time when it was created, symbol name, etc.
//!
//!  - Once the process has finished, `perf inject` is used to create an on-disk shared object for
//!    JITted code (e.g. `jitted-<pid>-0.so`) and an updated `perf.data` file referencing the
//!    shared object.
//!
//!  - Using the updated perf data file, `perf report` and FF profiler will then show the JITted
//!    traces properly.
//!
//!  Notes for developers:
//!
//!   - The jitdump file format is documented here:
//!     https://raw.githubusercontent.com/torvalds/linux/master/tools/perf/Documentation/jitdump-specification.txt
//!
//!   - `perf script --show-mmap-events -i <perf-data-file>` can be useful in debugging. It shows
//!     the MMAP records perf saw during recording and the order in which they appeared. The
//!     PROT_EXEC mapping mentioned above should appear before any JITted code is created.

use super::PlatformTraceProfiler;
use crate::compile::CompiledTrace;
use byteorder::{NativeEndian, WriteBytesExt};
use libc::{self, CLOCK_MONOTONIC, clock_gettime, getpid, timespec};
use memmap2::{Mmap, MmapOptions};
use parking_lot::Mutex;
use std::{
    error::Error,
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
    sync::{Arc, LazyLock},
};

/// The JIT dump file for the current process, if required, else None.
///
/// Because perf expects to pick up the dump from a file with a pid-specific-name, the file has to
/// capture traces created by any and all instances of meta-tracers within in the current process.
/// Hence this is a global variable.
///
/// This also means that multiple threads could share this, and therefore we have to synchronise
/// writing to the jitdump file so that output is written sequentially.
static JIT_DUMP: LazyLock<Mutex<JitDump>> = LazyLock::new(|| Mutex::new(JitDump::new().unwrap()));

/// Create a timestamp of "now" in nanoseconds.
fn now_timestamp() -> u64 {
    let mut ts = std::mem::MaybeUninit::<timespec>::uninit();
    // Note: For the JITted code to materialise at the right time in the profile, our clock source
    // must be the same as perf is using. The perf docs recommend using CLOCK_MONOTONIC, so we use
    // that. This means that `perf record` must be invoked with `-k CLOCK_MONOTONIC`.
    if unsafe { clock_gettime(CLOCK_MONOTONIC, ts.as_mut_ptr()) } != 0 {
        // If this fails, we have big problems, and it's probably not worth trying to recover.
        panic!("failed to read clock");
    }
    let ts = unsafe { ts.assume_init() };
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

struct JitDump {
    /// The jitdump file we will write records to.
    jitdump: std::fs::File,
    /// The mmap marker.
    ///
    /// We don't read or write to this mapping, but it is required by perf if we want it to
    /// recognise our jitdump file. Further, we need it to stay alive for the remainder of the
    /// profiling session, so we store it to prevent it from being dropped (and thus unmapped).
    #[allow(dead_code)]
    marker: Mmap,
    /// The current processes PID.
    pid: u32,
}

/// Identifies a JIT_CODE_LOAD record in a jitdump file.
const JIT_CODE_LOAD: u32 = 0;

impl JitDump {
    /// Create a jitdump file for the current PID.
    fn new() -> Result<Self, Box<dyn Error>> {
        let pid = unsafe { getpid() } as u32;
        let filename = format!("jit-{}.dump", pid);
        let mut f = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&filename)?;

        // Write the file header.
        //
        // uint32_t magic
        f.write_u32::<NativeEndian>(0x4A695444)?;
        // uint32_t version
        f.write_u32::<NativeEndian>(1)?;
        // uint32_t total_size: size in bytes of file header
        let fh_size_pos = f.stream_position()?;
        f.write_u32::<NativeEndian>(u32::MAX)?; // placeholder, patched later.
        // uint32_t elf_mach
        #[cfg(target_arch = "x86_64")]
        f.write_u32::<NativeEndian>(u32::from(libc::EM_X86_64))?;
        // uint32_t pad1
        f.write_u32::<NativeEndian>(0)?;
        // uint32_t pid
        f.write_u32::<NativeEndian>(pid)?;
        // uint64_t timestamp
        f.write_u64::<NativeEndian>(now_timestamp())?;
        // uint64_t flags
        f.write_u64::<NativeEndian>(0)?;

        // Patch the file header size field now we know how big it is.
        let end_fh_pos = f.stream_position()?;
        f.seek(SeekFrom::Start(fh_size_pos))?;
        f.write_u32::<NativeEndian>(u32::try_from(end_fh_pos).unwrap())?;
        f.seek(SeekFrom::Start(end_fh_pos))?;

        // For perf to see the and inject the JITted code, we have to map the jitdump file into
        // memory using a PROT_EXEC memmap().
        let opts = MmapOptions::new();
        let marker = unsafe { opts.map_exec(&f).unwrap() };

        Ok(Self {
            jitdump: f,
            marker,
            pid,
        })
    }

    fn emit_code_load_record(
        &mut self,
        ctr: &Arc<dyn CompiledTrace>,
    ) -> Result<(), Box<dyn Error>> {
        let pos_before = self.jitdump.stream_position()?;
        // Write record header
        //
        // uint32_t id
        // (note: not really an ID, rather the kind of JIT event)
        self.jitdump.write_u32::<NativeEndian>(JIT_CODE_LOAD)?;
        // uint32_t total_size
        let rh_size_pos = self.jitdump.stream_position()?;
        self.jitdump.write_u32::<NativeEndian>(u32::MAX)?; // placeholder, patched later.
        // uint64_t timestamp
        self.jitdump.write_u64::<NativeEndian>(now_timestamp())?;

        // Write the JIT_CODE_LOAD record.
        //
        // uint32_t pid
        self.jitdump.write_u32::<NativeEndian>(self.pid)?;
        // uint32_t tid
        self.jitdump
            .write_u32::<NativeEndian>(unsafe { u32::try_from(libc::gettid()).unwrap() })?;
        // uint64_t vma: virtual address of jitted code start
        #[cfg(target_pointer_width = "64")]
        let code_start = ctr.entry() as u64;
        self.jitdump.write_u64::<NativeEndian>(code_start)?;
        // uint64_t code_addr
        self.jitdump.write_u64::<NativeEndian>(code_start)?;
        // uint64_t code_size: size in bytes of the generated jitted code
        self.jitdump
            .write_u64::<NativeEndian>(u64::try_from(ctr.code().len()).unwrap())?;
        // uint64_t code_index
        self.jitdump
            .write_u64::<NativeEndian>(ctr.ctrid().as_u64())?;
        // char[n]: function name (including the null terminator)
        self.jitdump.write_all(ctr.name().as_bytes())?;
        self.jitdump.write_u8(0)?;
        // native code.
        self.jitdump.write_all(ctr.code())?;

        // patch in the record size.
        let rh_size = self.jitdump.stream_position()? - pos_before;
        self.jitdump.seek(SeekFrom::Start(rh_size_pos))?;
        self.jitdump
            .write_u32::<NativeEndian>(u32::try_from(rh_size).unwrap())?;
        self.jitdump.seek(SeekFrom::End(0))?;

        self.jitdump.flush()?;
        Ok(())
    }
}

pub(crate) struct LinuxPerf {}

impl LinuxPerf {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl PlatformTraceProfiler for LinuxPerf {
    fn register_ctr(&self, ctr: &Arc<dyn CompiledTrace>) -> Result<(), Box<dyn Error>> {
        JIT_DUMP.lock().emit_code_load_record(ctr)
    }
}
