//! An LLVM JIT backend. Currently a minimal wrapper around the fact that [MappedTrace]s are hardcoded
//! to be compiled with LLVM.

use crate::{
    compile::{CompilationError, CompiledTrace, Compiler},
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::TracedAOTBlock,
};
use object::{Object, ObjectSection};
use parking_lot::Mutex;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::{
    env,
    ffi::{c_char, c_int},
    ptr,
    sync::{Arc, LazyLock},
};
use tempfile::NamedTempFile;
use ykaddr::obj::SELF_BIN_MMAP;

pub static LLVM_BITCODE: LazyLock<&[u8]> = LazyLock::new(|| {
    let object = object::File::parse(&**SELF_BIN_MMAP).unwrap();
    let sec = object.section_by_name(".llvmbc").unwrap();
    sec.data().unwrap()
});

pub(crate) struct JITCLLVM;

impl Compiler for JITCLLVM {
    fn compile(
        &self,
        mt: Arc<MT>,
        irtrace: Vec<TracedAOTBlock>,
        sti: Option<SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<CompiledTrace, CompilationError> {
        let (func_names, bbs, trace_len) = self.encode_trace(&irtrace);

        let llvmbc = llvmbc_section();
        let (di_tmp, di_fd, di_tmpname_c) = Self::create_debuginfo_temp_file();

        let (callstack, aotvalsptr, aotvalslen) = match sti {
            Some(sti) => (sti.callstack, sti.aotvalsptr, sti.aotvalslen),
            None => (std::ptr::null(), std::ptr::null(), 0),
        };

        let ret = unsafe {
            yktracec::__yktracec_irtrace_compile(
                func_names.as_ptr(),
                bbs.as_ptr(),
                trace_len,
                llvmbc.as_ptr(),
                u64::try_from(llvmbc.len()).unwrap(),
                di_fd,
                di_tmpname_c,
                callstack,
                aotvalsptr,
                aotvalslen,
            )
        };
        if ret.is_null() {
            // The LLVM backend is now legacy code and is pending deletion, so it's not worth us
            // spending time auditing all of the failure modes and categorising them into
            // recoverable/temporary. So for now we say any error is temporary.
            Err(CompilationError::Temporary("llvm backend error".into()))
        } else {
            Ok(CompiledTrace::new(mt, ret, di_tmp, Arc::downgrade(&hl)))
        }
    }
}

impl JITCLLVM {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(JITCLLVM)
    }

    fn encode_trace(&self, irtrace: &Vec<TracedAOTBlock>) -> (Vec<*const i8>, Vec<usize>, usize) {
        let trace_len = irtrace.len();
        let mut func_names = Vec::with_capacity(trace_len);
        let mut bbs = Vec::with_capacity(trace_len);
        for blk in irtrace {
            if blk.is_unmappable() {
                // The block was unmappable. Indicate this with a null function name.
                func_names.push(ptr::null());
                // Block indices for unmappable blocks are irrelevant so we may pass anything here.
                bbs.push(0);
            } else {
                func_names.push(blk.func_name().as_ptr());
                bbs.push(blk.bb());
            }
        }
        (func_names, bbs, trace_len)
    }

    // If necessary, create a temporary file for us to write the trace's debugging "source code"
    // into. Elsewhere, the JIT module will have `DebugLoc`s inserted into it which will point to
    // lines in this temporary file.
    //
    // If the `YKD_TRACE_DEBUGINFO` environment variable is set to "1", then this function returns
    // a `NamedTempFile`, a non-negative file descriptor, and a path to the file.
    //
    // If the `YKD_TRACE_DEBUGINFO` environment variable is *not* set to "1", then no file is
    // created and this function returns `(None, -1, ptr::null())`.
    #[cfg(unix)]
    fn create_debuginfo_temp_file() -> (Option<NamedTempFile>, c_int, *const c_char) {
        let mut di_tmp = None;
        let mut di_fd = -1;
        let mut di_tmpname_c = ptr::null() as *const c_char;
        if let Ok(di_val) = env::var("YKD_TRACE_DEBUGINFO") {
            if di_val == "1" {
                let tmp = NamedTempFile::new().unwrap();
                di_tmpname_c = tmp.path().to_str().unwrap().as_ptr() as *const c_char;
                di_fd = tmp.as_raw_fd();
                di_tmp = Some(tmp);
            }
        }
        (di_tmp, di_fd, di_tmpname_c)
    }
}

/// Returns a pointer to (and the size of) the raw LLVM bitcode in the current address space.
pub(crate) fn llvmbc_section() -> &'static [u8] {
    &LLVM_BITCODE
}
