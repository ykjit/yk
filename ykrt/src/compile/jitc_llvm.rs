//! An LLVM JIT backend. Currently a minimal wrapper around the fact that [MappedTrace]s are hardcoded
//! to be compiled with LLVM.

use crate::{
    compile::{CompiledTrace, Compiler},
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::MappedTrace,
};
use libc::dlsym;
use parking_lot::Mutex;
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::{
    env,
    error::Error,
    ffi::{c_char, c_int, CString},
    ptr,
    sync::Arc,
};
use tempfile::NamedTempFile;

pub(crate) struct JITCLLVM;

impl Compiler for JITCLLVM {
    fn compile(
        &self,
        mt: Arc<MT>,
        irtrace: MappedTrace,
        sti: Option<SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<CompiledTrace, Box<dyn Error>> {
        let (func_names, bbs, trace_len) = self.encode_trace(&irtrace);

        let mut faddr_keys = Vec::new();
        let mut faddr_vals = Vec::new();
        for k in irtrace.faddrs().iter() {
            faddr_keys.push(k.0.as_ptr());
            faddr_vals.push(*k.1);
        }

        let (llvmbc_data, llvmbc_len) = llvmbc_section();
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
                faddr_keys.as_ptr(),
                faddr_vals.as_ptr(),
                faddr_keys.len(),
                llvmbc_data,
                llvmbc_len,
                di_fd,
                di_tmpname_c,
                callstack,
                aotvalsptr,
                aotvalslen,
            )
        };
        if ret.is_null() {
            Err("Could not compile trace.".into())
        } else {
            Ok(CompiledTrace::new(mt, ret, di_tmp, Arc::downgrade(&hl)))
        }
    }

    #[cfg(feature = "yk_testing")]
    unsafe fn compile_for_tc_tests(
        &self,
        irtrace: MappedTrace,
        llvmbc_data: *const u8,
        llvmbc_len: u64,
    ) {
        let (func_names, bbs, trace_len) = self.encode_trace(&irtrace);
        let (_di_tmp, di_fd, di_tmpname_c) = Self::create_debuginfo_temp_file();

        // These would only need to be populated if we were to load the resulting compiled code
        // into the address space, which for trace compiler tests, we don't.
        let faddr_keys = Vec::new();
        let faddr_vals = Vec::new();

        let ret = yktracec::__yktracec_irtrace_compile_for_tc_tests(
            func_names.as_ptr(),
            bbs.as_ptr(),
            trace_len,
            faddr_keys.as_ptr(),
            faddr_vals.as_ptr(),
            faddr_keys.len(),
            llvmbc_data,
            llvmbc_len,
            di_fd,
            di_tmpname_c,
        );
        assert_ne!(ret, ptr::null());
    }
}

impl JITCLLVM {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(JITCLLVM))
    }

    fn encode_trace(&self, irtrace: &MappedTrace) -> (Vec<*const i8>, Vec<usize>, usize) {
        let trace_len = irtrace.len();
        let mut func_names = Vec::with_capacity(trace_len);
        let mut bbs = Vec::with_capacity(trace_len);
        for blk in irtrace.blocks() {
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

/// The `llvm.embedded.module` symbol in the `.llvmbc` section.
#[repr(C)]
struct EmbeddedModule {
    /// The length of the bitcode.
    len: u64,
    /// The start of the bitcode itself.
    first_byte_of_bitcode: u8,
}

/// Returns a pointer to (and the size of) the raw LLVM bitcode in the current address space.
pub(crate) fn llvmbc_section() -> (*const u8, u64) {
    // ykllvm adds the `SHF_ALLOC` flag to the `.llvmbc` section so that the loader puts it into
    // our address space at load time.
    let bc = unsafe {
        &*(dlsym(
            std::ptr::null_mut(),
            CString::new("llvm.embedded.module")
                .unwrap()
                .as_c_str()
                .as_ptr(),
        ) as *const EmbeddedModule)
    };
    (&bc.first_byte_of_bitcode as *const u8, bc.len)
}
