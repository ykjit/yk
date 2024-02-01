// Exporting parts of the LLVM C++ API not present in the LLVM C API.

#![allow(clippy::new_without_default)]
#![feature(c_variadic)]

// FIXME: C++ exceptions may unwind over the Rust FFI?
// https://github.com/ykjit/yk/issues/426

use libc::{c_void, size_t};
use std::ffi::{c_char, c_int};

pub mod promote;

extern "C" {
    pub fn __yktracec_irtrace_compile(
        func_names: *const *const c_char,
        bbs: *const size_t,
        trace_len: size_t,
        llvmbc_data: *const u8,
        llvmbc_len: u64,
        debuginfo_fd: c_int,
        debuginfo_path: *const c_char,
        jitcallstack: *const c_void,
        aotvalsptr: *const c_void,
        aotvalslen: usize,
    ) -> *const c_void;
}
