// Exporting parts of the LLVM C++ API not present in the LLVM C API.

// FIXME: C++ exceptions may unwind over the Rust FFI?
// https://github.com/ykjit/yk/issues/426

use libc::{c_void, size_t};
use std::os::raw::c_char;

pub mod symbolizer;

extern "C" {
    pub fn __ykllvmwrap_irtrace_compile(
        func_names: *const *const c_char,
        bbs: *const size_t,
        len: size_t,
        faddr_keys: *const *const c_char,
        faddr_vals: *const *const c_void,
        len: size_t,
        llvmbc_data: *const u8,
        llvmbc_len: size_t,
    ) -> *const c_void;
}
