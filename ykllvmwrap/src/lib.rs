// Exporting parts of the LLVM C++ API not present in the LLVM C API.

use libc::{c_void, size_t};
use std::os::raw::c_char;

pub mod symbolizer;

extern "C" {
    pub fn __ykllvmwrap_irtrace_compile(
        func_names: *const *const c_char,
        bbs: *const size_t,
        len: size_t,
        sec_ptr: *const c_char,
        sec_size: size_t,
    ) -> *const c_void;
}
