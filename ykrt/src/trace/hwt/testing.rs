//! This module is only enabled when the `yk_testing` feature is enabled. It contains functions
//! that are only needed when testing internal yk code.

use hwtracer::llvm_blockmap::LLVM_BLOCK_MAP;

#[no_mangle]
pub extern "C" fn __yktrace_hwt_mapper_blockmap_len() -> usize {
    LLVM_BLOCK_MAP.len()
}
