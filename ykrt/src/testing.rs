//! This module is only enabled when the `yk_testing` feature is enabled. It contains functions
//! that are only needed when testing internal yk code.

use hwtracer::llvm_blockmap::BlockMap;

#[no_mangle]
pub extern "C" fn __yktrace_hwt_mapper_blockmap_new() -> *mut BlockMap {
    Box::into_raw(Box::new(BlockMap::new()))
}

#[no_mangle]
pub extern "C" fn __yktrace_hwt_mapper_blockmap_free(bm: *mut BlockMap) {
    unsafe { Box::from_raw(bm) };
}

#[no_mangle]
pub extern "C" fn __yktrace_hwt_mapper_blockmap_len(bm: *mut BlockMap) -> usize {
    unsafe { &*bm }.len()
}
