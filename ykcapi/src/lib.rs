//! This crate exports the Yk API via the C ABI.
//!
//! We use a dedicated crate for exporting to C, as you quickly get into linkage trouble if you try
//! and mix Rust dynamic libraries (namely you can get duplicate copies of dependencies).
//!
//! The sane solution is to have only one `cdylib` crate in our workspace (this crate) and all
//! other crates are regular `rlibs`.

#![feature(bench_black_box)]
#![feature(c_variadic)]
#![feature(once_cell)]

#[cfg(feature = "c_testing")]
use std::env;
use std::ffi::c_void;
#[cfg(feature = "c_testing")]
use std::lazy::SyncLazy;
use ykrt::{Location, MT};
use ykutil;

#[cfg(feature = "c_testing")]
static SERIALISE_COMPILATION: SyncLazy<bool> =
    SyncLazy::new(|| &env::var("YKD_SERIALISE_COMPILATION").unwrap_or("0".to_owned()) == "1");

// The "dummy control point" that is replaced in an LLVM pass.
#[no_mangle]
pub extern "C" fn yk_control_point(_loc: *mut Location) {
    // Intentionally empty.
}

// The "real" control point, that is called once the interpreter has been patched by ykllvm.
#[no_mangle]
pub extern "C" fn __ykrt_control_point(loc: *mut Location, ctrlp_vars: *mut c_void) {
    debug_assert!(!ctrlp_vars.is_null());
    if !loc.is_null() {
        let loc = unsafe { &*loc };
        MT::transition_location(loc, ctrlp_vars);

        #[cfg(feature = "c_testing")]
        {
            if *SERIALISE_COMPILATION {
                loc.block_if_compiling();
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn yk_location_new() -> Location {
    Location::new()
}

#[no_mangle]
pub extern "C" fn yk_location_drop(loc: Location) {
    drop(loc)
}

/// Return a pointer to (and the size of) the .llvmbc section of the current executable.
#[no_mangle]
pub extern "C" fn __ykutil_get_llvmbc_section(res_addr: *mut *const c_void, res_size: *mut usize) {
    let (addr, size) = ykutil::obj::llvmbc_section();
    unsafe {
        *res_addr = addr as *const c_void;
        *res_size = size;
    }
}

/// The following module contains exports only used for testing from external C code.
/// These symbols are not shipped as part of the main API.
#[cfg(feature = "c_testing")]
mod c_testing {
    use yktrace::BlockMap;

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
}
