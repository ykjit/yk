//! Utilities for dealing with object files.

use libc::{c_void, dladdr, Dl_info};
use std::{ffi::CStr, mem::MaybeUninit, path::PathBuf, ptr, sync::LazyLock};

// The name of the main object as it appears in the program headers.
//
// On Linux this is the empty string.
#[cfg(target_os = "linux")]
pub static PHDR_MAIN_OBJ: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::new());

extern "C" {
    fn find_main() -> *const c_void;
}

/// The path to the currently running binary.
///
/// This relies on there being an exported symbol for `main()`.
pub static SELF_BIN_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    let addr = unsafe { find_main() };
    if addr == ptr::null_mut() as *mut c_void {
        panic!("couldn't find address of main()");
    }
    let mut info = MaybeUninit::<Dl_info>::uninit();
    if unsafe { dladdr(addr, info.as_mut_ptr()) } == 0 {
        panic!("couldn't find Dl_info for main()");
    }
    let info = unsafe { info.assume_init() };
    PathBuf::from(unsafe { CStr::from_ptr(info.dli_fname) }.to_str().unwrap())
});

/// The `llvm.embedded.module` symbol in the `.llvmbc` section.
#[repr(C)]
struct EmbeddedModule {
    /// The length of the bitcode.
    len: usize,
    /// The start of the bitcode itself.
    first_byte_of_bitcode: u8,
}

// ykllvm adds the `SHF_ALLOC` flag to the `.llvmbc` section so that the loader puts it into our
// address space at load time.
extern "C" {
    #[link_name = "llvm.embedded.module"]
    static LLVMBC: EmbeddedModule;
}

/// Returns a pointer to (and the size of) the raw LLVM bitcode in the current address space.
pub fn llvmbc_section() -> (*const u8, usize) {
    let bc = unsafe { &LLVMBC };
    (&bc.first_byte_of_bitcode as *const u8, bc.len)
}
