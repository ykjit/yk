//! Utilities for dealing with object files.

use crate::addr::dladdr;
use libc::c_void;
use std::{fs, path::PathBuf, sync::LazyLock};

unsafe extern "C" {
    fn find_main() -> *const c_void;
}

/// The path to the currently running binary.
///
/// This relies on there being an exported symbol for `main()`.
pub static SELF_BIN_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    let addr = unsafe { find_main() };
    if addr.is_null() {
        panic!("couldn't find address of main()");
    }
    // If this fails, there's little we can do but crash.
    let info = dladdr(addr as usize).unwrap(); // ptr to usize cast always safe.
    PathBuf::from(info.dli_fname().unwrap().to_str().unwrap())
});

// The main binary's ELF executable mapped into the address space.
pub static SELF_BIN_MMAP: LazyLock<memmap2::Mmap> = LazyLock::new(|| {
    let file = fs::File::open(SELF_BIN_PATH.as_path()).unwrap();
    unsafe { memmap2::Mmap::map(&file).unwrap() }
});
