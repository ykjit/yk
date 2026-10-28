//! Utilities for dealing with object files.

use std::{fs::File, sync::LazyLock};

// The main binary's ELF executable mapped into the address space.
pub static SELF_BIN_MMAP: LazyLock<memmap2::Mmap> = LazyLock::new(|| {
    let self_file = self_as_file();
    unsafe { memmap2::Mmap::map(&self_file).unwrap() }
});

#[cfg(target_os = "linux")]
fn self_as_file() -> File {
    File::open("/proc/self/exe").unwrap()
}
