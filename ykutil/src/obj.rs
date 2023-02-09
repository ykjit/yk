//! Utilities for dealing with object files.

use crate::addr::dladdr;
use libc::c_void;
#[cfg(target_pointer_width = "64")]
use libc::Elf64_Addr as Elf_Addr;
use phdrs;
use std::{
    ffi::{CStr, CString},
    ops::Deref,
    path::PathBuf,
    ptr,
    sync::LazyLock,
};

/// A thread-safe wrapper around `phdrs::ProgramHeader`.
///
/// Because the headers are given out by the loader, and for now we assume no use of `dlclose()`,
/// we can safely use `unsafe impl` for `Send ` and `Sync`.
pub struct ProgramHeader(pub phdrs::ProgramHeader);

unsafe impl Send for ProgramHeader {}
unsafe impl Sync for ProgramHeader {}

impl From<&phdrs::ProgramHeader> for ProgramHeader {
    fn from(phdr: &phdrs::ProgramHeader) -> Self {
        Self(phdr.to_owned())
    }
}

impl Deref for ProgramHeader {
    type Target = phdrs::ProgramHeader;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A thread-safe wrapper around `phdrs::Object`.
pub struct Object(phdrs::Object);

unsafe impl Send for Object {}
unsafe impl Sync for Object {}

impl Deref for Object {
    type Target = phdrs::Object;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<&phdrs::Object> for Object {
    fn from(pobj: &phdrs::Object) -> Self {
        Self(pobj.to_owned())
    }
}

/// A program header cache.
///
/// This stashes the result of `dl_iterate_phdr(3)` (via the `phdr` crate), thus avoiding a (slow)
/// chain of C callbacks each time we want to inspect the program headers.
///
/// Since (for now) we assume that there can be no dlopen/dlclose, the cache is immutable.
pub static PHDR_OBJECT_CACHE: LazyLock<Vec<Object>> = LazyLock::new(|| {
    phdrs::objects()
        .iter()
        .map(|p| p.into())
        .collect::<Vec<Object>>()
});

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
    // If this fails, there's little we can do but crash.
    let info = dladdr(addr as usize).unwrap(); // ptr to usize cast always safe.
    PathBuf::from(info.dli_fname().unwrap().to_str().unwrap())
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
