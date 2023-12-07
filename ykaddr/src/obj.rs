//! Utilities for dealing with object files.

use crate::addr::dladdr;
use libc::c_void;
#[cfg(target_pointer_width = "64")]
use libc::{
    Elf64_Addr as Elf_Addr, Elf64_Off as Elf_Off, Elf64_Word as Elf_Word, Elf64_Xword as Elf_Xword,
};
use memmap2;
use phdrs;
use std::{
    ffi::{CStr, CString},
    fs,
    path::PathBuf,
    sync::LazyLock,
};

/// A thread-safe (containing no raw pointers) version of `phdrs::ProgramHeader`.
pub struct ProgramHeader {
    flags: Elf_Word,
    type_: Elf_Word,
    vaddr: Elf_Addr,
    memsz: Elf_Xword,
    filesz: Elf_Xword,
    offset: Elf_Off,
}

impl From<&phdrs::ProgramHeader> for ProgramHeader {
    fn from(phdr: &phdrs::ProgramHeader) -> Self {
        Self {
            flags: phdr.flags(),
            type_: phdr.type_(),
            vaddr: phdr.vaddr(),
            memsz: phdr.memsz(),
            filesz: phdr.filesz(),
            offset: phdr.offset(),
        }
    }
}

impl ProgramHeader {
    pub fn flags(&self) -> Elf_Word {
        self.flags
    }

    pub fn type_(&self) -> Elf_Word {
        self.type_
    }

    pub fn vaddr(&self) -> Elf_Addr {
        self.vaddr
    }

    pub fn memsz(&self) -> Elf_Xword {
        self.memsz
    }

    pub fn filesz(&self) -> Elf_Xword {
        self.filesz
    }

    pub fn offset(&self) -> Elf_Off {
        self.offset
    }
}

/// A thread-safe (containing no raw pointers) version of `phdrs::Object`.
pub struct Object {
    /// The base address of the object.
    addr: Elf_Addr,
    /// The name of the object.
    name: CString,
    /// Vector of program headers.
    phdrs: Vec<ProgramHeader>,
}

impl Object {
    pub fn addr(&self) -> Elf_Addr {
        self.addr
    }

    pub fn name(&self) -> &CStr {
        &self.name
    }

    pub fn phdrs(&self) -> &Vec<ProgramHeader> {
        &self.phdrs
    }
}

impl From<&phdrs::Object> for Object {
    fn from(pobj: &phdrs::Object) -> Self {
        Self {
            addr: pobj.addr(),
            name: pobj.name().to_owned(),
            phdrs: pobj.iter_phdrs().map(|ref p| p.into()).collect::<Vec<_>>(),
        }
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
pub static PHDR_MAIN_OBJ: LazyLock<PathBuf> = LazyLock::new(PathBuf::new);

extern "C" {
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
    let file = fs::File::open(&SELF_BIN_PATH.as_path()).unwrap();
    unsafe { memmap2::Mmap::map(&file).unwrap() }
});
