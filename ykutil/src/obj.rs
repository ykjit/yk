//! Utilities for dealing with object files.

use memmap2::Mmap;
use object::{self, Object, ObjectSection, Section};
use std::lazy::SyncLazy;
use std::{convert::TryFrom, env, fs};

/// The current executable mmaped into the address space.
///
/// PERF: Consider making the segment containing .llvmbc loadable so that ld.so loads it
/// automatically when spawning the process.
static EXE_MMAP: SyncLazy<Mmap> = SyncLazy::new(|| {
    let pathb = env::current_exe().unwrap();
    let file = fs::File::open(&pathb.as_path()).unwrap();
    unsafe { memmap2::Mmap::map(&file).unwrap() }
});

/// The current executable parsed to an `object::File`.
static EXE_OBJ: SyncLazy<object::File> =
    SyncLazy::new(|| object::File::parse(&**EXE_MMAP).unwrap());

/// The .llvmbc section of the current executable.
static LLVMBC: SyncLazy<Section> = SyncLazy::new(|| EXE_OBJ.section_by_name(".llvmbc").unwrap());

/// Returns a pointer to (and the size of) the raw LLVM bitcode encoded in the .llvmbc section of
/// the current binary.
pub fn llvmbc_section() -> (*const u8, usize) {
    let sec_ptr = LLVMBC.data().unwrap().as_ptr();
    let sec_size = usize::try_from(LLVMBC.size()).unwrap();
    (sec_ptr, sec_size)
}
