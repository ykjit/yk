//! Utilities for dealing with object files.

use libc::{c_void, dladdr, Dl_info};
use memmap2::Mmap;
use object::{self, Object, ObjectSection, Section};
use std::{
    convert::TryFrom, env, ffi::CStr, fs, mem::MaybeUninit, path::PathBuf, ptr, sync::LazyLock,
};

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

/// The current executable mmaped into the address space.
///
/// PERF: Consider making the segment containing .llvmbc loadable so that ld.so loads it
/// automatically when spawning the process.
static EXE_MMAP: LazyLock<Mmap> = LazyLock::new(|| {
    let pathb = env::current_exe().unwrap();
    let file = fs::File::open(&pathb.as_path()).unwrap();
    unsafe { memmap2::Mmap::map(&file).unwrap() }
});

/// The current executable parsed to an `object::File`.
static EXE_OBJ: LazyLock<object::File> =
    LazyLock::new(|| object::File::parse(&**EXE_MMAP).unwrap());

/// The .llvmbc section of the current executable.
static LLVMBC: LazyLock<Section> = LazyLock::new(|| EXE_OBJ.section_by_name(".llvmbc").unwrap());

/// Returns a pointer to (and the size of) the raw LLVM bitcode encoded in the .llvmbc section of
/// the current binary.
pub fn llvmbc_section() -> (*const u8, usize) {
    let sec_ptr = LLVMBC.data().unwrap().as_ptr();
    let sec_size = usize::try_from(LLVMBC.size()).unwrap();
    (sec_ptr, sec_size)
}
