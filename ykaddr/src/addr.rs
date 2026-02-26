//! Address utilities.

use cached::proc_macro::cached;
use libc::{Dl_info, RTLD_DEFAULT, c_void, dlsym};
use std::{
    error::Error,
    ffi::{CStr, CString},
    mem::MaybeUninit,
};

/// A Rust wrapper around `libc::Dl_info` using FFI types.
///
/// The strings inside are handed out by the loader and can (for now, since we don't support
/// dlclose) be considered of static lifetime. This makes the struct thread safe, and thus
/// cacheable using `#[cached]`.
#[derive(Debug, Clone)]
pub struct DLInfo {
    dli_fname: Option<&'static CStr>,
    dli_fbase: usize,
    dli_sname: Option<&'static CStr>,
    dli_saddr: usize,
}

impl From<Dl_info> for DLInfo {
    fn from(dli: Dl_info) -> Self {
        let dli_fname = if !dli.dli_fname.is_null() {
            Some(unsafe { CStr::from_ptr(dli.dli_fname) })
        } else {
            None
        };

        let dli_sname = if !dli.dli_sname.is_null() {
            Some(unsafe { CStr::from_ptr(dli.dli_sname) })
        } else {
            None
        };

        Self {
            dli_fname,
            dli_fbase: dli.dli_fbase as usize,
            dli_sname,
            dli_saddr: dli.dli_saddr as usize,
        }
    }
}

impl DLInfo {
    pub fn dli_fname(&self) -> Option<&'static CStr> {
        self.dli_fname
    }
    pub fn dli_fbase(&self) -> usize {
        self.dli_fbase
    }
    pub fn dli_sname(&self) -> Option<&'static CStr> {
        self.dli_sname
    }
    pub fn dli_saddr(&self) -> usize {
        self.dli_saddr
    }
}

/// Wraps `libc::dlinfo`.
///
/// Returns `Err` if the underlying call to `libc::dlddr` fails.
///
/// FIXME: This cache should be invalidated (in part, if possible) when a object is loaded or
/// unloaded from the address space.
///
/// FIXME: Consider using a LRU cache to limit memory consumption. The cached crate can do this for
/// us if we can give it a suitable cache size.
///
/// FIXME: This cache is cloning. Performance could probably be improved more.
#[cached]
pub fn dladdr(vaddr: usize) -> Option<DLInfo> {
    let mut info = MaybeUninit::<Dl_info>::uninit();
    if unsafe { libc::dladdr(vaddr as *const c_void, info.as_mut_ptr()) } != 0 {
        Some(unsafe { info.assume_init() }.into())
    } else {
        None
    }
}

/// A thin wrapper around `dlsym()` for mapping symbol names to virtual addresses.
///
/// FIXME: Look for raw uses of `dlsym()` throughout our code base and replace them with a call to
/// this wrapper. Related: https://github.com/ykjit/yk/issues/835
pub fn symbol_to_ptr(name: &str) -> Result<*const (), Box<dyn Error>> {
    let s = CString::new(name).unwrap();
    let p = unsafe { dlsym(RTLD_DEFAULT, s.as_ptr()) };
    if !p.is_null() {
        Ok(p as *const _)
    } else {
        Err(format!("dlsym(\"{name}\") returned NULL").into())
    }
}
