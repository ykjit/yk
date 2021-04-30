//! A wrapper around llvm::symbolizer::LLVMSymbolizer.

use libc::free;
use std::{
    ffi::{c_void, CStr},
    ops::Drop,
    os::raw::c_char,
};

extern "C" {
    fn __yk_symbolizer_new() -> *mut c_void;
    fn __yk_symbolizer_free(symbolizer: *mut c_void);
    fn __yk_symbolizer_find_code_sym(symbolizer: *mut c_void, addr: usize) -> *mut c_char;
}

pub struct Symbolizer(*mut c_void);

impl Symbolizer {
    pub fn new() -> Self {
        Self(unsafe { __yk_symbolizer_new() })
    }

    /// Returns the name of the symbol at the provided virtual address.
    pub fn find_code_sym(&self, vaddr: usize) -> Option<String> {
        let ptr = unsafe { __yk_symbolizer_find_code_sym(self.0, vaddr) };
        if ptr.is_null() {
            None
        } else {
            let ret = {
                let sym = unsafe { CStr::from_ptr(ptr) };
                String::from(sym.to_str().unwrap())
            };
            unsafe { free(ptr as *mut _) };
            Some(ret)
        }
    }
}

impl Drop for Symbolizer {
    fn drop(&mut self) {
        unsafe { __yk_symbolizer_free(self.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::Symbolizer;

    extern "C" {
        fn getuid();
    }

    #[inline(never)]
    fn symbolize_me_mangled() {}

    // This function has a different signature to the one above to prevent LLVM from merging the
    // functions (and their symbols) when optimising in --release mode.
    #[inline(never)]
    #[no_mangle]
    fn symbolize_me_unmangled() -> u8 {
        1
    }

    #[test]
    fn find_code_sym_mangled() {
        let f_addr = symbolize_me_mangled as *const fn() as *const _ as usize;
        let s = Symbolizer::new();
        let sym = s.find_code_sym(f_addr).unwrap();
        // The symbol will be suffixed with an auto-generated module name, e.g.:
        // ykllvmwrap::symbolizer::tests::symbolize_me_mangled::hc7a76ddceae6f9c4
        assert!(sym.starts_with("ykllvmwrap::symbolizer::tests::symbolize_me_mangled::"));
        let elems = sym.split("::");
        assert_eq!(elems.count(), 5);
    }

    #[test]
    fn find_code_sym_unmangled() {
        let f_addr = symbolize_me_unmangled as *const fn() as *const _ as usize;
        let s = Symbolizer::new();
        let sym = s.find_code_sym(f_addr).unwrap();
        assert_eq!(sym, "symbolize_me_unmangled");
    }

    #[test]
    fn find_code_sym_libc() {
        let f_addr = getuid as *const fn() as *const _ as usize;
        let s = Symbolizer::new();
        let sym = s.find_code_sym(f_addr).unwrap();
        assert_eq!(sym, "getuid");
    }

    #[test]
    fn find_code_sym_not_found() {
        // Virtual address 0x1 is obviously nonsense.
        assert!(Symbolizer::new().find_code_sym(1).is_none());
    }
}
