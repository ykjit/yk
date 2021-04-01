//! Tests for finding symbols.

use libc::{self, c_void};
use std::ptr;
use untraced_api::find_symbol;

// Test finding a symbol in a shared object.
#[test]
fn find_symbol_shared() {
    assert!(find_symbol("printf") == libc::printf as *mut c_void);
}

// Test finding a symbol in the main binary.
// For this to work the binary must have been linked with `--export-dynamic`, which ykrustc
// appends to the linker command line.
#[test]
#[no_mangle]
fn find_symbol_main() {
    assert!(find_symbol("find_symbol_main") == find_symbol_main as *mut c_void);
}

// Check that a non-existent symbol cannot be found.
#[test]
fn find_nonexistent_symbol() {
    assert_eq!(find_symbol("__xxxyyyzzz__"), ptr::null_mut());
}
