//! This module provides functionality for dynamically patching and
//! restoring `yk_trace_basicblock` function at runtime. It includes
//! mechanisms to save the original instructions of a target function,
//! patch the target function with new instructions, and restore the
//! original instructions.
//!
//! This is particularly useful in scenarios such as software tracing,
//! where the tracing function may be dynamically modified to have
//! single return instruction and later restored to their original state
//! for performance gain.
//!
//! The module relies on low-level operations like memory protection
//! changes to manipulate function code safely. It is designed to be
//! used with `tracer_swt` configurations, enabling conditional
//! compilation for tracing functionalities.
//!
//! # Warning
//!
//! This module performs low-level memory operations and modifies the
//! execution flow of functions at runtime. Improper use can lead to
//! undefined behaviour, memory corruption, or crashes.

use crate::trace::swt::yk_trace_basicblock;
use libc::{mprotect, size_t, sysconf, PROT_EXEC, PROT_READ, PROT_WRITE, _SC_PAGESIZE};
use std::alloc::Layout;
use std::{ffi::c_void, sync::Once};

// This is used to ensure that the original instructions are only saved once.
static ORIGINAL_INSTRUCTIONS_INIT: Once = Once::new();

// Original instructions of the function that is patched with `PATCH_X86_INSTRUCTIONS`.
static mut ORIGINAL_INSTRUCTIONS: [u8; 1] = [0; 1];

// 0xC3 is a `ret` instruction on x86_64.
static mut PATCH_X86_INSTRUCTIONS: [u8; 1] = [0xC3];

/// This function is used to save the original instructions of a function to .
///
/// # Arguments
///
/// * `function_ptr` - A usize representing the memory address of the function.
/// * `instructions` - A mutable pointer to a u8 where the original instructions will be saved.
/// * `num_of_instructions` - A usize indicating the number of instructions to save.
///
unsafe fn save_original_instructions(
    function_ptr: usize,
    instructions: *mut u8,
    num_of_instructions: usize,
) {
    let func_ptr: *const () = function_ptr as *const ();
    std::ptr::copy_nonoverlapping(func_ptr as *const u8, instructions, num_of_instructions);
}

/// This function is used to patch a function instructions at runtime.
///
/// # Arguments
///
/// * `function_ptr` - A usize representing the memory address of the function to be patched.
/// * `code` - A constant pointer to a u8 vector where the new instructions are located.
/// * `size` - A size_t indicating the number of bytes to copy from `code`.
///
unsafe fn patch_function(function_ptr: usize, code: *const u8, size: size_t) {
    let page_size = sysconf(_SC_PAGESIZE) as usize;

    let page_address = (function_ptr & !(page_size - 1)) as *mut c_void;
    let start_offset = function_ptr - (page_address as usize) + size;

    match Layout::from_size_align(start_offset, page_size) {
        Ok(layout) => {
            // Set function memory page as writable
            let result = mprotect(page_address, layout.size(), PROT_READ | PROT_WRITE);
            if result != 0 {
                panic!("Failed to change memory protection to be writable");
            }
            // Copy the new code over
            std::ptr::copy_nonoverlapping(code, function_ptr as *mut u8, size);
            // Set function memory page as readable
            let result = mprotect(page_address, layout.size(), PROT_READ | PROT_EXEC);
            if result != 0 {
                panic!("Failed to change memory protection back to executable");
            }
        }
        Err(e) => {
            panic!(
                "Failed to create layout for fuction instuction patch: {:?}",
                e
            );
        }
    }
}

/// This function is used to patch the `yk_trace_basicblock`
/// function with a single `ret` (0xC3) instruction.
pub(crate) unsafe fn patch_trace_function() {
    ORIGINAL_INSTRUCTIONS_INIT.call_once(|| {
        save_original_instructions(
            yk_trace_basicblock as usize,
            ORIGINAL_INSTRUCTIONS.as_mut_ptr(),
            1,
        );
    });
    #[cfg(target_arch = "x86_64")]
    patch_function(
        yk_trace_basicblock as usize,
        PATCH_X86_INSTRUCTIONS.as_ptr(),
        1,
    );
}

/// This function is used to restore the original behavior of a
/// previously patched `yk_trace_basicblock` function.
pub(crate) unsafe fn restore_trace_function() {
    ORIGINAL_INSTRUCTIONS_INIT.call_once(|| {
        save_original_instructions(
            yk_trace_basicblock as usize,
            ORIGINAL_INSTRUCTIONS.as_mut_ptr(),
            1,
        );
    });
    patch_function(
        yk_trace_basicblock as usize,
        ORIGINAL_INSTRUCTIONS.as_ptr(),
        1,
    );
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    fn test_function() -> i32 {
        return 42;
    }

    #[test]
    fn test_runtime_patch() {
        unsafe {
            assert_eq!(test_function(), 42);
            save_original_instructions(
                test_function as usize,
                ORIGINAL_INSTRUCTIONS.as_mut_ptr(),
                1,
            );
            patch_function(test_function as usize, PATCH_X86_INSTRUCTIONS.as_ptr(), 1);
            assert_eq!(test_function(), 0);
            patch_function(test_function as usize, ORIGINAL_INSTRUCTIONS.as_ptr(), 1);
            assert_eq!(test_function(), 42);
        }
    }
}
