use libc::{mprotect, size_t, sysconf, PROT_EXEC, PROT_READ, PROT_WRITE};
use std::mem;
use std::{ffi::c_void, sync::Once};

#[cfg(tracer_swt)]
use crate::yk_trace_basicblock;

#[cfg(tracer_swt)]
static ORIGINAL_INSTRUCTIONS_INIT: Once = Once::new();
#[cfg(tracer_swt)]
static mut ORIGINAL_INSTRUCTIONS: [u8; 1] = [0; 1];
#[cfg(tracer_swt)]
static mut PATCH_INSTRUCTIONS: [u8; 1] = [0xC3];

#[cfg(tracer_swt)]
unsafe fn save_original_instructions(
    function_ptr: usize,
    instructions: *mut u8,
    num_of_instructions: usize,
) {
    let func_ptr: *const () = function_ptr as *const ();
    std::ptr::copy_nonoverlapping(func_ptr as *const u8, instructions, num_of_instructions);
}

#[cfg(tracer_swt)]
unsafe fn patch_function(function_ptr: usize, code: *const u8, size: size_t) {
    let page_size = sysconf(libc::_SC_PAGESIZE) as usize;

    let func_address = ((function_ptr as usize) & !(page_size - 1)) as *mut c_void;
    let page_size_aligned = (((function_ptr as usize) + mem::size_of_val(&function_ptr))
        - (func_address as usize)) as usize;

    // Set function memory region to be writable
    let result = mprotect(
        func_address,
        page_size_aligned,
        PROT_READ | PROT_WRITE | PROT_EXEC,
    );
    if result != 0 {
        panic!("Failed to change memory protection to be writable");
    }

    // Set function memory region back to be non-writable
    std::ptr::copy_nonoverlapping(code, function_ptr as *mut u8, size);

    let result = mprotect(func_address, page_size_aligned, PROT_READ | PROT_EXEC);
    if result != 0 {
        panic!("Failed to change memory protection to not writable");
    }
}

#[cfg(tracer_swt)]
pub(crate) unsafe fn patch_trace_function() {
    ORIGINAL_INSTRUCTIONS_INIT.call_once(|| {
        save_original_instructions(
            yk_trace_basicblock as usize,
            ORIGINAL_INSTRUCTIONS.as_mut_ptr(),
            1,
        );
    });

    patch_function(yk_trace_basicblock as usize, PATCH_INSTRUCTIONS.as_ptr(), 1);
}

#[cfg(tracer_swt)]
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
