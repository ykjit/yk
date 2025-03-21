use std::os::raw::{c_int, c_void};

use libc::{dlsym, free, malloc, pthread_create};
use std::{ffi::CString, ptr::null_mut};

// The size of the shadow stack. This is the same size as the default shadow stack in ykllvm.
const SHADOW_STACK_SIZE: usize = 1000000;

#[derive(Debug)]
struct Target {
    pub func: extern "C" fn(*mut c_void) -> *mut c_void,
    pub arg: *mut c_void,
}

extern "C" fn wrap_thread_routine(tgt: *mut c_void) -> *mut c_void {
    let str = CString::new("shadowstack_0").unwrap();
    let tgt = unsafe { Box::from_raw(tgt as *mut Target) };
    // Obtain address of a shadowstack_0 symbol
    let shadowstack_symbol_addr = unsafe { dlsym(null_mut(), str.as_ptr()) };
    if shadowstack_symbol_addr.is_null() {
        panic!("Unable to find shadowstack address")
    }
    let newsstack = unsafe { malloc(SHADOW_STACK_SIZE) };
    if newsstack.is_null() {
        panic!("Unable to allocate stack")
    }
    unsafe {
        // Set shadowstack symbol with new allocated stack
        *(shadowstack_symbol_addr as *mut *mut c_void) = newsstack;
    }
    let ret = (tgt.func)(tgt.arg);
    unsafe { free(newsstack) };
    ret
}

#[no_mangle]
pub extern "C" fn __wrap_pthread_create(
    thread: *mut libc::pthread_t,
    attr: *const libc::pthread_attr_t,
    start_routine: extern "C" fn(*mut c_void) -> *mut c_void,
    arg: *mut c_void,
) -> c_int {
    let tgt = Box::new(Target {
        func: start_routine,
        arg,
    });
    unsafe {
        pthread_create(
            thread,
            attr,
            wrap_thread_routine,
            Box::into_raw(tgt) as *mut c_void,
        )
    }
}

#[no_mangle]
pub extern "C" fn __wrap_pthread_exit(_retval: *mut c_void) {
    // FIXME: Using `pthread_exit` doesn't return to `wrap_thread_routine` and thus doesn't free
    // the newly created shadowstack.
    todo!("No support for `pthread_exit` yet.");
}
