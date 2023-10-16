use std::os::raw::{c_int, c_void};

use std::{ffi::CString, ptr::null_mut};

use libc::{dlsym, free, malloc, pthread_attr_t, pthread_create, pthread_t};

const SHADOW_STACK_SIZE: usize = 1000000;

#[derive(Debug)]
struct ThreadRoutine {
    pub args: *mut c_void,
    pub start_routine: extern "C" fn(*mut c_void) -> *mut c_void,
}

extern "C" fn wrap_thread_routine(arg: *mut c_void) -> *mut c_void {
    let str = CString::new("shadowstack_0").unwrap();
    let shadowstack_symbol_addr = unsafe {
        //  Obtain address of a shadowstack_0 symbol
        dlsym(null_mut(), str.as_ptr() as *const i8)
    };
    if shadowstack_symbol_addr.is_null() {
        panic!("Unable to find shadowstack address")
    }
    let stack_addr = unsafe {
        // Allocate stack
        malloc(SHADOW_STACK_SIZE)
    };
    if stack_addr.is_null() {
        panic!("Unable allocate stack")
    }
    unsafe {
        // Set shadowstack symbol with new allocated stack
        *(shadowstack_symbol_addr as *mut *mut c_void) = stack_addr;
    }
    let thread_routine = unsafe {
        // Obtain ThreadRoutine struct
        Box::from_raw(arg as *mut ThreadRoutine)
    };

    // Call original thread routine
    let result = (thread_routine.start_routine)(thread_routine.args);

    unsafe {
        // Free allocated stack
        free(stack_addr)
    }

    result
    
}

#[no_mangle]
pub extern "C" fn __wrap_pthread_create(
    thread: *mut pthread_t,
    attr: *const pthread_attr_t,
    start_routine: extern "C" fn(*mut c_void) -> *mut c_void,
    args: *mut c_void,
) -> c_int {
    let ptr = Box::into_raw(Box::new(ThreadRoutine {
        args,
        start_routine,
    }));
    println!(" [YK] PTR {:p}", ptr);
    unsafe {
        return pthread_create(thread, attr, wrap_thread_routine, ptr as *mut c_void);
    }
}
