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
    unsafe {
        //  Obtain address of a shadowstack_0 symbol
        let shadowstack_addr: *mut libc::c_void = dlsym(null_mut(), str.as_ptr() as *const i8);
        if shadowstack_addr.is_null() {
            panic!("Unable to find shadowstack address!")
        }
        // Allocate stack
        let stack_addr = malloc(SHADOW_STACK_SIZE);
        if stack_addr.is_null() {
            panic!("Unable allocate stack!")
        }
        // Set shadowstack addr with new allocated stack
        *(shadowstack_addr as *mut *mut c_void) = stack_addr;
        let thread_routine: &ThreadRoutine = (arg as *mut ThreadRoutine)
            .as_ref()
            .expect("Thread routine function and args data.");
        // Call original thread routine
        let result = (thread_routine.start_routine)(thread_routine.args);
        free(stack_addr);
        result
    }
}

#[no_mangle]
pub extern "C" fn __wrap_pthread_create(
    thread: *mut pthread_t,
    attr: *const pthread_attr_t,
    start_routine: extern "C" fn(*mut c_void) -> *mut c_void,
    args: *mut c_void,
) -> c_int {
    unsafe {
        let routine_args = &ThreadRoutine {
            args,
            start_routine,
        } as *const ThreadRoutine as *mut c_void;
        return pthread_create(thread, attr, wrap_thread_routine, routine_args);
    }
}
