use std::os::raw::{c_int, c_void};

#[no_mangle]
pub extern "C" fn __wrap_pthread_create(
    _: *mut c_void,
    _: *const c_void,
    _: extern "C" fn(*mut c_void) -> *mut c_void,
    _: *mut c_void,
) -> c_int {
    todo!("Allocate and set thread-local ShadowStack instance. No support for threads yet!");
}
