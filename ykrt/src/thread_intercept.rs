use std::os::raw::{c_int, c_void};

use libc::{free, malloc, pthread_create};
use parking_lot::Mutex;
use std::cell::{OnceCell, RefCell};
use ykaddr::addr::symbol_to_ptr;

thread_local! {
    /// The start of the current thread's shadow stack
    static THREAD_SHADOW_START: OnceCell<*mut c_void> = const { OnceCell::new() };
}

// The size of the shadow stack. This is the same size as the default shadow stack in ykllvm.
const SHADOW_STACK_SIZE: usize = 1000000;

static SHADOW_STACKS: Mutex<RefCell<ShadowStacks>> = Mutex::new(RefCell::new(ShadowStacks::new()));

struct ShadowStackPtr(*mut c_void);
unsafe impl Sync for ShadowStackPtr {}
unsafe impl Send for ShadowStackPtr {}

struct ShadowStacks {
    stacks: Vec<ShadowStackPtr>,
}

impl ShadowStacks {
    const fn new() -> Self {
        ShadowStacks { stacks: Vec::new() }
    }

    fn register_current_thread(&mut self) {
        self.stacks.push(ShadowStackPtr(thread_shadow_start()));
    }
}

#[derive(Debug)]
struct Target {
    pub func: extern "C" fn(*mut c_void) -> *mut c_void,
    pub arg: *mut c_void,
}

/// Call a function for each shadow stack.
pub fn yk_foreach_shadowstack(f: extern "C" fn(*mut c_void, *mut c_void)) {
    for ptr in SHADOW_STACKS.lock().borrow().stacks.iter() {
        let end = ptr.0.wrapping_byte_add(SHADOW_STACK_SIZE);
        f(ptr.0.cast() as *mut c_void, end as *mut c_void);
    }
}

/// Return the start of the current thread's shadow stack.
fn thread_shadow_start() -> *mut c_void {
    symbol_to_ptr("shadowstack_0").expect("Unable to find shadowstack address") as *mut c_void
}

/// Return the start and end of the current thread's shadow stack.
pub fn yk_thread_shadowstack_bounds() -> (*mut c_void, *mut c_void) {
    let start = THREAD_SHADOW_START.with(|oc| {
        // For "non-main" threads, `THREAD_SHADOW_START` is guaranteed to be initialised by now.
        // For the main thread it may not be, so we have to use `get_or_init()`.
        *oc.get_or_init(thread_shadow_start)
    });
    let end = start.wrapping_byte_add(SHADOW_STACK_SIZE);
    (start, end)
}

// Called at program startup to register the shadowstack of the main thread.
pub fn yk_init() {
    SHADOW_STACKS.lock().borrow_mut().register_current_thread();
}

/// The function is called just after a new thread has been created by `pthread_create`. We use it
/// to create a new shadow stack, and then call the "real" `routine` passed to `pthread_create`.
extern "C" fn wrap_thread_start(tgt: *mut c_void) -> *mut c_void {
    let tgt = unsafe { Box::from_raw(tgt as *mut Target) };
    // Obtain address of a shadowstack_0 symbol
    let shadowstack_symbol_addr = thread_shadow_start();
    let newsstack = unsafe { malloc(SHADOW_STACK_SIZE) };
    if newsstack.is_null() {
        panic!("Unable to allocate stack")
    }
    unsafe {
        // Set shadowstack symbol with new allocated stack
        *(shadowstack_symbol_addr as *mut *mut c_void) = newsstack;
        SHADOW_STACKS.lock().borrow_mut().register_current_thread();
    }
    THREAD_SHADOW_START.with(|oc| oc.set(newsstack).unwrap());
    // Call the `routine` passed to `pthread_create`.
    let ret = (tgt.func)(tgt.arg);
    unsafe { free(newsstack) };
    ret
}

/// Wraps system pthread create
#[unsafe(no_mangle)]
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
            wrap_thread_start,
            Box::into_raw(tgt) as *mut c_void,
        )
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __wrap_pthread_exit(_retval: *mut c_void) {
    // FIXME: Using `pthread_exit` doesn't return to `wrap_thread_routine` and thus doesn't free
    // the newly created shadowstack.
    todo!("No support for `pthread_exit` yet.");
}
