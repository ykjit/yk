//! Software constant recorder.
//!
//! This is used for promoting variables to constants when compiling a trace.

use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    ffi::{c_char, CStr},
};

thread_local! {
    /// The thread's value recorder.
    ///
    /// This has to be specific to a thread to avoid cross-talk from other threads which may be
    /// executing functions with promoted arguments at the same time.
    pub static VAL_REC: RefCell<ValueRecorder> = RefCell::new(ValueRecorder::default());
}

/// The unique identifier for a promoted variable.
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct PromoteKey {
    /// The name of the function whose argument will be promoted.
    func_name: String,
    /// The index of the argument to promote.
    arg_idx: usize,
}

impl PromoteKey {
    pub fn new(func_name: String, arg_idx: usize) -> Self {
        Self { func_name, arg_idx }
    }
}

/// A value recorder observes and records constant values during tracing. The trace compiler
/// queries this when building a trace to replace promoted variables with the observed constants.
#[derive(Debug, Default)]
pub struct ValueRecorder {
    // `true` when recording promotions.
    record_enable: bool,
    // Promote buffers.
    //
    // For each function we store a list of observed values. The list is ordered from earliest
    // observation to oldest.
    //
    // FIXME: currently you may only promote pointer-sized integers.
    pbufs: HashMap<PromoteKey, VecDeque<usize>>,
}

impl ValueRecorder {
    /// Enable (`record=true`) or disable (`record=false`) observations.
    pub fn record_enable(&mut self, record: bool) {
        debug_assert_ne!(self.record_enable, record);
        self.record_enable = record;
    }

    /// Record the concrete value of variable to be promoted.
    pub fn record_const(&mut self, key: PromoteKey, val: usize) {
        // OPT: a quicker way than a boolean check probably exists.
        if self.record_enable {
            self.pbufs.entry(key).or_default().push_back(val);
        }
    }

    /// Remove and return he oldest observation of the specified variable.
    pub fn lookup_const(&mut self, key: &PromoteKey) -> usize {
        debug_assert!(!self.record_enable);
        if let Some(buf) = self.pbufs.get_mut(&key) {
            buf.pop_front()
                .expect(&format!("promote buffer undeflow for key: {:?}", key))
        } else {
            panic!("non-existent promote buffer underflow for key {:?}!", key);
        }
    }
}

/// For the current thread, enable or disable constant recording.
pub fn thread_record_enable(record: bool) {
    VAL_REC.with_borrow_mut(|r| {
        r.record_enable(record);
    })
}

/// For the current thread, remove and return he oldest observation of the specified variable.
#[no_mangle]
pub extern "C" fn __yk_lookup_promote_usize(func_name: *const c_char, arg_idx: usize) -> usize {
    let func_name = unsafe { CStr::from_ptr(func_name) };
    let key = PromoteKey::new(func_name.to_str().unwrap().to_owned(), arg_idx);
    VAL_REC.with_borrow_mut(|r| r.lookup_const(&key))
}

/// For the current thread, record the concrete value of a variable to be promoted.
///
/// `pairs` is a series of `(arg_idx: u64, val: u64)` pairs.
#[no_mangle]
pub unsafe extern "C" fn __yk_record_promote_usize(
    func_name: *const c_char,
    num_pairs: usize,
    mut pairs: ...
) {
    let func_name = unsafe { CStr::from_ptr(func_name) };
    for _ in 0..num_pairs {
        let (arg_idx, val) = unsafe { (pairs.arg(), pairs.arg()) };
        let key = PromoteKey::new(func_name.to_str().unwrap().to_owned(), arg_idx);
        VAL_REC.with_borrow_mut(|r| {
            r.record_const(key, val);
        });
    }
}
