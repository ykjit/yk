//! Software constant recorder.
//!
//! This is used for promoting values to constants when compiling a trace.

use std::{cell::RefCell, collections::VecDeque};

// FIXME: Try to move this into ykrt in the existing thread local.
thread_local! {
    /// The thread's value recorder.
    ///
    /// This has to be specific to a thread to avoid cross-talk from other threads which may be
    /// executing functions with promoted arguments at the same time.
    pub static VAL_REC: RefCell<ValueRecorder> = RefCell::new(ValueRecorder::default());
}

/// A value recorder observes and records constant values during tracing. The trace compiler
/// queries this when building a trace to replace promoted values with the observed constants.
#[derive(Debug, Default)]
pub struct ValueRecorder {
    // `true` when recording promotions.
    record_enable: bool,
    // A vector of to-be-promoted values, in the order their `yk_promote()` calls were encountered
    // in the trace through the AOT module.
    //
    // FIXME: currently you may only promote pointer-sized integers.
    pbuf: VecDeque<usize>,
}

impl ValueRecorder {
    /// Enable (`record=true`) or disable (`record=false`) observations.
    pub fn record_enable(&mut self, record: bool) {
        debug_assert_ne!(self.record_enable, record);
        self.record_enable = record;
    }

    /// Record a constant for a value that will be promoted.
    pub fn push(&mut self, val: usize) {
        if self.record_enable {
            self.pbuf.push_back(val);
        }
    }

    /// Remove and return the oldest recorded constant.
    pub fn pop(&mut self) -> usize {
        debug_assert!(!self.record_enable);
        self.pbuf.pop_front().expect("promote buffer undeflow")
    }
}

/// For the current thread, enable or disable constant recording.
pub fn thread_record_enable(record: bool) {
    VAL_REC.with_borrow_mut(|r| {
        r.record_enable(record);
    })
}

/// For the current thread, remove and return the oldest recorded value.
#[no_mangle]
pub extern "C" fn __yk_lookup_promote_usize() -> usize {
    VAL_REC.with_borrow_mut(|r| r.pop())
}

/// Promote a value.
///
/// When encountered during trace collection, the returned value will be considered constant in the
/// resulting compiled trace.
///
/// The user sees this as `yk_promote` via a macro.
#[no_mangle]
pub extern "C" fn __ykllvm_recognised_promote(val: usize) -> usize {
    VAL_REC.with_borrow_mut(|r| {
        r.push(val);
    });
    val
}
