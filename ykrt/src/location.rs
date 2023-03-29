//! Trace location: track the state of a program location (counting, tracing, compiled, etc).

use std::{
    convert::TryFrom,
    sync::{
        atomic::{AtomicPtr, AtomicUsize, Ordering},
        Arc,
    },
};

use crate::mt::{HotThreshold, TraceFailureThreshold};
use parking_lot::Mutex;
use parking_lot_core::{
    park, unpark_one, ParkResult, SpinWait, UnparkResult, UnparkToken, DEFAULT_PARK_TOKEN,
};
use strum::EnumDiscriminants;
use yktrace::CompiledTrace;

/// A `Location` stores state that the meta-tracer needs to identify hot loops and run associated
/// machine code.
///
/// Each position in the end user's program that may be a control point (i.e. the possible start of
/// a trace) must have an associated `Location`. The `Location` does not need to be at a stable
/// address in memory and can be freely moved.
///
/// Program positions that can't be control points don't need an associated `Location`. For
/// interpreters that can't (or don't want) to be as selective, a simple (if moderately wasteful)
/// mechanism is for every bytecode or AST node to have its own `Location` (even for bytecodes or
/// nodes that can't be control points).
#[repr(C)]
#[derive(Debug)]
pub struct Location {
    // A Location is a state machine which operates as follows (where Counting is the start
    // state):
    //
    //              ┌──────────────┐
    //              │              │─────────────┐
    //   reprofile  │   Counting   │             │
    //  ┌──────────▶│              │◀────────────┘
    //  │           └──────────────┘    increment
    //  │             │                 count
    //  │             │ start tracing
    //  │             ▼
    //  │           ┌──────────────┐
    //  │           │              │ incomplete  ┌─────────────┐
    //  │           │   Tracing    │────────────▶│  DontTrace  │
    //  │           │              │             └─────────────┘
    //  │           └──────────────┘
    //  │             │ start compiling trace
    //  │             │ in thread
    //  │             ▼
    //  │           ┌──────────────┐             ┌───────────┐
    //  │           │  Compiling   │────────────▶│  Dropped  │
    //  │           └──────────────┘             └───────────┘
    //  │             │
    //  │             │ trace compiled
    //  │             ▼
    //  │           ┌──────────────┐
    //  └───────────│   Compiled   │
    //              └──────────────┘
    //
    // We hope that a Location soon reaches the `Compiled` state (aka "the happy state") and stays
    // there.
    //
    // The state machine is encoded in a `usize` in a not-entirely-simple way, as we don't want to
    // allocate any memory for Locations that do not become hot. The layout is as follows (on a 64
    // bit machine):
    //
    //   bit(s) | 63..3   | 2           | 1         | 0
    //          | payload | IS_COUNTING | IS_PARKED | IS_LOCKED
    //
    // In the `Counting` state, `IS_COUNTING` is set to 1, and `IS_PARKED` and `IS_LOCKED` are
    // unused and must remain set at 0. The payload representing the count is incremented
    // locklessly. All other states have `IS_COUNTING` set to 0 and the payload is the address of a
    // `Box<HotLocation>`, access to which is controlled by `IS_LOCKED`.
    //
    // The possible combinations of the counting and mutex bits are thus as follows:
    //
    //   payload          | IS_COUNTING | IS_PARKED | IS_LOCKED | Notes
    //   -----------------+-----------+---------+---------+-----------------------------------
    //   count            | 1           | 0         | 0         | Start state
    //                    | 1           | 0         | 1         | Illegal state
    //                    | 1           | 1         | 0         | Illegal state
    //                    | 1           | 1         | 1         | Illegal state
    //   Box<HotLocation> | 0           | 0         | 0         | Not locked, no-one waiting
    //   Box<HotLocation> | 0           | 0         | 1         | Locked, no thread(s) waiting
    //   Box<HotLocation> | 0           | 1         | 0         | Not locked, thread(s) waiting
    //   Box<HotLocation> | 0           | 1         | 1         | Locked, thread(s) waiting
    //
    // where `count` is an integer and `Box<HotLocation>` is a pointer to a `malloc`ed block of
    // memory in the heap containing a `HotLocation` enum.
    //
    // The precise semantics of locking and, in particular, parking are subtle: interested readers
    // are directed to https://github.com/Amanieu/parking_lot/blob/master/src/raw_mutex.rs#L33 for
    // a more precise definition.
    inner: AtomicUsize,
}

impl Location {
    /// Create a new location.
    pub fn new() -> Self {
        // Locations start in the counting state with a count of 0.
        Self {
            inner: AtomicUsize::new(LocationInner::new().x),
        }
    }

    /// Return this Location's internal state.
    pub(super) fn load(&self, order: Ordering) -> LocationInner {
        LocationInner {
            x: self.inner.load(order),
        }
    }

    pub(super) fn compare_exchange(
        &self,
        current: LocationInner,
        new: LocationInner,
        success: Ordering,
        failure: Ordering,
    ) -> Result<LocationInner, LocationInner> {
        match self
            .inner
            .compare_exchange(current.x, new.x, success, failure)
        {
            Ok(x) => Ok(LocationInner { x }),
            Err(x) => Err(LocationInner { x }),
        }
    }

    pub(super) fn compare_exchange_weak(
        &self,
        current: LocationInner,
        new: LocationInner,
        success: Ordering,
        failure: Ordering,
    ) -> Result<LocationInner, LocationInner> {
        match self
            .inner
            .compare_exchange_weak(current.x, new.x, success, failure)
        {
            Ok(x) => Ok(LocationInner { x }),
            Err(x) => Err(LocationInner { x }),
        }
    }

    /// Locks this `State` with `Acquire` ordering. If the location was in, or moves to, the
    /// Counting state this will return `Err`.
    pub(super) fn lock(&self) -> Result<LocationInner, ()> {
        {
            let ls = self.load(Ordering::Relaxed);
            if ls.is_counting() {
                return Err(());
            }
            let new_ls = ls.with_lock().with_unparked();
            if self
                .inner
                .compare_exchange_weak(
                    ls.with_unlock().with_unparked().x,
                    new_ls.x,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return Ok(new_ls);
            }
        }

        let mut spinwait = SpinWait::new();
        let mut ls = self.load(Ordering::Relaxed);
        loop {
            if ls.is_counting() {
                return Err(());
            }

            if !ls.is_locked() {
                let new_ls = ls.with_lock();
                match self.inner.compare_exchange_weak(
                    ls.x,
                    new_ls.x,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Ok(new_ls),
                    Err(x) => ls = LocationInner::from_usize(x),
                }
                continue;
            }

            if !ls.is_parked() {
                if spinwait.spin() {
                    ls = self.load(Ordering::Relaxed);
                    continue;
                } else if let Err(x) = self.inner.compare_exchange_weak(
                    ls.x,
                    ls.with_parked().x,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    ls = LocationInner::from_usize(x);
                    continue;
                }
            }

            let key = unsafe { ls.hot_location() } as *const _ as usize;
            debug_assert_ne!(key, 0);
            let validate = || {
                let ls = self.load(Ordering::Relaxed);
                ls.is_locked() && ls.is_parked()
            };
            let before_sleep = || {};
            let timed_out = |_, _| unreachable!();
            match unsafe {
                park(
                    key,
                    validate,
                    before_sleep,
                    timed_out,
                    DEFAULT_PARK_TOKEN,
                    None,
                )
            } {
                ParkResult::Unparked(TOKEN_HANDOFF) => return Ok(self.load(Ordering::Relaxed)),
                ParkResult::Invalid | ParkResult::Unparked(_) => (),
                ParkResult::TimedOut => unreachable!(),
            }

            // Loop back and try locking again
            spinwait.reset();
            ls = self.load(Ordering::Relaxed);
        }
    }

    /// Unlocks this `State` with `Release` ordering.
    pub(super) fn unlock(&self) {
        let ls = self.load(Ordering::Relaxed);
        debug_assert!(ls.is_locked());
        debug_assert!(!ls.is_counting());
        if self
            .inner
            .compare_exchange(
                ls.with_unparked().x,
                ls.with_unparked().with_unlock().x,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            return;
        }

        // At this point we know another thread has parked itself.
        let key = unsafe { ls.hot_location() } as *const _ as usize;
        debug_assert_ne!(key, 0);
        let callback = |result: UnparkResult| {
            if result.unparked_threads != 0 && result.be_fair {
                if !result.have_more_threads {
                    debug_assert!(ls.is_locked());
                    self.inner.store(ls.with_unparked().x, Ordering::Relaxed);
                }
                return TOKEN_HANDOFF;
            }

            if result.have_more_threads {
                self.inner
                    .store(ls.with_unlock().with_parked().x, Ordering::Release);
            } else {
                self.inner
                    .store(ls.with_unparked().with_unlock().x, Ordering::Release);
            }
            TOKEN_NORMAL
        };
        unsafe {
            unpark_one(key, callback);
        }
    }

    /// Try obtaining a lock, returning the new [LocationInner] if successful. If a lock wasn't
    /// obtained, the `Location` was either: in the Counting state (for which locks are
    /// nonsensical); or another thread held the lock.
    pub(super) fn try_lock(&self) -> Option<LocationInner> {
        let mut ls = self.load(Ordering::Relaxed);
        loop {
            if ls.is_counting() || ls.is_locked() {
                return None;
            }
            let new_ls = ls.with_lock();
            match self.compare_exchange_weak(ls, new_ls, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return Some(new_ls),
                Err(x) => ls = x,
            }
        }
    }
}

impl Drop for Location {
    fn drop(&mut self) {
        let ls = self.load(Ordering::Relaxed);
        if !ls.is_counting() {
            self.lock().unwrap();
            let ls = self.load(Ordering::Relaxed);
            let hl = unsafe { ls.hot_location() };
            if let HotLocationKind::Compiled(_) = hl.kind {
                // FIXME: we can't drop this memory as another thread may still be executing the
                // trace that's pointed to. There should be a ref count that we decrement and free
                // memory if it reaches zero.
                self.unlock();
            } else if let HotLocationKind::Compiling(_)
            | HotLocationKind::DontTrace
            | HotLocationKind::Tracing(_) = hl.kind
            {
                self.unlock();
                unsafe {
                    let _ = Box::from_raw(hl);
                }
            } else {
                unreachable!();
            }
        }
    }
}

#[cfg(target_pointer_width = "64")]
const STATE_TAG: usize = 0b111; // All of the other tag data must fit in this.
#[cfg(target_pointer_width = "64")]
const STATE_NUM_BITS: usize = 3;

const STATE_IS_LOCKED: usize = 0b001;
const STATE_IS_PARKED: usize = 0b010;
const STATE_IS_COUNTING: usize = 0b100;

const TOKEN_NORMAL: UnparkToken = UnparkToken(0);
const TOKEN_HANDOFF: UnparkToken = UnparkToken(1);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) struct LocationInner {
    x: usize,
}

impl LocationInner {
    /// Return a new `State` in the counting phase with a count 0.
    pub(super) fn new() -> Self {
        LocationInner {
            x: STATE_IS_COUNTING,
        }
    }

    fn from_usize(x: usize) -> Self {
        LocationInner { x }
    }

    pub(super) fn is_locked(&self) -> bool {
        self.x & STATE_IS_LOCKED != 0
    }

    pub(super) fn is_parked(&self) -> bool {
        self.x & STATE_IS_PARKED != 0
    }

    /// Is the Location in the counting or a non-counting state?
    pub(super) fn is_counting(&self) -> bool {
        self.x & STATE_IS_COUNTING != 0
    }

    /// If, and only if, the Location is in the counting state, return the current count.
    pub(super) fn count(&self) -> HotThreshold {
        debug_assert!(self.is_counting());
        debug_assert!(!self.is_locked());
        u32::try_from(self.x >> STATE_NUM_BITS).unwrap()
    }

    /// If this `State` is not counting, return its `HotLocation`. It is undefined behaviour to
    /// call this function if this `State` is in the counting phase and/or if this `State` is not
    /// locked.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn hot_location(&self) -> &mut HotLocation {
        debug_assert!(!self.is_counting());
        debug_assert!(self.is_locked());
        &mut *((self.x & !STATE_TAG) as *mut _)
    }

    /// Return a version of this `State` with the locked bit set.
    pub(super) fn with_lock(&self) -> LocationInner {
        LocationInner {
            x: self.x | STATE_IS_LOCKED,
        }
    }

    /// Return a version of this `State` with the locked bit unset.
    fn with_unlock(&self) -> LocationInner {
        LocationInner {
            x: self.x & !STATE_IS_LOCKED,
        }
    }

    /// Return a version of this `State` with the parked bit set.
    fn with_parked(&self) -> LocationInner {
        LocationInner {
            x: self.x | STATE_IS_PARKED,
        }
    }

    /// Return a version of this `State` with the parked bit unset.
    fn with_unparked(&self) -> LocationInner {
        LocationInner {
            x: self.x & !STATE_IS_PARKED,
        }
    }

    /// Return a version of this `State` with the count set to `count`. It is undefined behaviour
    /// to call this function if this `State` is not in the counting phase.
    pub(super) fn with_count(&self, count: HotThreshold) -> Self {
        debug_assert!(self.is_counting());
        debug_assert_eq!(count << STATE_NUM_BITS >> STATE_NUM_BITS, count);
        LocationInner {
            x: (self.x & STATE_TAG) | (usize::try_from(count).unwrap() << STATE_NUM_BITS),
        }
    }

    /// Set this `State`'s `HotLocation`. It is undefined behaviour for this `State` to already
    /// have a `HotLocation`.
    pub(super) fn with_hotlocation(&self, hl_ptr: *mut HotLocation) -> Self {
        debug_assert!(self.is_counting());
        let hl_ptr = hl_ptr as usize;
        debug_assert_eq!(hl_ptr & !STATE_TAG, hl_ptr);
        LocationInner {
            x: (self.x & (STATE_TAG & !STATE_IS_COUNTING)) | hl_ptr,
        }
    }
}

pub(crate) struct HotLocation {
    pub(crate) kind: HotLocationKind,
    pub(crate) trace_failure: TraceFailureThreshold,
}

/// A `Location`'s non-counting states.
#[derive(EnumDiscriminants)]
pub(crate) enum HotLocationKind {
    /// Points to executable machine code that can be executed instead of the interpreter for this
    /// HotLocation.
    Compiled(*const CompiledTrace),
    /// This HotLocation is being compiled in another thread: when compilation has completed the
    /// `Option` will change from `None` to `Some`.
    Compiling(Arc<Mutex<Option<Box<CompiledTrace>>>>),
    /// This HotLocation has encountered problems (e.g. traces which are too long) and shouldn't be
    /// traced again.
    DontTrace,
    /// This HotLocation started a trace which is ongoing.
    Tracing(Arc<AtomicPtr<HotLocation>>),
}
