//! Trace location: track the state of a program location (counting, tracing, compiled, etc).

use std::{
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use crate::{
    compile::CompiledTrace,
    mt::{HotThreshold, SideTraceInfo, TraceFailureThreshold, MT},
};
use parking_lot::Mutex;

#[cfg(target_pointer_width = "64")]
const STATE_TAG: usize = 0b1; // All of the tag data must fit in this.
#[cfg(target_pointer_width = "64")]
const STATE_NUM_BITS: usize = 1;

/// Because hot locations will be most common, we save ourselves the effort of ANDing bits away by
/// having `STATE_HOT` be 0, expecting that `ptr & !0` will be optimised to just `ptr`.
const STATE_HOT: usize = 0;
/// In the not hot state, we have to do `(inner & !1) >> STATE_NUM_BITS` to derive the count.
const STATE_NOT_HOT: usize = 0b1;

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
    // A Location is a state machine which operates as follows (where Counting is the start state):
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
    //  └───────────│   Compiled   │◀────────────┐
    //              └──────────────┘             │
    //                │                          │
    //                │ guard failed             │
    //                ▼                          │
    //              ┌──────────────┐             │
    //              │  SideTracing │─────────────┘
    //              └──────────────┘
    //
    // We hope that a Location soon reaches the `Compiled` state (aka "the happy state") and stays
    // there. However, many Locations will not be used frequently enough to reach such a state, so
    // we don't want to waste resources on them.
    //
    // We therefore encode a Location as a tagged integer: in the initial (Counting) state, no
    // memory is allocated; if the location is used frequently enough it becomes hot, memory
    // is allocated for it, and a pointer stored instead of an integer. Note that once memory for a
    // hot location is allocated, it can only be (scheduled for) deallocation when a Location
    // is dropped, as the Location may have handed out `&` references to that allocated memory.
    //
    // The layout of a Location is as follows: bit 0 = <STATE_NOT_HOT|STATE_HOT>; bits 1..<machine
    // width> = payload. In the `STATE_NOT_HOT` state, the payload is an integer; in a `STATE_HOT`
    // state, the payload is a pointer from `Arc::into_raw::<Mutex<HotLocation>>()`.
    inner: AtomicUsize,
}

impl Location {
    /// Create a new location.
    pub fn new() -> Self {
        // Locations start in the counting state with a count of 0.
        Self {
            inner: AtomicUsize::new(STATE_NOT_HOT),
        }
    }

    /// If `self` is in the `Counting` state, return its count, or `None` otherwise.
    pub(crate) fn count(&self) -> Option<HotThreshold> {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_NOT_HOT != 0 {
            // For the `as` to be safe, `HotThreshold` can't be bigger than `usize`
            debug_assert!(mem::size_of::<HotThreshold>() <= mem::size_of::<usize>());
            Some((x >> STATE_NUM_BITS) as HotThreshold)
        } else {
            None
        }
    }

    /// Change `self`s count to `new` if: `self` is in the `Counting` state; and the current count
    /// is `old`. If the transition is successful, return `true`.
    pub(crate) fn count_set(&self, old: HotThreshold, new: HotThreshold) -> bool {
        // `HotThreshold` must be unsigned
        debug_assert_eq!(HotThreshold::MIN, 0);
        // `HotThreshold` can't be bigger than `usize`
        debug_assert!(mem::size_of::<HotThreshold>() <= mem::size_of::<usize>());
        // The particular value of `new` must fit in the bits we have available.
        debug_assert!((new as usize)
            .checked_shl(u32::try_from(STATE_NUM_BITS).unwrap())
            .is_some());

        self.inner
            .compare_exchange_weak(
                ((old as usize) << STATE_NUM_BITS) | STATE_NOT_HOT,
                ((new as usize) << STATE_NUM_BITS) | STATE_NOT_HOT,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Change `self` to be a [HotLocation] `hl` if: `self` is in the `Counting` state; and the
    /// count is `old`. If the transition is successful, return a clone of the [Arc] that now
    /// wraps the [HotLocation].
    pub(crate) fn count_to_hot_location(
        &self,
        old: HotThreshold,
        hl: HotLocation,
    ) -> Option<Arc<Mutex<HotLocation>>> {
        let hl = Arc::new(Mutex::new(hl));
        let cl: *const Mutex<HotLocation> = Arc::into_raw(Arc::clone(&hl));
        debug_assert_eq!((cl as usize) & !STATE_TAG, cl as usize);
        match self.inner.compare_exchange(
            ((old as usize) << STATE_NUM_BITS) | STATE_NOT_HOT,
            (cl as usize) | STATE_HOT,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => Some(hl),
            Err(_) => {
                unsafe {
                    Arc::from_raw(cl);
                }
                None
            }
        }
    }

    /// If `self` has a [HotLocation] return a reference to the `Mutex` that directly wraps it, or
    /// `None` otherwise.
    pub(crate) fn hot_location(&self) -> Option<&Mutex<HotLocation>> {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_NOT_HOT == 0 {
            // `Arc::into_raw::<Mutex<T>>` returns `*mut Mutex<T>` so the address we're wrapping is
            // a pointer to the `Mutex` itself. By returning a `&` reference we ensure that the
            // reference to the `Mutex` can't outlive this `Location`.
            Some(unsafe { &*(x as *const _) })
        } else {
            None
        }
    }

    /// If `self` has a [HotLocation] return a clone of the [Arc] that wraps it, or `None`
    /// otherwise.
    pub(crate) fn hot_location_arc_clone(&self) -> Option<Arc<Mutex<HotLocation>>> {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_NOT_HOT == 0 {
            let raw = unsafe { Arc::from_raw(x as *mut _) };
            let cl = Arc::clone(&raw);
            mem::forget(raw);
            Some(cl)
        } else {
            None
        }
    }
}

impl Drop for Location {
    fn drop(&mut self) {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_NOT_HOT == 0 {
            drop(unsafe { Arc::from_raw(x as *mut Mutex<HotLocation>) });
        }
    }
}

#[derive(Debug)]
pub(crate) struct HotLocation {
    pub(crate) kind: HotLocationKind,
    pub(crate) trace_failure: TraceFailureThreshold,
}

impl HotLocation {
    /// Mark a trace starting at this `HotLocation` as having failed. The return value indicates
    /// whether further traces for this location should be generated or not.
    pub(crate) fn trace_failed(&mut self, mt: &Arc<MT>) -> TraceFailed {
        if self.trace_failure < mt.trace_failure_threshold() {
            self.trace_failure += 1;
            self.kind = HotLocationKind::Tracing;
            TraceFailed::KeepTrying
        } else {
            self.kind = HotLocationKind::DontTrace;
            TraceFailed::DontTrace
        }
    }
}

/// A `Location`'s non-counting states.
#[derive(Debug)]
pub(crate) enum HotLocationKind {
    /// Points to executable machine code that can be executed instead of the interpreter for this
    /// HotLocation.
    Compiled(Arc<dyn CompiledTrace>),
    /// A trace for this HotLocation is being compiled in another trace. When compilation is
    /// complete, the compiling thread will update the state of this HotLocation.
    Compiling,
    /// This HotLocation has encountered problems (e.g. traces which are too long) and shouldn't be
    /// traced again.
    DontTrace,
    /// This HotLocation started a trace which is ongoing.
    Tracing,
    /// While executing JIT compiled code, a guard failed often enough for us to want to generate a
    /// side trace for this HotLocation.
    SideTracing(
        Arc<dyn CompiledTrace>,
        SideTraceInfo,
        Arc<dyn CompiledTrace>,
    ),
}

/// When a [HotLocation] has failed to compile a valid trace, should the [HotLocation] be tried
/// again or not?
pub(crate) enum TraceFailed {
    KeepTrying,
    DontTrace,
}
