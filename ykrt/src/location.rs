//! Trace location: track the state of a program location (counting, tracing, compiled, etc).

use std::{
    mem,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use crate::{
    compile::{CompiledTrace, GuardIdx},
    mt::{HotThreshold, TraceCompilationErrorThreshold, MT},
};
use parking_lot::Mutex;

#[cfg(target_pointer_width = "64")]
const STATE_TAG_MASK: usize = 0b11; // All of the tag data must fit in this.
#[cfg(target_pointer_width = "64")]
const STATE_NUM_BITS: usize = 2;

const STATE_NULL: usize = 0b00;
/// The tag value for a not-yet-hot [Location]. Because null [Location]s have an inner value of 0,
/// this value *must* be non-zero. To derive the count of a not-yet-hot [Location], we have to do
/// `(inner & !1) >> STATE_NUM_BITS` to derive the count.
const STATE_NOT_HOT: usize = 0b01;
/// The tag value for a hot [Location]; its [HotLocation] address will be contained in the non-tag
/// bits.
const STATE_HOT: usize = 0b10;

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
    /// A Location is a state machine. "Null" locations are always "null". Non-"null" locations
    /// operate operate as follows (where Counting is the start state):
    ///
    /// ```text
    ///                                           │
    ///                                           │
    ///                                           ▼
    ///                                         ┌─────────────────────────────────────────────────────────────────────────────────────┐   increment count
    ///                                         │                                                                                     │ ──────────────────┐
    ///                                         │                                      Counting                                       │                   │
    ///                                         │                                                                                     │ ◀─────────────────┘
    ///                                         └─────────────────────────────────────────────────────────────────────────────────────┘
    ///                                           │                                ▲                          ▲
    ///                                           │ start tracing                  │ failed below threshold   │ failed below threshold
    ///                                           ▼                                │                          │
    /// ┌───────────┐  failed above threshold   ┌───────────────────────────────┐  │                          │
    /// │ DontTrace │ ◀──────────────────────── │            Tracing            │ ─┘                          │
    /// └───────────┘                           └───────────────────────────────┘                             │
    ///   ▲                                       │                                                           │
    ///   │                                       │                                                           │
    ///   │                                       ▼                                                           │
    ///   │           failed above threshold    ┌───────────────────────────────┐                             │
    ///   └──────────────────────────────────── │           Compiling           │ ────────────────────────────┘
    ///                                         └───────────────────────────────┘
    ///                                           │
    ///                                           │
    ///                                           ▼
    ///                                         ┌───────────────────────────────┐
    ///                                         │           Compiled            │ ◀┐
    ///                                         └───────────────────────────────┘  │
    ///                                           │                                │
    ///                                           │ guard failed above threshold   │ sidetracing completed
    ///                                           ▼                                │
    ///                                         ┌───────────────────────────────┐  │
    ///                                         │          SideTracing          │ ─┘
    ///                                         └───────────────────────────────┘
    /// ```
    ///
    /// This diagram was created with [this tool](https://dot-to-ascii.ggerganov.com/) using this
    /// GraphViz input:
    ///
    /// ```text
    /// digraph {
    ///   init [label="", shape=point];
    ///   init -> Counting;
    ///   Counting -> Counting [label="increment count"];
    ///   Counting -> Tracing [label="start tracing"];
    ///   Tracing -> Compiling;
    ///   Tracing -> Counting [label="failed below threshold"];
    ///   Tracing -> DontTrace [label="failed above threshold"];
    ///   Compiling -> Compiled;
    ///   Compiling -> Counting [label="failed below threshold"];
    ///   Compiling -> DontTrace [label="failed above threshold"];
    ///   Compiled -> SideTracing [label="guard failed above threshold"];
    ///   SideTracing -> Compiled [label="sidetracing completed"];
    /// }
    /// ```
    ///
    /// We hope that a Location soon reaches the `Compiled` state (aka "the happy state") and stays
    /// there. However, many Locations will not be used frequently enough to reach such a state, so
    /// we don't want to waste resources on them.
    ///
    /// We therefore encode a Location as a tagged integer: when initialised, no memory is
    /// allocated; if the location is used frequently enough it becomes hot, memory is allocated
    /// for it, and a pointer stored instead of an integer. Note that once memory for a hot
    /// location is allocated, it can only be (scheduled for) deallocation when a Location is
    /// dropped, as the Location may have handed out `&` references to that allocated memory. That
    /// means that the `Counting` state is encoded in two separate ways: both with and without
    /// allocated memory.
    ///
    /// The layout of a Location is as follows: bit 0 = <STATE_NOT_HOT|STATE_HOT>; bits 1..<machine
    /// width> = payload. In the `STATE_NOT_HOT` state, the payload is an integer; in a `STATE_HOT`
    /// state, the payload is a pointer from `Arc::into_raw::<Mutex<HotLocation>>()`.
    inner: AtomicUsize,
}

impl Location {
    /// Create a new location.
    pub fn new() -> Self {
        // Locations start in the counting state with a count of 0.
        debug_assert_ne!(STATE_NOT_HOT, 0);
        Self {
            inner: AtomicUsize::new(STATE_NOT_HOT),
        }
    }

    /// Create a new "null" location, denoting a point in a program which can never contribute to a
    /// trace.
    pub fn null() -> Self {
        Self {
            inner: AtomicUsize::new(STATE_NULL),
        }
    }

    /// Returns true if this is a "null" location.
    pub fn is_null(&self) -> bool {
        self.inner.load(Ordering::Relaxed) == STATE_NULL
    }

    /// If `self` is:
    ///   1. in the `Counting` state and
    ///   2. has not had a `HotLocation` allocated for it
    ///
    /// then increment and return its count, or `None` otherwise.
    pub(crate) fn inc_count(&self) -> Option<HotThreshold> {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_TAG_MASK == STATE_NOT_HOT {
            // `HotThreshold` must be unsigned
            debug_assert_eq!(HotThreshold::MIN, 0);
            // For the `as` to be safe, `HotThreshold` can't be bigger than `usize`
            debug_assert!(mem::size_of::<HotThreshold>() <= mem::size_of::<usize>());
            let old = (x >> STATE_NUM_BITS) as HotThreshold;
            // The particular value of `new` must fit in the bits we have available.
            let new = old + 1;
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
                .ok()
                .map(|_| new)
        } else {
            None
        }
    }

    /// If `self` is:
    ///   1. in the `Counting` state and
    ///   2. has not had a `HotLocation` allocated for it
    ///
    /// return its count, or `None` otherwise
    #[cfg(test)]
    pub(crate) fn count(&self) -> Option<HotThreshold> {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_TAG_MASK == STATE_NOT_HOT {
            // `HotThreshold` must be unsigned
            debug_assert_eq!(HotThreshold::MIN, 0);
            Some((x >> STATE_NUM_BITS) as HotThreshold)
        } else {
            None
        }
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
        debug_assert_eq!((cl as usize) & !STATE_TAG_MASK, cl as usize);
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
        if x & STATE_TAG_MASK == STATE_HOT {
            // `Arc::into_raw::<Mutex<T>>` returns `*mut Mutex<T>` so the address we're wrapping is
            // a pointer to the `Mutex` itself. By returning a `&` reference we ensure that the
            // reference to the `Mutex` can't outlive this `Location`.
            Some(unsafe { &*((x & !STATE_TAG_MASK) as *const _) })
        } else {
            None
        }
    }

    /// If `self` has a [HotLocation] return a clone of the [Arc] that wraps it, or `None`
    /// otherwise.
    pub(crate) fn hot_location_arc_clone(&self) -> Option<Arc<Mutex<HotLocation>>> {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_TAG_MASK == STATE_HOT {
            let raw = unsafe { Arc::from_raw((x & !STATE_TAG_MASK) as *mut _) };
            let cl = Arc::clone(&raw);
            mem::forget(raw);
            Some(cl)
        } else {
            None
        }
    }
}

impl Default for Location {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Location {
    fn drop(&mut self) {
        let x = self.inner.load(Ordering::Relaxed);
        if x & STATE_TAG_MASK == STATE_HOT {
            drop(unsafe { Arc::from_raw((x & !STATE_TAG_MASK) as *mut Mutex<HotLocation>) });
        }
    }
}

#[derive(Debug)]
pub(crate) struct HotLocation {
    pub(crate) kind: HotLocationKind,
    /// How often has tracing or compilation starting directly from this hot location led to an
    /// error?
    pub(crate) tracecompilation_errors: TraceCompilationErrorThreshold,
}

impl HotLocation {
    /// A trace, or the compilation of a trace, starting at this [HotLocation] led to an error. The
    /// return value indicates whether further traces for this location should be generated or not.
    pub(crate) fn tracecompilation_error(&mut self, mt: &Arc<MT>) -> TraceFailed {
        if self.tracecompilation_errors < mt.trace_failure_threshold() {
            self.tracecompilation_errors += 1;
            TraceFailed::KeepTrying
        } else {
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
    /// Because of a failure in compiling / tracing, we have reentered the `Counting` state. This
    /// can be seen as a way of implementing back-off in the face of errors.
    Counting(HotThreshold),
    /// This HotLocation has encountered problems (e.g. traces which are too long) and shouldn't be
    /// traced again.
    DontTrace,
    /// This HotLocation started a trace which is ongoing.
    Tracing,
    /// While executing JIT compiled code, a guard failed often enough for us to want to generate a
    /// side trace starting at this HotLocation.
    SideTracing {
        /// The root [CompiledTrace]: while one thread is side tracing a (possibly many levels
        /// deep) side trace that ultimately relates to this [CompiledTrace], other threads can
        /// execute this compiled trace.
        root_ctr: Arc<dyn CompiledTrace>,
        /// The ID of the guard that failed (inside `parent`).
        gidx: GuardIdx,
        /// The [CompiledTrace] that the guard failed in. This will either be `root_ctr` or a
        /// descendent of `root_ctr`.
        parent_ctr: Arc<dyn CompiledTrace>,
    },
}

/// When a [HotLocation] has failed to compile a valid trace, should the [HotLocation] be tried
/// again or not?
#[derive(Debug)]
pub(crate) enum TraceFailed {
    KeepTrying,
    DontTrace,
}
