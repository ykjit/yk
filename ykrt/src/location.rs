use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use crate::mt::ThreadIdInner;
use ykshim_client::CompiledTrace;

// The current meta-tracing phase of a given location in the end-user's code. Consists of a tag and
// (optionally) a value. We expect the most commonly encountered tag at run-time is PHASE_COMPILED
// whose value is a pointer to memory. By also making that tag 0b00, we allow that index to be
// accessed without any further operations after the initial tag check.
pub(crate) const PHASE_NUM_BITS: usize = 3;
pub(crate) const PHASE_TAG: usize = 0b111; // All of the other PHASE_ tags must fit in this.
pub(crate) const PHASE_COMPILED: usize = 0b000; // Value is a pointer to a chunk of memory containing a
                                                // CompiledTrace<I>.
pub(crate) const PHASE_TRACING: usize = 0b001; // Value is a pointer to an Arc<ThreadIdInner> representing a
                                               // thread ID.
pub(crate) const PHASE_TRACING_LOCK: usize = 0b010; // No associated value
pub(crate) const PHASE_COMPILING: usize = 0b011; // Value is a pointer to a `Box<CompilingTrace<I>>`.
pub(crate) const PHASE_COUNTING: usize = 0b100; // Value specifies how many times we've seen this Location.
pub(crate) const PHASE_LOCKED: usize = 0b101; // No associated value.
pub(crate) const PHASE_DONT_TRACE: usize = 0b110; // No associated value.

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct State(usize);

impl State {
    pub(crate) fn phase(&self) -> usize {
        self.0 & PHASE_TAG
    }

    pub(crate) fn phase_compiled<I>(trace: Box<CompiledTrace<I>>) -> Self {
        let ptr = Box::into_raw(trace);
        debug_assert_eq!(ptr as usize & PHASE_TAG, 0);
        State(ptr as usize | PHASE_COMPILED)
    }

    pub(crate) fn phase_tracing(tid: Arc<ThreadIdInner>) -> Self {
        let ptr = Arc::into_raw(tid);
        debug_assert_eq!(ptr as usize & PHASE_TAG, 0);
        State(ptr as usize | PHASE_TRACING)
    }

    pub(crate) fn phase_tracing_lock() -> Self {
        State(PHASE_TRACING_LOCK)
    }

    pub(crate) fn phase_compiling<I>(trace: Arc<CompilingTrace<I>>) -> Self {
        let ptr: *const CompilingTrace<I> = Arc::into_raw(trace);
        debug_assert_eq!(ptr as usize & PHASE_TAG, 0);
        State(ptr as usize | PHASE_COMPILING)
    }

    pub(crate) fn phase_counting(count: usize) -> Self {
        debug_assert_eq!(count << PHASE_NUM_BITS >> PHASE_NUM_BITS, count);
        State(PHASE_COUNTING | (count << PHASE_NUM_BITS))
    }

    pub(crate) fn phase_locked() -> Self {
        State(PHASE_LOCKED)
    }

    pub(crate) fn phase_dont_trace() -> Self {
        State(PHASE_DONT_TRACE)
    }

    pub(crate) fn number_data(&self) -> usize {
        debug_assert_eq!(self.phase(), PHASE_COUNTING);
        self.0 >> PHASE_NUM_BITS
    }

    pub(crate) fn pointer_data<T>(&self) -> *const T {
        (self.0 & !PHASE_TAG) as *const T
    }

    pub(crate) unsafe fn ref_data<T>(&self) -> &T {
        &*self.pointer_data()
    }
}

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
#[derive(Debug)]
pub struct Location<I> {
    // A Location is a state machine which operates as follows (where PHASE_COUNTING is the start
    // state):
    //
    //              ┌────────────────────────────────────────┐
    //              │                                        │─────────────┐
    //   reprofile  │             PHASE_COUNTING             │             │
    //  ┌──────────▶│                                        │◀────────────┘
    //  │           └────────────────────────────────────────┘    increment
    //  │             │                                             count
    //  │             │ start tracing
    //  │             ▼
    //  │           ┌────────────────────┐
    //  │           │   PHASE_TRACING    │
    //  │           └────────────────────┘
    //  │             │ check for    ▲
    //  │             │ stuckness    | still
    //  │             ▼              | active
    //  │           ┌────────────────────┐
    //  │           │                    │  abort   ┌────────────────────┐
    //  │           │ PHASE_TRACING_LOCK │─────────▶│  PHASE_DONT_TRACE  │
    //  │           │                    │          └────────────────────┘
    //  │           └────────────────────┘
    //  │             │ start compiling trace
    //  │             │ in thread
    //  │             ▼
    //  │           ┌────────────────────┐
    //  │           │                    │
    //  |           |  PHASE_COMPILING   |
    //  │           │                    │
    //  │           └────────────────────┘
    //  │             │ trace maybe    ▲
    //  │             │ compiled       | trace not yet
    //  │             ▼                | compiled
    //  │           ┌────────────────────┐
    //  │           │    PHASE_LOCKED    │
    //  │           └────────────────────┘
    //  │             │ trace definitely
    //  │             │ compiled
    //  │             ▼
    //  │           ┌────────────────────┐
    //  │           │                    │──────┐
    //  │           │   PHASE_COMPILED   │      │
    //  └───────────│                    │◀─────┘
    //              └────────────────────┘
    //
    // We hope that a Location soon reaches PHASE_COMPILED (aka "the happy state") and stays there.
    state: AtomicUsize,
    phantom: PhantomData<I>,
}

impl<I> Location<I> {
    pub fn new() -> Self {
        Self {
            state: AtomicUsize::new(PHASE_COUNTING),
            phantom: PhantomData,
        }
    }

    pub(crate) fn load(&self, order: Ordering) -> State {
        State(self.state.load(order))
    }

    pub(crate) fn compare_and_swap(&self, current: State, new: State, order: Ordering) -> State {
        State(
            self.state
                .compare_exchange(current.0, new.0, order, Ordering::Relaxed)
                .unwrap_or_else(|e| e),
        )
    }

    pub(crate) fn store(&self, state: State, order: Ordering) {
        self.state.store(state.0, order);
    }
}

impl<I> Drop for Location<I> {
    fn drop(&mut self) {
        let lp = *self.state.get_mut();
        if lp & PHASE_TAG == PHASE_TRACING {
            unsafe { Arc::from_raw((lp & !PHASE_TAG) as *mut u8) };
        }
    }
}

pub(crate) type CompilingTrace<I> = Mutex<Option<Box<CompiledTrace<I>>>>;
