#[cfg(test)]
use std::time::Duration;
use std::{
    alloc::{alloc, dealloc, Layout},
    io,
    marker::PhantomData,
    mem,
    panic::{catch_unwind, resume_unwind, UnwindSafe},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{channel, Receiver, TryRecvError},
        Arc,
    },
    thread::{self, JoinHandle},
};
use ykcompile::{CompiledTrace, TraceCompiler};
use yktrace::{sir::SIR, start_tracing, tir::TirTrace, ThreadTracer, TracingKind};

pub type HotThreshold = usize;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;
const PHASE_NUM_BITS: usize = 2;

// The current meta-tracing phase of a given location in the end-user's code. Consists of a tag and
// (optionally) a value. We expect the most commonly encountered tag at run-time is PHASE_COMPILED
// whose value is a pointer to memory. By also making that tag 0b00, we allow that index to be
// accessed without any further operations after the initial tag check.
const PHASE_TAG: usize = 0b11; // All of the other PHASE_ tags must fit in this.
const PHASE_COMPILED: usize = 0b00; // Value is a pointer to a chunk of memory containing a
                                    // CompiledTrace<I>.
const PHASE_TRACING: usize = 0b01; // Value is a pointer to a malloc'd block that allows us to
                                   // precisely identify when we have reached the same Location.
const PHASE_COMPILING: usize = 0b10; // Value is a pointer to a `Box<CompilingTrace<I>>`.
const PHASE_COUNTING: usize = 0b11; // Value specifies how many times we've seen this Location.

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
    //               ┌────────────────────────────────────────┐
    //               │                                        │ ─────────────┐
    //   reprofile   │             PHASE_COUNTING             │              │
    //  ┌──────────▶ │                                        │ ◀────────────┘
    //  │            └────────────────────────────────────────┘    increment
    //  │              │                     ▲         ▲           count
    //  │              │ start tracing       │ abort   │ abort
    //  │              ▼                     │         │
    //  │            ┌────────────────────┐  │         │
    //  │            │   PHASE_TRACING    │ ─┘         │
    //  │            └────────────────────┘            │
    //  │              │                               │
    //  │              │ start compiling trace         │
    //  │              │ in thread                     │
    //  │              ▼                               │
    //  │            ┌────────────────────┐            │
    //  |            |                    | ───────────┘
    //  │            │                    │
    //  │            │  PHASE_COMPILING   │ ──────┐
    //  │            │                    │       │ still compiling in thread
    //  │            │                    │ ◀─────┘
    //  │            └────────────────────┘
    //  │              │
    //  │              │ trace compiled
    //  │              ▼
    //  │            ┌────────────────────┐
    //  │            │                    │ ──────┐
    //  │            │   PHASE_COMPILED   │       │
    //  └─────────── │                    │ ◀─────┘
    //               └────────────────────┘
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
}

impl<I> Drop for Location<I> {
    fn drop(&mut self) {
        let lp = self.state.load(Ordering::Relaxed);
        if lp & PHASE_TAG == PHASE_COMPILING {
            unsafe {
                Box::from_raw((lp & !PHASE_TAG) as *mut CompilingTrace<I>);
            }
        }
    }
}

/// Configure a meta-tracer. Note that a process can only have one meta-tracer active at one point.
pub struct MTBuilder {
    hot_threshold: HotThreshold,
    /// The kind of tracer to use.
    tracing_kind: TracingKind,
}

impl MTBuilder {
    /// Create a meta-tracer with default parameters.
    pub fn new() -> Self {
        Self {
            hot_threshold: DEFAULT_HOT_THRESHOLD,
            tracing_kind: TracingKind::default(),
        }
    }

    /// Consume the `MTBuilder` and create a meta-tracer, returning the
    /// [`MTThread`](struct.MTThread.html) representing the current thread.
    pub fn init(self) -> MTThread {
        MTInner::init(self.hot_threshold, self.tracing_kind)
    }

    /// Change this meta-tracer builder's `hot_threshold` value.
    pub fn hot_threshold(mut self, hot_threshold: HotThreshold) -> Self {
        self.hot_threshold = hot_threshold;
        self
    }

    /// Select the kind of tracing to use.
    pub fn tracing_kind(mut self, tracing_kind: TracingKind) -> Self {
        self.tracing_kind = tracing_kind;
        self
    }
}

#[derive(Clone)]
/// A meta-tracer. Note that this is conceptually a "front-end" to the actual meta-tracer akin to
/// an `Rc`: this struct can be freely `clone()`d without duplicating the underlying meta-tracer.
pub struct MT {
    inner: Arc<MTInner>,
}

impl MT {
    /// Return this meta-tracer's hot threshold.
    pub fn hot_threshold(&self) -> HotThreshold {
        self.inner.hot_threshold.load(Ordering::Relaxed)
    }

    /// Return the kind of tracing that this meta-tracer is using.
    pub fn tracing_kind(&self) -> TracingKind {
        self.inner.tracing_kind
    }

    /// Create a new thread that can be used in the meta-tracer: the new thread that is created is
    /// handed a [`MTThread`](struct.MTThread.html) from which the `MT` itself can be accessed.
    pub fn spawn<F, T>(&self, f: F) -> io::Result<JoinHandle<T>>
    where
        F: FnOnce(MTThread) -> T,
        F: Send + UnwindSafe + 'static,
        T: Send + 'static,
    {
        let mt_cl = self.clone();
        thread::Builder::new().spawn(move || {
            mt_cl.inner.active_threads.fetch_add(1, Ordering::Relaxed);
            let r = catch_unwind(|| f(MTThreadInner::init(mt_cl.clone())));
            mt_cl.inner.active_threads.fetch_sub(1, Ordering::Relaxed);
            match r {
                Ok(r) => r,
                Err(e) => resume_unwind(e),
            }
        })
    }
}

impl Drop for MT {
    fn drop(&mut self) {
        MT_ACTIVE.store(false, Ordering::Relaxed);
    }
}

/// The innards of a meta-tracer.
struct MTInner {
    hot_threshold: AtomicUsize,
    active_threads: AtomicUsize,
    tracing_kind: TracingKind,
}

/// It's only safe to have one `MT` instance active at a time.
static MT_ACTIVE: AtomicBool = AtomicBool::new(false);

impl MTInner {
    /// Create a new `MT`, wrapped immediately in an [`MTThread`](struct.MTThread.html).
    fn init(hot_threshold: HotThreshold, tracing_kind: TracingKind) -> MTThread {
        // A process can only have a single MT instance.

        // In non-testing, we panic if the user calls this method while an MT instance is active.
        #[cfg(not(test))]
        {
            if MT_ACTIVE.swap(true, Ordering::Relaxed) {
                panic!("Only one MT can be active at once.");
            }
        }

        // In testing, we simply sleep until the other MT instance has gone away: this has the
        // effect of serialising tests which use MT (but allowing other tests to run in parallel).
        #[cfg(test)]
        {
            loop {
                if !MT_ACTIVE.swap(true, Ordering::Relaxed) {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }

        let mtc = Self {
            hot_threshold: AtomicUsize::new(hot_threshold),
            active_threads: AtomicUsize::new(1),
            tracing_kind,
        };
        let mt = MT {
            inner: Arc::new(mtc),
        };
        MTThreadInner::init(mt)
    }
}

/// The communication mechanism between a compiling thread and an interpreter thread.
struct CompilingTrace<I> {
    rcv: Receiver<CompiledTrace<I>>,
}

/// A meta-tracer aware thread. Note that this is conceptually a "front-end" to the actual
/// meta-tracer thread akin to an `Rc`: this struct can be freely `clone()`d without duplicating
/// the underlying meta-tracer thread. Note that this struct is neither `Send` nor `Sync`: it
/// can only be accessed from within a single thread.
#[derive(Clone)]
pub struct MTThread {
    inner: Rc<MTThreadInner>,
}

impl MTThread {
    /// Return a meta-tracer [`MT`](struct.MT.html) struct.
    pub fn mt(&self) -> &MT {
        &self.inner.mt
    }

    /// Attempt to execute a compiled trace for location `loc`.
    pub fn control_point<S, I: Send + 'static>(
        &mut self,
        loc: Option<&Location<I>>,
        step_fn: S,
        inputs: &mut I,
    ) where
        S: Fn(&mut I),
    {
        // If a loop can start at this position then update the location and potentially start/stop
        // this thread's tracer.
        if let Some(loc) = loc {
            if let Some(func) = self.transition_location::<I>(loc) {
                if self.exec_trace(func, inputs) {
                    // Trace succesfully executed.
                    return;
                } else {
                    // FIXME blackholing
                    todo!("Guard failure!")
                }
            }
        }
        step_fn(inputs)
    }

    fn exec_trace<I>(&mut self, func: fn(&mut I) -> bool, inputs: &mut I) -> bool {
        func(inputs)
    }

    /// `Location`s represent a statemachine: this function transitions to the next state (which
    /// may be the same as the previous state!). If this results in a compiled trace, it returns
    /// `Some(pointer_to_trace_function)`.
    fn transition_location<I: Send + 'static>(
        &mut self,
        loc: &Location<I>,
    ) -> Option<fn(&mut I) -> bool> {
        // Since we don't hold an explicit lock, updating a Location is tricky: we might read a
        // Location, work out what we'd like to update it to, and try updating it, only to find
        // that another thread interrupted us part way through. We therefore use compare_and_swap
        // to update values, allowing us to detect if we were interrupted. If we were interrupted,
        // we simply retry the whole operation.
        loop {
            // We need Acquire ordering, as PHASE_COMPILED will need to read information written to
            // external data as a result of the PHASE_TRACING -> PHASE_COMPILED transition.
            let lp = loc.state.load(Ordering::Acquire);
            match lp & PHASE_TAG {
                PHASE_COUNTING => {
                    let count = (lp & !PHASE_TAG) >> PHASE_NUM_BITS;
                    if count >= self.inner.hot_threshold {
                        if self.inner.tracer.is_some() {
                            // This thread is already tracing another Location, so either another
                            // thread needs to trace this Location or this thread needs to wait
                            // until the current round of tracing has completed. Either way,
                            // there's no point incrementing the hot count.
                            return None;
                        }
                        let new_state = self.inner.tid as usize | PHASE_TRACING;
                        if loc.state.compare_and_swap(lp, new_state, Ordering::Release) == lp {
                            Rc::get_mut(&mut self.inner).unwrap().tracer =
                                Some((start_tracing(self.inner.tracing_kind), self.inner.tid));
                            return None;
                        }
                        // We raced with another thread, which probably moved the Location from
                        // PHASE_COUNTING to PHASE_TRACING. There's a small chance that the other
                        // thread managed to both trace and compile the Location during the race,
                        // but that's unlikely enough that it's not worth paying the performance
                        // penalty of going around the loop again.
                        return None;
                    } else {
                        let new_state = PHASE_COUNTING | ((count + 1) << PHASE_NUM_BITS);
                        if loc.state.compare_and_swap(lp, new_state, Ordering::Release) == lp {
                            return None;
                        }
                        // We raced with another thread while incrementing the count. It's worth us
                        // trying to go around the loop again, as unless the count exceeded the hot
                        // threshold, we'll still be able to increment it, ensuring that we don't
                        // get weird non-determinism in when a Location is compiled.
                    }
                }
                PHASE_TRACING => {
                    if let Some((_, tid)) = self.inner.tracer {
                        // This thread is tracing something...
                        if tid != ((lp & !PHASE_TAG) as *mut u8) {
                            // ...but we didn't start at the current Location.
                            return None;
                        } else {
                            // ...and we started at this Location, so we've got a complete loop!
                        }
                    } else {
                        // Another thread is tracing this location.
                        return None;
                    }

                    // Stop the tracer
                    let sir_trace = Rc::get_mut(&mut self.inner)
                        .unwrap()
                        .tracer
                        .take()
                        .unwrap()
                        .0
                        .stop_tracing()
                        .unwrap();
                    dbg!(sir_trace.raw_len());

                    // Start a compilation thread.
                    let (snd, rcv) = channel();
                    thread::spawn(move || {
                        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
                        snd.send(TraceCompiler::<I>::compile(tir_trace)).ok();
                    });
                    let ptr = Box::into_raw(Box::new(CompilingTrace { rcv }));
                    debug_assert_eq!(ptr as usize & PHASE_TAG, 0);
                    let new_state = ptr as usize | PHASE_COMPILING;
                    loc.state.store(new_state, Ordering::Release);

                    Rc::get_mut(&mut self.inner).unwrap().tracer = None;
                    return None;
                }
                PHASE_COMPILING => {
                    let compiling =
                        unsafe { Box::from_raw((lp & !PHASE_TAG) as *mut CompilingTrace<I>) };
                    match compiling.rcv.try_recv() {
                        Ok(ct) => {
                            // FIXME: free up the memory when the trace is no longer used.
                            let ptr: *mut CompiledTrace<I> = Box::into_raw(Box::new(ct));
                            let new_state = ptr as usize | PHASE_COMPILED;
                            loc.state.store(new_state, Ordering::Release);

                            let bct = unsafe { Box::from_raw(ptr) };
                            let f = unsafe { mem::transmute::<_, fn(&mut I) -> bool>(bct.ptr()) };
                            mem::forget(bct);
                            return Some(f);
                        }
                        Err(TryRecvError::Empty) => {
                            // The compiling thread is still operating.
                            mem::forget(compiling);
                            return None;
                        }
                        Err(TryRecvError::Disconnected) => {
                            // The compiling thread has gone wrong in some way.
                            todo!();
                        }
                    }
                }
                PHASE_COMPILED => {
                    let bct = unsafe { Box::from_raw((lp & !PHASE_TAG) as *mut CompiledTrace<I>) };
                    let f = unsafe { mem::transmute::<_, fn(&mut I) -> bool>(bct.ptr()) };
                    mem::forget(bct);
                    return Some(f);
                }
                _ => unreachable!(),
            }
        }
    }
}

/// The innards of a meta-tracer thread.
struct MTThreadInner {
    mt: MT,
    /// A value that uniquely identifies a thread. Since this ID needs to be ORable with PHASE_TAG,
    /// we use a pointer to a malloc'd chunk of memory. We guarantee a) that chunk is aligned to a
    /// machine word b) that it is a non-zero chunk of memory (and thus guaranteed to be a unique
    /// pointer).
    tid: *mut u8,
    hot_threshold: HotThreshold,
    tracing_kind: TracingKind,
    /// The active tracer and the unique identifier of the [Location] that is being traced. We use
    /// a pointer to a 1 byte malloc'd chunk of memory since the resulting address is guaranteed to
    /// be unique.
    tracer: Option<(ThreadTracer, *mut u8)>,
}

impl MTThreadInner {
    fn init(mt: MT) -> MTThread {
        let hot_threshold = mt.hot_threshold();
        let tracing_kind = mt.tracing_kind();
        let tid = {
            let layout =
                Layout::from_size_align(mem::size_of::<usize>(), mem::size_of::<usize>()).unwrap();
            unsafe { alloc(layout) }
        };
        let inner = MTThreadInner {
            mt,
            tid,
            hot_threshold,
            tracing_kind,
            tracer: None,
        };
        MTThread {
            inner: Rc::new(inner),
        }
    }
}

impl Drop for MTThreadInner {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(mem::size_of::<usize>(), mem::size_of::<usize>()).unwrap();
        unsafe { dealloc(self.tid, layout) };
    }
}

#[cfg(test)]
mod tests {
    use std::thread::yield_now;
    extern crate test;
    use self::test::{black_box, Bencher};
    use super::*;

    struct DummyIO {}

    #[interp_step]
    fn dummy_step(_inputs: &mut DummyIO) {}

    #[test]
    fn threshold_passed() {
        let hot_thrsh = 1500;
        let mut mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let lp = Location::new();
        let mut io = DummyIO {};
        for i in 0..hot_thrsh {
            mtt.control_point(Some(&lp), dummy_step, &mut io);
            assert_eq!(
                lp.state.load(Ordering::Relaxed),
                PHASE_COUNTING | ((i + 1) << PHASE_NUM_BITS)
            );
        }
        assert_eq!(lp.state.load(Ordering::Relaxed) & PHASE_TAG, PHASE_COUNTING);
        mtt.control_point(Some(&lp), dummy_step, &mut io);
        assert_eq!(lp.state.load(Ordering::Relaxed) & PHASE_TAG, PHASE_TRACING);
        mtt.control_point(Some(&lp), dummy_step, &mut io);
        let fs = lp.state.load(Ordering::Relaxed) & PHASE_TAG;
        assert!(fs == PHASE_COMPILING || fs == PHASE_COMPILED);
    }

    #[test]
    fn stop_while_tracing() {
        let hot_thrsh = 5;
        let mut mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let lp = Location::new();
        let mut io = DummyIO {};
        for i in 0..hot_thrsh {
            mtt.control_point(Some(&lp), dummy_step, &mut io);
            assert_eq!(
                lp.state.load(Ordering::Relaxed),
                PHASE_COUNTING | ((i + 1) << PHASE_NUM_BITS)
            );
        }
        assert_eq!(lp.state.load(Ordering::Relaxed) & PHASE_TAG, PHASE_COUNTING);
        mtt.control_point(Some(&lp), dummy_step, &mut io);
        assert_eq!(lp.state.load(Ordering::Relaxed) & PHASE_TAG, PHASE_TRACING);
    }

    #[test]
    fn threaded_threshold_passed() {
        let hot_thrsh = 4000;
        let mut mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let loc = Arc::new(Location::new());
        let mut thrs = vec![];
        for _ in 0..hot_thrsh / 4 {
            let loc = Arc::clone(&loc);
            let t = mtt
                .mt()
                .spawn(move |mut mtt| {
                    let mut io = DummyIO {};
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c1 = loc.state.load(Ordering::Relaxed);
                    assert_eq!(c1 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c2 = loc.state.load(Ordering::Relaxed);
                    assert_eq!(c2 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c3 = loc.state.load(Ordering::Relaxed);
                    assert_eq!(c3 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c4 = loc.state.load(Ordering::Relaxed);
                    assert_eq!(c4 & PHASE_TAG, PHASE_COUNTING);
                    assert!(c4 > c3);
                    assert!(c3 > c2);
                    assert!(c2 > c1);
                })
                .unwrap();
            thrs.push(t);
        }
        for t in thrs {
            t.join().unwrap();
        }
        {
            let mut io = DummyIO {};
            mtt.control_point(Some(&loc), dummy_step, &mut io);
            assert_eq!(loc.state.load(Ordering::Relaxed) & PHASE_TAG, PHASE_TRACING);
            mtt.control_point(Some(&loc), dummy_step, &mut io);
            mtt.control_point(Some(&loc), dummy_step, &mut io);

            while loc.state.load(Ordering::Relaxed) & PHASE_TAG == PHASE_COMPILING {
                yield_now();
                mtt.control_point(Some(&loc), dummy_step, &mut io);
            }
            assert_eq!(
                loc.state.load(Ordering::Relaxed) & PHASE_TAG,
                PHASE_COMPILED
            );
        }
    }

    #[test]
    fn dumb_interpreter() {
        let mut mtt = MTBuilder::new().hot_threshold(2).init();

        // The program is silly. Do nothing twice, then start again.
        const NOP: u8 = 0;
        const RESTART: u8 = 1;
        let prog = vec![NOP, NOP, RESTART];

        // Suppose the bytecode compiler for this imaginary language knows that the first bytecode
        // is the only place a loop can start.
        let locs = vec![Some(Location::new()), None, None];

        struct IO {
            prog: Vec<u8>,
            pc: usize,
            count: usize,
        }

        #[interp_step]
        fn dumb_interp_step(tio: &mut IO) {
            match tio.prog[tio.pc] {
                NOP => {
                    tio.pc += 1;
                    tio.count += 1;
                }
                RESTART => tio.pc = 0,
                _ => unreachable!(),
            }
        }

        let mut tio = IO {
            prog,
            pc: 0,
            count: 0,
        };

        // The interpreter loop. In reality this would (syntactically) be an infinite loop.
        for _ in 0..12 {
            let loc = locs[tio.pc].as_ref();
            mtt.control_point(loc, dumb_interp_step, &mut tio);
        }

        loop {
            yield_now();
            let loc = locs[tio.pc].as_ref();
            if tio.pc == 0
                && loc.unwrap().state.load(Ordering::Relaxed) & PHASE_TAG != PHASE_COMPILING
            {
                break;
            }
            mtt.control_point(loc, dumb_interp_step, &mut tio);
        }

        assert_eq!(
            locs[0].as_ref().unwrap().state.load(Ordering::Relaxed) & PHASE_TAG,
            PHASE_COMPILED
        );

        assert_eq!(tio.pc, 0);

        // A trace was just compiled. Running it should execute NOP twice.
        tio.count = 8;
        for i in 0..10 {
            let loc = locs[tio.pc].as_ref();
            mtt.control_point(loc, dumb_interp_step, &mut tio);
            assert_eq!(tio.count, 10 + i * 2);
        }
    }

    #[bench]
    fn bench_single_threaded_control_point(b: &mut Bencher) {
        let mut mtt = MTBuilder::new().init();
        let lp = Location::new();
        let mut io = DummyIO {};
        b.iter(|| {
            for _ in 0..100000 {
                black_box(mtt.control_point(Some(&lp), dummy_step, &mut io));
            }
        });
    }

    #[bench]
    fn bench_multi_threaded_control_point(b: &mut Bencher) {
        let mtt = MTBuilder::new().init();
        let loc = Arc::new(Location::new());
        b.iter(|| {
            let mut thrs = vec![];
            for _ in 0..4 {
                let loc = Arc::clone(&loc);
                let t = mtt
                    .mt()
                    .spawn(move |mut mtt| {
                        for _ in 0..100 {
                            let mut io = DummyIO {};
                            black_box(mtt.control_point(Some(&loc), dummy_step, &mut io));
                        }
                    })
                    .unwrap();
                thrs.push(t);
            }
            for t in thrs {
                t.join().unwrap();
            }
        });
    }
}
