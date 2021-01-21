#[cfg(test)]
use std::time::Duration;
use std::{
    io,
    marker::PhantomData,
    mem,
    panic::{catch_unwind, resume_unwind, UnwindSafe},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex, TryLockError,
    },
    thread::{self, yield_now, JoinHandle},
};

use crate::location::{
    CompilingTrace, Location, State, PHASE_COMPILED, PHASE_COMPILING, PHASE_COUNTING,
    PHASE_DONT_TRACE, PHASE_LOCKED, PHASE_TRACING, PHASE_TRACING_LOCK,
};
use ykshim_client::{compile_trace, start_tracing, CompiledTrace, ThreadTracer, TracingKind};

pub type HotThreshold = usize;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;

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
            #[cfg(tracermode = "hw")]
            tracing_kind: TracingKind::HardwareTracing,
            #[cfg(tracermode = "sw")]
            tracing_kind: TracingKind::SoftwareTracing,
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

pub(crate) struct ThreadIdInner;

/// A meta-tracer aware thread. Note that this is conceptually a "front-end" to the actual
/// meta-tracer thread akin to an `Rc`: this struct can be freely `clone()`d without duplicating
/// the underlying meta-tracer thread. Note that this struct is neither `Send` nor `Sync`: it
/// can only be accessed from within a single thread.
#[derive(Clone)]
pub struct MTThread {
    inner: Rc<MTThreadInner>,
    // Raw pointers are neither send nor sync.
    _dont_send_or_sync_me: PhantomData<*mut ()>,
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

        // We need Acquire ordering, as PHASE_COMPILING and PHASE_COMPILED need to read information
        // written to external data. Alternatively, this load could be Relaxed but we would then
        // need to place an Acquire fence in PHASE_COMPILING and PHASE_COMPILED.
        let mut lp = loc.load(Ordering::Acquire);

        match lp.phase() {
            PHASE_COUNTING => {
                loop {
                    let count = lp.number_data();
                    if count >= self.inner.hot_threshold {
                        if self.inner.tracer.is_some() {
                            // This thread is already tracing another Location, so either another
                            // thread needs to trace this Location or this thread needs to wait
                            // until the current round of tracing has completed. Either way,
                            // there's no point incrementing the hot count.
                            return None;
                        }
                        let new_state = State::phase_tracing(Arc::clone(&self.inner.tid));
                        if loc.compare_and_swap(lp, new_state, Ordering::Relaxed) == lp {
                            Rc::get_mut(&mut self.inner).unwrap().tracer = Some((
                                start_tracing(self.inner.tracing_kind),
                                Arc::clone(&self.inner.tid),
                            ));
                            return None;
                        }
                        // We raced with another thread, which probably moved the Location from
                        // PHASE_COUNTING to PHASE_TRACING.
                        unsafe {
                            Arc::from_raw(new_state.pointer_data::<u8>() as *mut u8);
                        }
                        return None;
                    } else {
                        let new_state = State::phase_counting(count + 1);
                        let old_lp = lp;
                        lp = loc.compare_and_swap(lp, new_state, Ordering::Relaxed);
                        if lp == old_lp {
                            return None;
                        }
                        // We raced with another thread.
                        if lp.phase() != PHASE_COUNTING {
                            // The other thread probably moved the Location from PHASE_COUNTING to
                            // PHASE_TRACING.
                            return None;
                        }
                        // This Location is still being counted, so go around the loop again and
                        // try to apply our increment.
                    }
                }
            }
            PHASE_TRACING | PHASE_TRACING_LOCK => {
                if lp.phase() == PHASE_TRACING_LOCK {
                    // PHASE_TRACING_LOCK requires special handling, because another thread might
                    // have grabbed it on this Location even though this thread is tracing this
                    // location.

                    if self.inner.tracer.is_none() {
                        // If this thread isn't tracing anything, there's little point in spinning
                        // as the thread tracing this location is still plausibly alive.
                        return None;
                    }

                    // We wait to grab the lock.
                    // FIXME: this is a terrible spinlock.
                    while lp.phase() == PHASE_TRACING_LOCK {
                        yield_now();
                        lp = loc.load(Ordering::Acquire);
                    }

                    if lp.phase() != PHASE_TRACING {
                        // While waiting for the Location to move beyond PHASE_TRACING_LOCK, we
                        // probably raced with the thread tracing this location.
                        return None;
                    }
                }

                // Before we can do anything with this phase's value, we need to transition to
                // `PHASE_TRACING_LOCK`.
                loop {
                    let old_lp = lp;
                    lp = loc.compare_and_swap(lp, State::phase_tracing_lock(), Ordering::Acquire);
                    if lp == old_lp {
                        break;
                    }
                    if self.inner.tracer.is_none() {
                        // This thread isn't tracing anything, so it's not worth us spinning and
                        // trying to lock this location.
                        return None;
                    }
                    // We are tracing something, and since it might be the current Location, we
                    // need to transition to TRACING_LOCK to avoid tracing more than one iteration
                    // of the loop. However, we might be able to determine that this thread
                    // couldn't have been tracing this location, at which point we can bail out.
                    // FIXME: this is a terrible spinlock.
                    loop {
                        match lp.phase() {
                            PHASE_TRACING => break,
                            PHASE_TRACING_LOCK => (),
                            PHASE_COMPILED | PHASE_COMPILING | PHASE_LOCKED | PHASE_COUNTING
                            | PHASE_DONT_TRACE => {
                                // If we observe any of these states, then another thread took
                                // responsibility for this Location, proving that this thread can't
                                // be responsible for it.
                                return None;
                            }
                            _ => unreachable!(),
                        }
                        yield_now();
                        lp = loc.load(Ordering::Relaxed);
                    }
                }

                let loc_tid = unsafe { Arc::from_raw(lp.pointer_data::<ThreadIdInner>()) };
                if let Some((_, ref tid)) = self.inner.tracer {
                    // This thread is tracing something...
                    if !Arc::ptr_eq(tid, &loc_tid) {
                        // ...but we didn't start at the current Location.
                        loc.store(lp, Ordering::Relaxed);
                        mem::forget(loc_tid);
                        return None;
                    } else {
                        // ...and we started at this Location, so we've got a complete loop!
                    }
                } else {
                    // Another thread is tracing this location.
                    if Arc::strong_count(&loc_tid) == 1 {
                        // The other thread has stopped. Mark this Location as DONT_TRACE.
                        loc.store(State::phase_dont_trace(), Ordering::Relaxed);
                        return None;
                    } else {
                        // The other thread is still alive.
                        loc.store(lp, Ordering::Relaxed);
                        mem::forget(loc_tid);
                        return None;
                    }
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

                // Start a compilation thread.
                let mtx = Arc::new(Mutex::new(None));
                let mtx_cl = Arc::clone(&mtx);
                thread::spawn(move || {
                    let compiled = compile_trace::<I>(sir_trace).unwrap();
                    *mtx_cl.lock().unwrap() = Some(Box::new(compiled));
                    // FIXME: although we've now put the compiled trace into the mutex, there's no
                    // guarantee that the Location for which we're compiling will ever be executed
                    // again. In such a case, the memory has, in essence, leaked.
                });
                let new_state = State::phase_compiling(mtx);
                loc.store(new_state, Ordering::Release);

                return None;
            }
            PHASE_COMPILING => {
                // We need to free the memory allocated earlier. To ensure we don't race with
                // another thread and both try and free the memory, we need to transition to
                // PHASE_LOCKED so that only this thread will try to free the memory.
                if loc.compare_and_swap(lp, State::phase_locked(), Ordering::Acquire) != lp {
                    // We raced with another thread that's probably transitioned this Location to
                    // PHASE_LOCK.
                    return None;
                }
                let mtx = unsafe { lp.ref_data::<CompilingTrace<I>>() };
                match mtx.try_lock() {
                    Ok(mut gd) => {
                        if let Some(tr) = (*gd).take() {
                            let f = unsafe { mem::transmute::<_, fn(&mut I) -> bool>(tr.ptr()) };
                            loc.store(State::phase_compiled(tr), Ordering::Release);
                            return Some(f);
                        }
                    }
                    Err(TryLockError::WouldBlock) => {
                        // The compiling thread is still operating.
                    }
                    Err(TryLockError::Poisoned(_)) => {
                        // The compiling thread has gone wrong in some way.
                        todo!();
                    }
                }
                loc.store(lp, Ordering::Relaxed);
                return None;
            }
            PHASE_LOCKED | PHASE_DONT_TRACE => return None,
            PHASE_COMPILED => {
                let bct = unsafe { lp.ref_data::<CompiledTrace<I>>() };
                let f = unsafe { mem::transmute::<_, fn(&mut I) -> bool>(bct.ptr()) };
                return Some(f);
            }
            _ => unreachable!(),
        }
    }
}

/// The innards of a meta-tracer thread.
struct MTThreadInner {
    mt: MT,
    /// An Arc whose pointer address uniquely identifies this thread. When a Location is traced,
    /// this Arc's strong count will be incremented. If, after this thread drops, this Arc's strong
    /// count remains > 0, it means that it was in the process of tracing a loop, implying that
    /// there is (or, at least, was at some point) a Location stuck in PHASE_TRACING.
    tid: Arc<ThreadIdInner>,
    hot_threshold: HotThreshold,
    tracing_kind: TracingKind,
    /// The active tracer and the unique identifier of the [Location] that is being traced. We use
    /// a pointer to a 1 byte malloc'd chunk of memory since the resulting address is guaranteed to
    /// be unique.
    tracer: Option<(ThreadTracer, Arc<ThreadIdInner>)>,
}

impl MTThreadInner {
    fn init(mt: MT) -> MTThread {
        let hot_threshold = mt.hot_threshold();
        let tracing_kind = mt.tracing_kind();
        let inner = MTThreadInner {
            mt,
            tid: Arc::new(ThreadIdInner),
            hot_threshold,
            tracing_kind,
            tracer: None,
        };
        MTThread {
            inner: Rc::new(inner),
            _dont_send_or_sync_me: PhantomData,
        }
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
            assert_eq!(lp.load(Ordering::Relaxed), State::phase_counting(i + 1));
        }
        assert_eq!(lp.load(Ordering::Relaxed).phase(), PHASE_COUNTING);
        mtt.control_point(Some(&lp), dummy_step, &mut io);
        assert_eq!(lp.load(Ordering::Relaxed).phase(), PHASE_TRACING);
        mtt.control_point(Some(&lp), dummy_step, &mut io);
        let fs = lp.load(Ordering::Relaxed).phase();
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
            assert_eq!(lp.load(Ordering::Relaxed), State::phase_counting(i + 1));
        }
        assert_eq!(lp.load(Ordering::Relaxed).phase(), PHASE_COUNTING);
        mtt.control_point(Some(&lp), dummy_step, &mut io);
        assert_eq!(lp.load(Ordering::Relaxed).phase(), PHASE_TRACING);
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
                    let c1 = loc.load(Ordering::Relaxed);
                    assert_eq!(c1.phase(), PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c2 = loc.load(Ordering::Relaxed);
                    assert_eq!(c2.phase(), PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c3 = loc.load(Ordering::Relaxed);
                    assert_eq!(c3.phase(), PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, &mut io);
                    let c4 = loc.load(Ordering::Relaxed);
                    assert_eq!(c4.phase(), PHASE_COUNTING);
                    assert!(c4.number_data() > c3.number_data());
                    assert!(c3.number_data() > c2.number_data());
                    assert!(c2.number_data() > c1.number_data());
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
            assert_eq!(loc.load(Ordering::Relaxed).phase(), PHASE_TRACING);
            mtt.control_point(Some(&loc), dummy_step, &mut io);
            mtt.control_point(Some(&loc), dummy_step, &mut io);

            while loc.load(Ordering::Relaxed).phase() == PHASE_COMPILING {
                yield_now();
                mtt.control_point(Some(&loc), dummy_step, &mut io);
            }
            assert_eq!(loc.load(Ordering::Relaxed).phase(), PHASE_COMPILED);
        }
    }

    #[test]
    fn simple_interpreter() {
        let mut mtt = MTBuilder::new().hot_threshold(2).init();

        const INC: u8 = 0;
        const RESTART: u8 = 1;
        // The program is silly. Do nothing twice, then start again.
        let prog = vec![INC, INC, RESTART];

        // Suppose the bytecode compiler for this imaginary language knows that the first bytecode
        // is the only place a loop can start.
        let locs = vec![Some(Location::new()), None, None];

        struct IO {
            prog: Vec<u8>,
            pc: usize,
            count: u64,
        }

        #[interp_step]
        fn simple_interp_step(tio: &mut IO) {
            match tio.prog[tio.pc] {
                INC => {
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
            mtt.control_point(loc, simple_interp_step, &mut tio);
        }

        loop {
            let loc = locs[tio.pc].as_ref();
            if tio.pc == 0 && loc.unwrap().load(Ordering::Relaxed).phase() == PHASE_COMPILED {
                break;
            }
            mtt.control_point(loc, simple_interp_step, &mut tio);
            yield_now();
        }

        assert_eq!(
            locs[0].as_ref().unwrap().load(Ordering::Relaxed).phase(),
            PHASE_COMPILED
        );

        assert_eq!(tio.pc, 0);

        // A trace was just compiled. Running it should execute INC twice.
        tio.count = 8;
        for i in 0..10 {
            let loc = locs[tio.pc].as_ref();
            mtt.control_point(loc, simple_interp_step, &mut tio);
            assert_eq!(tio.count, 10 + i * 2);
        }
    }

    #[test]
    fn simple_multithreaded_interpreter() {
        // If the threshold is too low (where "too low" is going to depend on many factors that we
        // can only guess at), it's less likely that we'll observe problematic interleavings,
        // because a single thread might execute everything it wants to without yielding once.
        const THRESHOLD: usize = 100000;
        const NUM_THREADS: usize = 16;

        let mut mtt = MTBuilder::new().hot_threshold(THRESHOLD).init();

        const INC: u8 = 0;
        const RESTART: u8 = 1;
        let prog = Arc::new(vec![INC, INC, RESTART]);

        struct IO {
            prog: Arc<Vec<u8>>,
            count: u64,
            pc: usize,
        }

        #[interp_step]
        fn simple_interp_step(tio: &mut IO) {
            match tio.prog[tio.pc] {
                INC => {
                    tio.pc += 1;
                    tio.count += 1;
                }
                RESTART => tio.pc = 0,
                _ => unreachable!(),
            }
        }

        // This tests for non-deterministic bugs in the Location statemachine: the only way we can
        // realistically find those is to keep trying tests over and over again.
        for _ in 0..10 {
            let locs = Arc::new(vec![Some(Location::new()), None, None]);
            let mut thrs = vec![];
            for _ in 0..NUM_THREADS {
                let locs = Arc::clone(&locs);
                let prog = Arc::clone(&prog);
                let mut tio = IO {
                    prog,
                    count: 0,
                    pc: 0,
                };
                let t = mtt
                    .mt()
                    .spawn(move |mut mtt| {
                        for _ in 0..tio.prog.len() * THRESHOLD {
                            let loc = locs[tio.pc].as_ref();
                            mtt.control_point(loc, simple_interp_step, &mut tio);
                        }
                    })
                    .unwrap();
                thrs.push(t);
            }

            let mut tio = IO {
                prog: Arc::clone(&prog),
                count: 0,
                pc: 0,
            };
            loop {
                let loc = locs[tio.pc].as_ref();
                if tio.pc == 0 {
                    let tag = loc.unwrap().load(Ordering::Relaxed).phase();
                    match tag {
                        PHASE_COUNTING | PHASE_TRACING | PHASE_TRACING_LOCK | PHASE_COMPILING
                        | PHASE_LOCKED => (),
                        PHASE_COMPILED => break,
                        _ => panic!(),
                    }
                }
                mtt.control_point(loc, simple_interp_step, &mut tio);
            }

            tio.pc = 0;
            tio.count = 8;
            for i in 0..10 {
                let loc = locs[tio.pc].as_ref();
                mtt.control_point(loc, simple_interp_step, &mut tio);
                assert_eq!(tio.count, 10 + i * 2);
            }

            for t in thrs {
                t.join().unwrap();
            }
        }
    }

    #[test]
    fn locations_dont_get_stuck_tracing() {
        const THRESHOLD: usize = 2;

        let mut mtt = MTBuilder::new().hot_threshold(THRESHOLD).init();

        const INC: u8 = 0;
        const RESTART: u8 = 1;
        let prog = Arc::new(vec![INC, INC, RESTART]);

        struct IO {
            prog: Arc<Vec<u8>>,
            count: u64,
            pc: usize,
        }

        #[interp_step]
        fn simple_interp_step(tio: &mut IO) {
            match tio.prog[tio.pc] {
                INC => {
                    tio.pc += 1;
                    tio.count += 1;
                }
                RESTART => tio.pc = 0,
                _ => unreachable!(),
            }
        }

        let locs = Arc::new(vec![Some(Location::new()), None, None]);
        let mut tio = IO {
            prog: Arc::clone(&prog),
            count: 0,
            pc: 0,
        };
        {
            let locs = Arc::clone(&locs);
            mtt.mt()
                .spawn(move |mut mtt| {
                    for _ in 0..tio.prog.len() * THRESHOLD + 1 {
                        let loc = locs[tio.pc].as_ref();
                        mtt.control_point(loc, simple_interp_step, &mut tio);
                    }
                })
                .unwrap()
                .join()
                .unwrap();
        }

        let loc = locs[0].as_ref();
        let tag = loc.unwrap().load(Ordering::Relaxed).phase();
        assert_eq!(tag, PHASE_TRACING);
        let mut tio = IO {
            prog: Arc::clone(&prog),
            count: 0,
            pc: 0,
        };
        mtt.control_point(loc, simple_interp_step, &mut tio);
        let tag = loc.unwrap().load(Ordering::Relaxed).phase();
        assert_eq!(tag, PHASE_DONT_TRACE);
    }

    #[test]
    fn stuck_locations_free_memory() {
        const THRESHOLD: usize = 2;

        let mtt = MTBuilder::new().hot_threshold(THRESHOLD).init();

        const INC: u8 = 0;
        const RESTART: u8 = 1;
        let prog = Arc::new(vec![INC, INC, RESTART]);

        struct IO {
            prog: Arc<Vec<u8>>,
            count: u64,
            pc: usize,
        }

        #[interp_step]
        fn simple_interp_step(tio: &mut IO) {
            match tio.prog[tio.pc] {
                INC => {
                    tio.pc += 1;
                    tio.count += 1;
                }
                RESTART => tio.pc = 0,
                _ => unreachable!(),
            }
        }

        let locs = Arc::new(vec![Some(Location::new()), None, None]);
        let mut tio = IO {
            prog: Arc::clone(&prog),
            count: 0,
            pc: 0,
        };
        {
            let locs = Arc::clone(&locs);
            mtt.mt()
                .spawn(move |mut mtt| {
                    for _ in 0..tio.prog.len() * THRESHOLD + 1 {
                        let loc = locs[tio.pc].as_ref();
                        mtt.control_point(loc, simple_interp_step, &mut tio);
                    }
                })
                .unwrap()
                .join()
                .unwrap();
        }

        let loc = locs[0].as_ref();
        let tag = loc.unwrap().load(Ordering::Relaxed).phase();
        assert_eq!(tag, PHASE_TRACING);
        // This is a weak test: at this point we hope that the dropped location frees its Arc. If
        // it doesn't, we won't have tested much. If it does, we at least get a modicum of "check
        // that it's not a double free" testing.
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
