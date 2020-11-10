use std::cmp::{Eq, PartialEq};
#[cfg(test)]
use std::time::Duration;
use std::{
    io,
    ops::Deref,
    panic::{catch_unwind, resume_unwind, UnwindSafe},
    ptr,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};
use yktrace::{start_tracing, ThreadTracer, TracingKind};

pub type HotThreshold = usize;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;
const PHASE_NUM_BITS: usize = 2;

// The current meta-tracing phase of a given location in the end-user's code. Consists of a tag and
// (optionally) a value. We expect the most commonly encountered tag at run-time is PHASE_COMPILED
// whose value is a pointer to memory. By also making that tag 0b00, we allow that index to be
// accessed without any further operations after the initial tag check.
const PHASE_TAG: usize = 0b11; // All of the other PHASE_ tags must fit in this.
const PHASE_COMPILED: usize = 0b00;
const PHASE_TRACING: usize = 0b01;
const PHASE_COUNTING: usize = 0b10; // The value specifies the current hot count.

/// A `Location` is a handle on a unique identifier for control point position in the end-user's
/// program (and is used by the `MT` to store data about that location). In other words, every
/// position that can be a control point also needs to have one `Location` value associated with
/// it. Note however that instances of `Location` may be freely cloned and shared across threads.
///
/// Program positions that can't be control points don't need an associated `Location`. For
/// interpreters that can't (or don't want) to be as selective, a simple (if moderately wasteful)
/// mechanism is for every bytecode or AST node to have its own `Location` (even for bytecodes or
/// nodes that can't be control points).
#[derive(Debug, Clone)]
pub struct Location {
    inner: Arc<LocationInner>,
}

impl Location {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(LocationInner::new()),
        }
    }
}

impl PartialEq for Location {
    fn eq(&self, other: &Location) -> bool {
        ptr::eq(Arc::as_ptr(&self.inner), Arc::as_ptr(&other.inner))
    }
}

impl Eq for Location {}

impl Deref for Location {
    type Target = LocationInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug)]
pub struct LocationInner {
    pack: AtomicUsize,
}

impl LocationInner {
    /// Create a fresh Location suitable for passing to `MT::control_point`.
    pub fn new() -> Self {
        Self {
            pack: AtomicUsize::new(PHASE_COUNTING),
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
    pub fn control_point<S, I>(&mut self, loc: Option<&Location>, step_fn: S, inputs: I) -> I
    where
        S: Fn(I) -> I,
    {
        // If a loop can start at this position then update the location and potentially start/stop
        // this thread's tracer.
        if let Some(loc) = loc {
            self._control_point(loc);
        }

        step_fn(inputs)
    }

    pub fn _control_point(&mut self, loc: &Location) {
        // Since we don't hold an explicit lock, updating a Location is tricky: we might read a
        // Location, work out what we'd like to update it to, and try updating it, only to find
        // that another thread interrupted us part way through. We therefore use compare_and_swap
        // to update values, allowing us to detect if we were interrupted. If we were interrupted,
        // we simply retry the whole operation.
        loop {
            let pack = &loc.inner.pack;
            // We need Acquire ordering, as PHASE_COMPILED will need to read information written to
            // external data as a result of the PHASE_TRACING -> PHASE_COMPILED transition.
            let lp = pack.load(Ordering::Acquire);
            match lp & PHASE_TAG {
                PHASE_COUNTING => {
                    let count = (lp & !PHASE_TAG) >> 2;
                    let new_pack;
                    if count >= self.inner.hot_threshold {
                        if self.inner.tracer.is_some() {
                            // This thread is already tracing. Note that we don't increment the hot
                            // count further.
                            break;
                        }
                        if pack.compare_and_swap(lp, PHASE_TRACING, Ordering::Release) == lp {
                            Rc::get_mut(&mut self.inner).unwrap().tracer =
                                Some((start_tracing(self.inner.tracing_kind), loc.clone()));
                            break;
                        }
                    } else {
                        new_pack = PHASE_COUNTING | ((count + 1) << PHASE_NUM_BITS);
                        if pack.compare_and_swap(lp, new_pack, Ordering::Release) == lp {
                            break;
                        }
                    }
                }
                PHASE_TRACING => {
                    // If this location is being traced by the current thread then we've finished a
                    // loop in the user program. In that case we can stop the tracer and compile
                    // code for the collected trace.
                    let stop_tracer = match self.inner.tracer {
                        // This thread isn't tracing, so another thread must be tracing this location.
                        None => false,
                        // We are tracing, but not this location. Another thread must be.
                        Some((_, ref tloc)) if tloc != loc => false,
                        // We are tracing this very location, so stop tracing.
                        Some(_) => true,
                    };

                    if stop_tracer {
                        if pack.compare_and_swap(lp, PHASE_COMPILED, Ordering::Release) == lp {
                            let _sir_trace = Rc::get_mut(&mut self.inner)
                                .unwrap()
                                .tracer
                                .take()
                                .unwrap()
                                .0
                                .stop_tracing();
                            // FIXME build TIR and compile. Eventually in a background thread?
                            break;
                        }
                    } else {
                        break;
                    }
                }
                PHASE_COMPILED => break, // FIXME call compiled trace, but don't call step_fn().
                _ => unreachable!(),
            }
        }
    }
}

/// The innards of a meta-tracer thread.
struct MTThreadInner {
    mt: MT,
    hot_threshold: HotThreshold,
    #[allow(dead_code)]
    tracing_kind: TracingKind,
    /// The active tracer and the location it started tracing from.
    /// The latter is a raw pointer to avoid lifetime issues. This is safe as the pointer is never
    /// dereferenced. It is only used as an identifier for a location.
    tracer: Option<(ThreadTracer, Location)>,
}

impl MTThreadInner {
    fn init(mt: MT) -> MTThread {
        let hot_threshold = mt.hot_threshold();
        let tracing_kind = mt.tracing_kind();
        let inner = MTThreadInner {
            mt,
            hot_threshold,
            tracing_kind,
            tracer: None,
        };
        MTThread {
            inner: Rc::new(inner),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::{black_box, Bencher};
    use super::*;

    fn dummy_step(inputs: ()) -> () {
        inputs
    }

    #[test]
    fn threshold_passed() {
        let hot_thrsh = 1500;
        let mut mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let lp = Location::new();
        for i in 0..hot_thrsh {
            mtt.control_point(Some(&lp), dummy_step, ());
            assert_eq!(
                lp.pack.load(Ordering::Relaxed),
                PHASE_COUNTING | ((i + 1) << PHASE_NUM_BITS)
            );
        }
        mtt.control_point(Some(&lp), dummy_step, ());
        assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_TRACING);
        mtt.control_point(Some(&lp), dummy_step, ());
        assert_eq!(lp.pack.load(Ordering::Relaxed), PHASE_COMPILED);
    }

    #[test]
    fn threaded_threshold_passed() {
        let hot_thrsh = 4000;
        let mut mtt = MTBuilder::new().hot_threshold(hot_thrsh).init();
        let loc = Location::new();
        let mut thrs = vec![];
        for _ in 0..hot_thrsh / 4 {
            let loc = loc.clone();
            let t = mtt
                .mt()
                .spawn(move |mut mtt| {
                    mtt.control_point(Some(&loc), dummy_step, ());
                    let c1 = loc.pack.load(Ordering::Relaxed);
                    assert_eq!(c1 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, ());
                    let c2 = loc.pack.load(Ordering::Relaxed);
                    assert_eq!(c2 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, ());
                    let c3 = loc.pack.load(Ordering::Relaxed);
                    assert_eq!(c3 & PHASE_TAG, PHASE_COUNTING);
                    mtt.control_point(Some(&loc), dummy_step, ());
                    let c4 = loc.pack.load(Ordering::Relaxed);
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
            mtt.control_point(Some(&loc), dummy_step, ());
            assert_eq!(loc.pack.load(Ordering::Relaxed), PHASE_TRACING);
            mtt.control_point(Some(&loc), dummy_step, ());
            assert_eq!(loc.pack.load(Ordering::Relaxed), PHASE_COMPILED);
        }
    }

    #[test]
    fn dumb_interpreter() {
        let mut mtt = MTBuilder::new().hot_threshold(2).init();

        // The program is silly. Do nothing twice, then start again.
        enum ByteCode {
            Nop,
            Restart,
        }
        let prog = vec![ByteCode::Nop, ByteCode::Nop, ByteCode::Restart];

        // Suppose the bytecode compiler for this imaginary language knows that the first bytecode
        // is the only place a loop can start.
        let locs = vec![Some(Location::new()), None, None];

        struct IO {
            prog: Vec<ByteCode>,
            pc: usize,
        }

        let interp_step = |mut tio: IO| {
            // FIXME make `inputs` a struct. Named fields would be much nicer.
            match tio.prog[tio.pc] {
                ByteCode::Nop => tio.pc += 1,
                ByteCode::Restart => tio.pc = 0,
            }
            tio
        };

        let mut tio = IO { prog, pc: 0 }; // bytecode, program counter.

        // The interpreter loop. In reality this would (syntactically) be an infinite loop.
        for _ in 0..10 {
            let loc = locs[tio.pc].as_ref();
            tio = mtt.control_point(loc, interp_step, tio);
        }

        assert_eq!(
            locs[0].as_ref().unwrap().pack.load(Ordering::Relaxed),
            PHASE_COMPILED
        );
    }

    #[bench]
    fn bench_single_threaded_control_point(b: &mut Bencher) {
        let mut mtt = MTBuilder::new().init();
        let lp = Location::new();
        b.iter(|| {
            for _ in 0..100000 {
                black_box(mtt.control_point(Some(&lp), dummy_step, ()));
            }
        });
    }

    #[bench]
    fn bench_multi_threaded_control_point(b: &mut Bencher) {
        let mtt = MTBuilder::new().init();
        let loc = Location::new();
        b.iter(|| {
            let mut thrs = vec![];
            for _ in 0..4 {
                let loc = loc.clone();
                let t = mtt
                    .mt()
                    .spawn(move |mut mtt| {
                        for _ in 0..100 {
                            black_box(mtt.control_point(Some(&loc), dummy_step, ()));
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

    #[test]
    fn clone_loc_ptr_ident() {
        let l = Location::new();
        for _ in 1..1000 {
            let c1 = l.clone();
            let c2 = l.clone();
            let c3 = l.clone();
            let c4 = l.clone();

            assert!(ptr::eq(Arc::as_ptr(&l.inner), Arc::as_ptr(&c1.inner)));
            assert!(ptr::eq(Arc::as_ptr(&l.inner), Arc::as_ptr(&c2.inner)));
            assert!(ptr::eq(Arc::as_ptr(&l.inner), Arc::as_ptr(&c3.inner)));
            assert!(ptr::eq(Arc::as_ptr(&l.inner), Arc::as_ptr(&c4.inner)));
        }
    }
}
