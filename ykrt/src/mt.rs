//! The main end-user interface to the meta-tracing system.

// FIXME these were added during the big code deletion event that occurred after we decided to use
// LLVM IR instead of SIR/TIR. Large chunks of the system suddenly became todo!() meaning that code
// that we do still want to keep (some of which is scary concurrency stuff) became dead in from
// Rust's POV.
#![allow(unreachable_code, unused_variables, dead_code)]

#[cfg(test)]
use std::time::Duration;
use std::{
    cmp,
    collections::VecDeque,
    error::Error,
    io,
    marker::PhantomData,
    ptr,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

use num_cpus;
use parking_lot::{Condvar, Mutex, MutexGuard};
use parking_lot_core::SpinWait;

use crate::location::{HotLocation, Location, State, ThreadIdInner};
use yktrace::{start_tracing, CompiledTrace, IRTrace, TracingKind};

// The HotThreshold must be less than a machine word wide for [`Location::Location`] to do its
// pointer tagging thing. We therefore choose a type which makes this statically clear to
// users rather than having them try to use (say) u64::max() on a 64 bit machine and get a run-time
// error.
#[cfg(target_pointer_width = "64")]
pub type HotThreshold = u32;
#[cfg(target_pointer_width = "64")]
type AtomicHotThreshold = AtomicU32;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;

/// Configure a meta-tracer. Note that a process can only have one meta-tracer active at one point.
pub struct MTBuilder {
    hot_threshold: HotThreshold,
    /// The maximum number of simultaneous worker (e.g. compilation) threads.
    max_worker_threads: usize,
    /// The kind of tracer to use.
    tracing_kind: TracingKind,
}

impl MTBuilder {
    /// Create a meta-tracer with default parameters.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            hot_threshold: DEFAULT_HOT_THRESHOLD,
            max_worker_threads: cmp::max(1, num_cpus::get() - 1),
            tracing_kind: TracingKind::default(),
        })
    }

    /// Consume the `MTBuilder` and create a meta-tracer, returning the
    /// [`MTThread`](struct.MTThread.html) representing the current thread.
    pub fn build(self) -> MTThread {
        MTInner::init(
            self.hot_threshold,
            self.max_worker_threads,
            self.tracing_kind,
        )
    }

    /// Change this meta-tracer builder's `hot_threshold` value. Returns `Ok` if `hot_threshold` is
    /// an acceptable value or `Err` otherwise.
    pub fn hot_threshold(mut self, hot_threshold: HotThreshold) -> Result<Self, ()> {
        self.hot_threshold = hot_threshold;
        Ok(self)
    }

    /// Sets the maximum number of worker threads to `max(1, num)`. Defaults to `max(1,
    /// num_cpus::get() - 1)`.
    pub fn max_worker_threads(mut self, num: usize) -> Self {
        self.max_worker_threads = cmp::max(1, num);
        self
    }

    /// Select the kind of tracing to use. Returns `Ok` if the run-time environment is capable of
    /// using `TracingKind` or `Err` otherwise.
    pub fn tracing_kind(mut self, tracing_kind: TracingKind) -> Result<Self, ()> {
        self.tracing_kind = tracing_kind;
        Ok(self)
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

    /// Return this meta-tracer's maximum number of worker threads.
    pub fn max_worker_threads(&self) -> usize {
        self.inner.max_worker_threads.load(Ordering::Relaxed)
    }

    /// Return the kind of tracing that this meta-tracer is using.
    pub fn tracing_kind(&self) -> TracingKind {
        self.inner.tracing_kind
    }

    /// Create a new thread that can be used in the meta-tracer: the new thread that is created is
    /// handed a [`MTThread`](struct.MTThread.html) from which the `MT` itself can be accessed.
    pub fn spawn<F, T>(&self, f: F) -> io::Result<JoinHandle<T>>
    where
        F: FnOnce(MTThread) -> T + Send + 'static,
        T: Send + 'static,
    {
        let mt_cl = self.clone();
        thread::Builder::new().spawn(move || {
            mt_cl
                .inner
                .active_user_threads
                .fetch_add(1, Ordering::Relaxed);
            let r = f(MTThreadInner::init(mt_cl.clone()));
            mt_cl
                .inner
                .active_user_threads
                .fetch_sub(1, Ordering::Relaxed);
            r
        })
    }

    /// Queue `job` to be run on a worker thread.
    fn queue_job(&self, job: Box<dyn FnOnce() + Send>) {
        // We have a very simple model of worker threads. Each time a job is queued, we spin up a
        // new worker thread iff we aren't already running the maximum number of worker threads.
        // Once started, a worker thread never dies, waiting endlessly for work.

        let inner = Arc::clone(&self.inner);
        let (cv, mtx) = &inner.job_queue;
        mtx.lock().push_back(job);
        cv.notify_one();

        let max_jobs = inner.max_worker_threads.load(Ordering::Relaxed);
        if inner.active_worker_threads.load(Ordering::Relaxed) < max_jobs {
            // At the point of the `load` on the previous line, we weren't running the maximum
            // number of worker threads. There is now a possible race condition where multiple
            // threads calling `queue_job` could try creating multiple worker threads and push us
            // over the maximum worker thread limit.
            if inner.active_worker_threads.fetch_add(1, Ordering::Relaxed) > max_jobs {
                // Another thread(s) is also spinning up another worker thread and they won the
                // race.
                inner.active_worker_threads.fetch_sub(1, Ordering::Relaxed);
                return;
            }

            let inner_cl = Arc::clone(&self.inner);
            thread::spawn(move || {
                let (cv, mtx) = &inner_cl.job_queue;
                let mut lock = mtx.lock();
                loop {
                    match lock.pop_front() {
                        Some(x) => MutexGuard::unlocked(&mut lock, x),
                        None => cv.wait(&mut lock),
                    }
                }
            });
        }
    }
}

impl Drop for MT {
    fn drop(&mut self) {
        MT_ACTIVE.store(false, Ordering::Relaxed);
    }
}

/// The innards of a meta-tracer.
struct MTInner {
    hot_threshold: AtomicHotThreshold,
    /// The number of threads currently running the user's interpreter (directly or via JITed
    /// code).
    active_user_threads: AtomicUsize,
    /// The ordered queue of compilation worker functions.
    job_queue: (Condvar, Mutex<VecDeque<Box<dyn FnOnce() + Send>>>),
    /// The hard cap on the number of worker threads.
    max_worker_threads: AtomicUsize,
    /// How many worker threads are currently running. Note that this may temporarily be `>`
    /// [`max_worker_threads`].
    active_worker_threads: AtomicUsize,
    tracing_kind: TracingKind,
}

/// It's only safe to have one `MT` instance active at a time.
static MT_ACTIVE: AtomicBool = AtomicBool::new(false);

impl MTInner {
    /// Create a new `MT`, wrapped immediately in an [`MTThread`](struct.MTThread.html).
    fn init(
        hot_threshold: HotThreshold,
        max_worker_threads: usize,
        tracing_kind: TracingKind,
    ) -> MTThread {
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
            hot_threshold: AtomicHotThreshold::new(hot_threshold),
            job_queue: (Condvar::new(), Mutex::new(VecDeque::new())),
            active_user_threads: AtomicUsize::new(1),
            max_worker_threads: AtomicUsize::new(max_worker_threads),
            active_worker_threads: AtomicUsize::new(0),
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
    // Raw pointers are neither send nor sync.
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl MTThread {
    /// Return a meta-tracer [`MT`](struct.MT.html) struct.
    pub fn mt(&self) -> &MT {
        &self.inner.mt
    }

    /// Attempt to execute a compiled trace for location `loc`.
    pub fn control_point(&mut self, loc: Option<&Location>) {
        // If a loop can start at this position then update the location and potentially start/stop
        // this thread's tracer.
        if let Some(loc) = loc {
            // FIXME transition_location() may execute a trace, in which case we shouldn't execute
            // the step_fn -- anyway the step_fn design is going to be changed too.
            self.transition_location(loc);
        }
    }

    /// `Location`s represent a statemachine: this function transitions to the next state (which
    /// may be the same as the previous state!). If this results in a compiled trace, it returns
    /// `Some(pointer_to_trace_function)`.
    fn transition_location(&mut self, loc: &Location) {
        let mut ls = loc.load(Ordering::Relaxed);

        if ls.is_counting() {
            debug_assert!(!ls.is_locked());
            debug_assert!(!ls.is_parked());

            let count = ls.count();
            if count < self.inner.hot_threshold {
                // Try incrementing this location's hot count. We make no guarantees that this will
                // succeed because under heavy contention we can end up racing with many other
                // threads and it's not worth our time to halt execution merely to have an accurate
                // hot count. However, we do try semi-hard to enforce monotonicity (i.e. preventing
                // the hot count from going backwards) which can happen if an "older" thread has
                // been paused for a long time. Even in that case, though, we do not try endlessly.
                let mut spinwait = SpinWait::new();
                loop {
                    match loc.compare_exchange_weak(
                        ls,
                        ls.with_count(count + 1),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(new_ls) => {
                            if !new_ls.is_counting() {
                                // Another thread has probably started tracing this Location.
                                break;
                            }
                            if new_ls.count() >= count {
                                // Although our increment hasn't worked, at least the count hasn't
                                // gone backwards: rather than holding this thread up, let's get
                                // back to the interpreter.
                                break;
                            }
                            ls = new_ls;
                            if spinwait.spin() {
                                // We don't want to park this thread, so even though we can see the
                                // count is going backwards, go back to the interpreter.
                                break;
                            }
                        }
                    }
                }
                return;
            } else {
                if self.inner.tracing.is_some() {
                    // This thread is already tracing another Location, so either another
                    // thread needs to trace this Location or this thread needs to wait
                    // until the current round of tracing has completed. Either way,
                    // there's no point incrementing the hot count.
                    return;
                }
                // To avoid racing with another thread that may also try starting to trace this
                // location at the same time, we need to initialise and lock the Location, which we
                // perform in a single step. Since this is such a critical step, and since we're
                // prepared to bail out early, there's no point in yielding: either we win the race
                // by trying repeatedly or we give up entirely.
                let hl_ptr = Box::into_raw(Box::new(HotLocation::Tracing(None)));
                let new_ls = State::new().with_hotlocation(hl_ptr).with_lock();
                loop {
                    debug_assert!(!ls.is_locked());
                    match loc.compare_exchange_weak(
                        ls,
                        new_ls,
                        Ordering::Acquire,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            // We've initialised this Location and obtained the lock, so we can now
                            // start tracing for real.
                            let tid = Arc::clone(&self.inner.tid);
                            let tt = start_tracing(self.inner.tracing_kind);
                            *unsafe { new_ls.hot_location() } =
                                HotLocation::Tracing(Some((tid, tt)));
                            Rc::get_mut(&mut self.inner).unwrap().tracing =
                                Some(hl_ptr as *const ());
                            loc.unlock();
                            return;
                        }
                        Err(x) => {
                            if x.is_locked() {
                                // We probably raced with another thread locking this Location in order to
                                // start tracing. It's unlikely to be worth us spending time contending
                                // with that other thread.
                                unsafe { Box::from_raw(hl_ptr) };
                                return;
                            }
                            ls = x;
                        }
                    }
                }
            }
        } else {
            // There's no point contending with other threads, so in general we don't want to
            // continually try grabbing the lock.
            match loc.try_lock() {
                Ok(x) => ls = x,
                Err(_) => {
                    // If this thread is tracing we need to grab the lock so that we can stop
                    // tracing, otherwise we return to the interpreter.
                    if self.inner.tracing.is_none() {
                        return;
                    }
                    match loc.lock() {
                        Ok(x) => ls = x,
                        Err(()) => unreachable!(),
                    }
                }
            }
            let hl = unsafe { ls.hot_location() };
            let hl_ptr = hl as *mut _ as *mut ();
            match hl {
                HotLocation::Compiled(tr) => {
                    // FIXME: If we want to free compiled traces, we'll need to refcount (or use
                    // a GC) to know if anyone's executing that trace at the moment.
                    todo!(); // FIXME run executable trace code here.
                    loc.unlock();
                }
                HotLocation::Compiling(mtx) => {
                    let tr = {
                        let gd = mtx.try_lock();
                        if gd.is_none() {
                            // Compilation is ongoing.
                            loc.unlock();
                            return;
                        }
                        let mut gd = gd.unwrap();
                        if gd.is_none() {
                            // Compilation is ongoing.
                            loc.unlock();
                            return;
                        }
                        (*gd).take().unwrap()
                    };
                    *hl = HotLocation::Compiled(tr);
                    loc.unlock();
                    return;
                }
                HotLocation::Tracing(opt) => {
                    match self.inner.tracing {
                        Some(other_hl_ptr) => {
                            // This thread is tracing something...
                            if !ptr::eq(hl_ptr, other_hl_ptr) {
                                // but not this Location.
                                loc.unlock();
                                return;
                            }
                        }
                        None => {
                            // This thread isn't tracing anything.
                            if Arc::strong_count(&opt.as_ref().unwrap().0) == 1 {
                                // Another thread was tracing this location but it's terminated.
                                // FIXME: we should probably have some sort of occasional retry
                                // heuristic rather than simply saying "never try tracing this
                                // Location again."
                                *hl = HotLocation::DontTrace;
                            }
                            loc.unlock();
                            return;
                        }
                    }
                    // This thread is tracing this location: we must, therefore, have finished
                    // tracing the loop. Notice that the ".1" implicitly drops the
                    // Arc<ThreadIdInner> so that other threads won't think this thread has died
                    // while tracing.
                    match opt.take().unwrap().1.stop_tracing() {
                        Ok(sir) => {
                            let mtx = Arc::new(Mutex::new(None));
                            let mtx_cl = Arc::clone(&mtx);
                            *hl = HotLocation::Compiling(mtx);
                            loc.unlock();
                            self.queue_compile_job(sir, mtx_cl);
                        }
                        Err(_) => todo!(),
                    }
                    return;
                }
                HotLocation::DontTrace => {
                    loc.unlock();
                    return;
                }
            }
        }
    }

    /// Add a compilation job for `sir` to the global work queue.
    fn queue_compile_job(&self, trace: IRTrace, mtx: Arc<Mutex<Option<Box<CompiledTrace>>>>) {
        let f = Box::new(move || {
            // FIXME compile IR trace to executable code here.
            let compiled = todo!();
            *mtx.lock() = Some(Box::new(compiled));
            // FIXME: although we've now put the compiled trace into the mutex, there's no
            // guarantee that the Location for which we're compiling will ever be executed
            // again. In such a case, the memory has, in essence, leaked.
        });

        self.inner.mt.queue_job(f);
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
    /// If this thread is tracing, store a pointer to the `HotLocation`: this allows us to
    /// differentiate which thread is actually tracing the location.
    tracing: Option<*const ()>,
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
            tracing: None,
        };
        MTThread {
            inner: Rc::new(inner),
            _dont_send_or_sync_me: PhantomData,
        }
    }
}

// FIXME These tests were broken during the big code deletion event of 2021 where we moved to LLVM
// IR. I've left them in (commented), because although they are busted, the ykrt API isn't going to
// change much and I think we can recover these test at some point.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::HotLocationDiscriminants;

    fn hotlocation_discriminant(loc: &Location) -> Option<HotLocationDiscriminants> {
        match loc.lock() {
            Ok(ls) => {
                let x = HotLocationDiscriminants::from(&*unsafe { ls.hot_location() });
                loc.unlock();
                Some(x)
            }
            Err(()) => None,
        }
    }

    #[test]
    #[ignore]
    fn threshold_passed() {
        let hot_thrsh: HotThreshold = 1500;
        let mut mtt = MTBuilder::new()
            .unwrap()
            .hot_threshold(hot_thrsh)
            .unwrap()
            .build();
        let loc = Location::new();
        for i in 0..hot_thrsh {
            mtt.control_point(Some(&loc));
            assert_eq!(loc.load(Ordering::Relaxed), State::new().with_count(i + 1));
        }
        assert!(loc.load(Ordering::Relaxed).is_counting());
        mtt.control_point(Some(&loc));
        assert_eq!(
            hotlocation_discriminant(&loc).unwrap(),
            HotLocationDiscriminants::Tracing
        );
        mtt.control_point(Some(&loc));
        assert!([
            HotLocationDiscriminants::Compiling,
            HotLocationDiscriminants::Compiled
        ]
        .contains(&hotlocation_discriminant(&loc).unwrap()));
    }

    //    #[test]
    //    fn stop_while_tracing() {
    //        let hot_thrsh = 5;
    //        let mut mtt = MTBuilder::new()
    //            .unwrap()
    //            .hot_threshold(hot_thrsh)
    //            .unwrap()
    //            .build();
    //        let loc = Location::new();
    //        let mut ctx = EmptyInterpCtx {};
    //        for i in 0..hot_thrsh {
    //            mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //            assert!(loc.load(Ordering::Relaxed).is_counting());
    //            assert_eq!(loc.load(Ordering::Relaxed).count(), i + 1);
    //        }
    //        assert!(loc.load(Ordering::Relaxed).is_counting());
    //        mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //        assert_eq!(
    //            hotlocation_discriminant(&loc).unwrap(),
    //            HotLocationDiscriminants::Tracing
    //        );
    //    }
    //
    //    #[test]
    //    fn threaded_threshold_passed() {
    //        let hot_thrsh = 4000;
    //        let mut mtt = MTBuilder::new()
    //            .unwrap()
    //            .hot_threshold(hot_thrsh)
    //            .unwrap()
    //            .build();
    //        let loc = Arc::new(Location::new());
    //        let mut thrs = vec![];
    //        for _ in 0..hot_thrsh / 4 {
    //            let loc = Arc::clone(&loc);
    //            let t = mtt
    //                .mt()
    //                .spawn(move |mut mtt| {
    //                    let mut ctx = EmptyInterpCtx {};
    //                    mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //                    let c1 = loc.load(Ordering::Relaxed);
    //                    assert!(c1.is_counting());
    //                    mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //                    let c2 = loc.load(Ordering::Relaxed);
    //                    assert!(c2.is_counting());
    //                    mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //                    let c3 = loc.load(Ordering::Relaxed);
    //                    assert!(c3.is_counting());
    //                    mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //                    let c4 = loc.load(Ordering::Relaxed);
    //                    assert!(c4.is_counting());
    //                    assert!(c4.count() >= c3.count());
    //                    assert!(c3.count() >= c2.count());
    //                    assert!(c2.count() >= c1.count());
    //                })
    //                .unwrap();
    //            thrs.push(t);
    //        }
    //        for t in thrs {
    //            t.join().unwrap();
    //        }
    //        {
    //            let mut ctx = EmptyInterpCtx {};
    //            // Thread contention means that some count updates might have been dropped, so we have
    //            // to keep the interpreter going until the threshold really is reached.
    //            while loc.load(Ordering::Relaxed).is_counting() {
    //                mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //            }
    //            assert_eq!(
    //                hotlocation_discriminant(&loc).unwrap(),
    //                HotLocationDiscriminants::Tracing
    //            );
    //            mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //
    //            while hotlocation_discriminant(&loc) == Some(HotLocationDiscriminants::Compiling) {
    //                yield_now();
    //                mtt.control_point(Some(&loc), empty_step, &mut ctx);
    //            }
    //            assert_eq!(
    //                hotlocation_discriminant(&loc).unwrap(),
    //                HotLocationDiscriminants::Compiled
    //            );
    //        }
    //    }
    //
    //    #[test]
    //    fn simple_interpreter() {
    //        let mut mtt = MTBuilder::new().unwrap().hot_threshold(2).unwrap().build();
    //
    //        const INC: u8 = 0;
    //        const RESTART: u8 = 1;
    //        // The program is silly. Do nothing twice, then start again.
    //        let prog = vec![INC, INC, RESTART];
    //
    //        // Suppose the bytecode compiler for this imaginary language knows that the first bytecode
    //        // is the only place a loop can start.
    //        let locs = vec![Some(Location::new()), None, None];
    //
    //        struct InterpCtx {
    //            prog: Vec<u8>,
    //            pc: usize,
    //            count: u64,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            match ctx.prog[ctx.pc] {
    //                INC => {
    //                    ctx.pc += 1;
    //                    ctx.count += 1;
    //                }
    //                RESTART => ctx.pc = 0,
    //                _ => unreachable!(),
    //            };
    //            true
    //        }
    //
    //        let mut ctx = InterpCtx {
    //            prog,
    //            pc: 0,
    //            count: 0,
    //        };
    //
    //        // The interpreter loop. In reality this would (syntactically) be an infinite loop.
    //        for _ in 0..12 {
    //            let loc = locs[ctx.pc].as_ref();
    //            mtt.control_point(loc, simple_interp_step, &mut ctx);
    //        }
    //
    //        loop {
    //            let loc = locs[ctx.pc].as_ref();
    //            if ctx.pc == 0
    //                && hotlocation_discriminant(&loc.unwrap())
    //                    == Some(HotLocationDiscriminants::Compiled)
    //            {
    //                break;
    //            }
    //            mtt.control_point(loc, simple_interp_step, &mut ctx);
    //            yield_now();
    //        }
    //
    //        // A trace was just compiled. Running it should execute INC twice.
    //        ctx.count = 8;
    //        for i in 0..10 {
    //            let loc = locs[ctx.pc].as_ref();
    //            mtt.control_point(loc, simple_interp_step, &mut ctx);
    //            assert_eq!(ctx.count, 10 + i * 2);
    //        }
    //    }
    //
    //    #[test]
    //    fn simple_multithreaded_interpreter() {
    //        // If the threshold is too low (where "too low" is going to depend on many factors that we
    //        // can only guess at), it's less likely that we'll observe problematic interleavings,
    //        // because a single thread might execute everything it wants to without yielding once.
    //        const THRESHOLD: HotThreshold = 100000;
    //        const NUM_THREADS: usize = 16;
    //
    //        let mut mtt = MTBuilder::new()
    //            .unwrap()
    //            .hot_threshold(THRESHOLD)
    //            .unwrap()
    //            .build();
    //
    //        const INC: u8 = 0;
    //        const RESTART: u8 = 1;
    //        let prog = Arc::new(vec![INC, INC, RESTART]);
    //
    //        struct InterpCtx {
    //            prog: Arc<Vec<u8>>,
    //            count: u64,
    //            pc: usize,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            match ctx.prog[ctx.pc] {
    //                INC => {
    //                    ctx.pc += 1;
    //                    ctx.count += 1;
    //                }
    //                RESTART => ctx.pc = 0,
    //                _ => unreachable!(),
    //            };
    //            true
    //        }
    //
    //        // This tests for non-deterministic bugs in the Location statemachine: the only way we can
    //        // realistically find those is to keep trying tests over and over again.
    //        for _ in 0..10 {
    //            let locs = Arc::new(vec![Some(Location::new()), None, None]);
    //            let mut thrs = vec![];
    //            for _ in 0..NUM_THREADS {
    //                let locs = Arc::clone(&locs);
    //                let prog = Arc::clone(&prog);
    //                let mut ctx = InterpCtx {
    //                    prog,
    //                    count: 0,
    //                    pc: 0,
    //                };
    //                let t = mtt
    //                    .mt()
    //                    .spawn(move |mut mtt| {
    //                        for _ in 0..ctx.prog.len() * usize::try_from(THRESHOLD).unwrap() {
    //                            let loc = locs[ctx.pc].as_ref();
    //                            mtt.control_point(loc, simple_interp_step, &mut ctx);
    //                        }
    //                    })
    //                    .unwrap();
    //                thrs.push(t);
    //            }
    //
    //            let mut ctx = InterpCtx {
    //                prog: Arc::clone(&prog),
    //                count: 0,
    //                pc: 0,
    //            };
    //            loop {
    //                let loc = locs[ctx.pc].as_ref();
    //                if ctx.pc == 0
    //                    && hotlocation_discriminant(&loc.unwrap())
    //                        == Some(HotLocationDiscriminants::Compiled)
    //                {
    //                    break;
    //                }
    //                mtt.control_point(loc, simple_interp_step, &mut ctx);
    //            }
    //
    //            ctx.pc = 0;
    //            ctx.count = 8;
    //            for i in 0..10 {
    //                let loc = locs[ctx.pc].as_ref();
    //                mtt.control_point(loc, simple_interp_step, &mut ctx);
    //                assert!(ctx.pc == 0 || ctx.pc == 1);
    //                while ctx.pc != 0 {
    //                    let loc = locs[ctx.pc].as_ref();
    //                    mtt.control_point(loc, simple_interp_step, &mut ctx);
    //                }
    //                assert_eq!(ctx.count, 10 + i * 2);
    //            }
    //
    //            for t in thrs {
    //                t.join().unwrap();
    //            }
    //        }
    //    }
    //
    //    #[test]
    //    fn locations_dont_get_stuck_tracing() {
    //        const THRESHOLD: HotThreshold = 2;
    //
    //        let mut mtt = MTBuilder::new()
    //            .unwrap()
    //            .hot_threshold(THRESHOLD)
    //            .unwrap()
    //            .build();
    //
    //        const INC: u8 = 0;
    //        const RESTART: u8 = 1;
    //        let prog = Arc::new(vec![INC, INC, RESTART]);
    //
    //        struct InterpCtx {
    //            prog: Arc<Vec<u8>>,
    //            count: u64,
    //            pc: usize,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            match ctx.prog[ctx.pc] {
    //                INC => {
    //                    ctx.pc += 1;
    //                    ctx.count += 1;
    //                }
    //                RESTART => ctx.pc = 0,
    //                _ => unreachable!(),
    //            };
    //            true
    //        }
    //
    //        let locs = Arc::new(vec![Some(Location::new()), None, None]);
    //        let mut ctx = InterpCtx {
    //            prog: Arc::clone(&prog),
    //            count: 0,
    //            pc: 0,
    //        };
    //        {
    //            let locs = Arc::clone(&locs);
    //            mtt.mt()
    //                .spawn(move |mut mtt| {
    //                    for _ in 0..ctx.prog.len() * usize::try_from(THRESHOLD).unwrap() + 1 {
    //                        let loc = locs[ctx.pc].as_ref();
    //                        mtt.control_point(loc, simple_interp_step, &mut ctx);
    //                    }
    //                })
    //                .unwrap()
    //                .join()
    //                .unwrap();
    //        }
    //
    //        let loc = locs[0].as_ref();
    //        assert_eq!(
    //            hotlocation_discriminant(&loc.unwrap()),
    //            Some(HotLocationDiscriminants::Tracing)
    //        );
    //        let mut ctx = InterpCtx {
    //            prog: Arc::clone(&prog),
    //            count: 0,
    //            pc: 0,
    //        };
    //        mtt.control_point(loc, simple_interp_step, &mut ctx);
    //        assert_eq!(
    //            hotlocation_discriminant(&loc.unwrap()),
    //            Some(HotLocationDiscriminants::DontTrace)
    //        );
    //    }
    //
    //    #[test]
    //    fn more_hot_locations_than_worker_threads() {
    //        // This test indirectly tests code around capping the maximum number of worker threads.
    //        // We'd really like to check that not too many worker threads run at once, but that's
    //        // surprisingly tricky to test reliably because for all we know the system might just so
    //        // happen to run our worker threads one after the other. We therefore settle for running a
    //        // program that compiles multiple locations, with luck more-or-less at the same time,
    //        // hoping that this tests the various interactions around queueing etc.
    //
    //        let num_threads: usize = cmp::max(4, num_cpus::get());
    //        let mtt = MTBuilder::new()
    //            .unwrap()
    //            .max_worker_threads(num_threads / 2)
    //            .hot_threshold(2)
    //            .unwrap()
    //            .build();
    //
    //        const INC: u8 = 0;
    //        const RESTART: u8 = 1;
    //        // This test assumes throughout that each loop in the program is 3 bytecodes long.
    //        let mut prog = Vec::with_capacity(3 * num_threads);
    //        let mut locs = Vec::with_capacity(3 * num_threads);
    //        for _ in 0..num_threads {
    //            prog.extend(vec![INC, INC, RESTART]);
    //            locs.extend(vec![Some(Location::new()), None, None]);
    //        }
    //        let prog = Arc::new(prog);
    //        let locs = Arc::new(locs);
    //
    //        struct InterpCtx {
    //            prog: Arc<Vec<u8>>,
    //            count: u64,
    //            pc: usize,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            match ctx.prog[ctx.pc] {
    //                INC => {
    //                    ctx.pc += 1;
    //                    ctx.count += 1;
    //                }
    //                RESTART => ctx.pc -= 2,
    //                _ => unreachable!(),
    //            };
    //            true
    //        }
    //
    //        let mut thrs = Vec::with_capacity(num_threads);
    //        for i in 0..num_threads {
    //            let prog = Arc::clone(&prog);
    //            let locs = Arc::clone(&locs);
    //            let t = mtt
    //                .mt()
    //                .spawn(move |mut mtt| {
    //                    let mut ctx = InterpCtx {
    //                        prog: Arc::clone(&prog),
    //                        count: 0,
    //                        pc: i * 3,
    //                    };
    //                    loop {
    //                        let loc = locs[ctx.pc].as_ref();
    //                        if let Some(l) = loc {
    //                            if hotlocation_discriminant(l)
    //                                == Some(HotLocationDiscriminants::Compiled)
    //                            {
    //                                break;
    //                            }
    //                        }
    //                        mtt.control_point(loc, simple_interp_step, &mut ctx);
    //                    }
    //                })
    //                .unwrap();
    //            thrs.push(t);
    //        }
    //        for t in thrs {
    //            t.join().unwrap();
    //        }
    //    }
    //
    //    #[test]
    //    fn stuck_locations_free_memory() {
    //        const THRESHOLD: HotThreshold = 2;
    //
    //        let mtt = MTBuilder::new()
    //            .unwrap()
    //            .hot_threshold(THRESHOLD)
    //            .unwrap()
    //            .build();
    //
    //        const INC: u8 = 0;
    //        const RESTART: u8 = 1;
    //        let prog = Arc::new(vec![INC, INC, RESTART]);
    //
    //        struct InterpCtx {
    //            prog: Arc<Vec<u8>>,
    //            count: u64,
    //            pc: usize,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            match ctx.prog[ctx.pc] {
    //                INC => {
    //                    ctx.pc += 1;
    //                    ctx.count += 1;
    //                }
    //                RESTART => ctx.pc = 0,
    //                _ => unreachable!(),
    //            };
    //            true
    //        }
    //
    //        let locs = Arc::new(vec![Some(Location::new()), None, None]);
    //        let mut ctx = InterpCtx {
    //            prog: Arc::clone(&prog),
    //            count: 0,
    //            pc: 0,
    //        };
    //        {
    //            let locs = Arc::clone(&locs);
    //            mtt.mt()
    //                .spawn(move |mut mtt| {
    //                    for _ in 0..ctx.prog.len() * usize::try_from(THRESHOLD).unwrap() + 1 {
    //                        let loc = locs[ctx.pc].as_ref();
    //                        mtt.control_point(loc, simple_interp_step, &mut ctx);
    //                    }
    //                })
    //                .unwrap()
    //                .join()
    //                .unwrap();
    //        }
    //
    //        let loc = locs[0].as_ref();
    //        assert_eq!(
    //            hotlocation_discriminant(&loc.unwrap()),
    //            Some(HotLocationDiscriminants::Tracing)
    //        );
    //        // This is a weak test: at this point we hope that the dropped location frees its Arc. If
    //        // it doesn't, we won't have tested much. If it does, we at least get a modicum of "check
    //        // that it's not a double free" testing.
    //    }
    //
    //    #[bench]
    //    fn bench_single_threaded_control_point(b: &mut Bencher) {
    //        let mut mtt = MTBuilder::new().unwrap().build();
    //        let lp = Location::new();
    //        let mut io = EmptyInterpCtx {};
    //        b.iter(|| {
    //            for _ in 0..100000 {
    //                black_box(mtt.control_point(Some(&lp), empty_step, &mut io));
    //            }
    //        });
    //    }
    //
    //    #[bench]
    //    fn bench_multi_threaded_control_point(b: &mut Bencher) {
    //        let mtt = MTBuilder::new().unwrap().build();
    //        let loc = Arc::new(Location::new());
    //        b.iter(|| {
    //            let mut thrs = vec![];
    //            for _ in 0..4 {
    //                let loc = Arc::clone(&loc);
    //                let t = mtt
    //                    .mt()
    //                    .spawn(move |mut mtt| {
    //                        for _ in 0..100 {
    //                            let mut io = EmptyInterpCtx {};
    //                            black_box(mtt.control_point(Some(&loc), empty_step, &mut io));
    //                        }
    //                    })
    //                    .unwrap();
    //                thrs.push(t);
    //            }
    //            for t in thrs {
    //                t.join().unwrap();
    //            }
    //        });
    //    }
    //
    //    #[test]
    //    fn stopgapping() {
    //        let mut mtt = MTBuilder::new().unwrap().hot_threshold(2).unwrap().build();
    //
    //        const INC: u8 = 0;
    //        const RESTART: u8 = 1;
    //        let prog = vec![INC, INC, RESTART];
    //        let locs = vec![Some(Location::new()), None, None];
    //
    //        struct InterpCtx {
    //            prog: Vec<u8>,
    //            pc: usize,
    //            run: u8,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            match ctx.prog[ctx.pc] {
    //                INC => {
    //                    ctx.pc += 1;
    //                    if ctx.run > 0 {
    //                        ctx.run += 1;
    //                    }
    //                }
    //                RESTART => ctx.pc = 0,
    //                _ => unreachable!(),
    //            };
    //            true
    //        }
    //
    //        let mut ctx = InterpCtx {
    //            prog,
    //            pc: 0,
    //            run: 0,
    //        };
    //
    //        // Run until a trace has been compiled.
    //        loop {
    //            let loc = locs[ctx.pc].as_ref();
    //            if ctx.pc == 0
    //                && hotlocation_discriminant(&loc.unwrap())
    //                    == Some(HotLocationDiscriminants::Compiled)
    //            {
    //                break;
    //            }
    //            mtt.control_point(locs[ctx.pc].as_ref(), simple_interp_step, &mut ctx);
    //        }
    //
    //        assert_eq!(ctx.pc, 0);
    //        assert_eq!(ctx.run, 0);
    //
    //        // Now fail a guard.
    //        ctx.run = 1;
    //        let loc = locs[ctx.pc].as_ref();
    //        mtt.control_point(loc, simple_interp_step, &mut ctx);
    //        assert_eq!(ctx.run, 2);
    //    }
    //
    //    /// Tests that the stopgap interpreter returns the correct boolean to abort interpreting.
    //    #[test]
    //    fn loop_interpreter() {
    //        let mut mtt = MTBuilder::new().unwrap().hot_threshold(2).unwrap().build();
    //
    //        let loc = Location::new();
    //
    //        struct InterpCtx {
    //            counter: u8,
    //        }
    //
    //        #[interp_step]
    //        fn simple_interp_step(ctx: &mut InterpCtx) -> bool {
    //            let a = ctx.counter;
    //            if ctx.counter < 100 {
    //                ctx.counter = a + 1;
    //            } else {
    //                ctx.counter = a + 2;
    //            }
    //            if ctx.counter > 110 {
    //                return true; // Exit the interpreter.
    //            }
    //            false
    //        }
    //
    //        let mut ctx = InterpCtx { counter: 0 };
    //
    //        // Run until we have compiled a trace.
    //        loop {
    //            ctx.counter = 0; // make sure we don't overflow
    //            mtt.control_point(Some(&loc), simple_interp_step, &mut ctx);
    //            if hotlocation_discriminant(&loc) == Some(HotLocationDiscriminants::Compiled) {
    //                break;
    //            }
    //        }
    //
    //        // Now reset counter and run the compiled trace until a guard fails.
    //        ctx.counter = 100;
    //        loop {
    //            if mtt.control_point(Some(&loc), simple_interp_step, &mut ctx) {
    //                break;
    //            }
    //        }
    //
    //        assert_eq!(ctx.counter, 112);
    //    }
}
