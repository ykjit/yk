//! The main end-user interface to the meta-tracing system.

#[cfg(feature = "c_testing")]
use std::env;
use std::{
    cell::Cell,
    cmp,
    collections::VecDeque,
    ffi::c_void,
    marker::PhantomData,
    mem::forget,
    ptr,
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use num_cpus;
use parking_lot::{Condvar, Mutex, MutexGuard};
use parking_lot_core::SpinWait;
use std::lazy::SyncLazy;

use crate::location::{HotLocation, Location, LocationInner, ThreadIdInner};
use yktrace::{start_tracing, stop_tracing, CompiledTrace, IRTrace, TracingKind};

// The HotThreshold must be less than a machine word wide for [`Location::Location`] to do its
// pointer tagging thing. We therefore choose a type which makes this statically clear to
// users rather than having them try to use (say) u64::max() on a 64 bit machine and get a run-time
// error.
#[cfg(target_pointer_width = "64")]
pub type HotThreshold = u32;
#[cfg(target_pointer_width = "64")]
type AtomicHotThreshold = AtomicU32;

// FIXME: just for parity with existing tests for now.
const DEFAULT_HOT_THRESHOLD: HotThreshold = 0;

static GLOBAL_MT: SyncLazy<MT> = SyncLazy::new(|| MT::new());
thread_local! {static THREAD_MTTHREAD: MTThread = MTThread::new();}

#[cfg(feature = "c_testing")]
static SERIALISE_COMPILATION: SyncLazy<bool> =
    SyncLazy::new(|| &env::var("YKD_SERIALISE_COMPILATION").unwrap_or("0".to_owned()) == "1");

#[derive(Clone)]
/// A meta-tracer. Note that this is conceptually a "front-end" to the actual meta-tracer akin to
/// an `Rc`: this struct can be freely `clone()`d without duplicating the underlying meta-tracer.
pub struct MT {
    inner: Arc<MTInner>,
}

impl MT {
    /// Return a reference to the global [`MT`] instance: at any point, there is at most one of
    /// these per process and an instance will be created if it does not already exist.
    pub fn global() -> &'static Self {
        &*GLOBAL_MT
    }

    fn new() -> Self {
        let inner = MTInner {
            hot_threshold: AtomicHotThreshold::new(DEFAULT_HOT_THRESHOLD),
            job_queue: (Condvar::new(), Mutex::new(VecDeque::new())),
            max_worker_threads: AtomicUsize::new(cmp::max(1, num_cpus::get() - 1)),
            active_worker_threads: AtomicUsize::new(0),
            tracing_kind: TracingKind::default(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Return this `MT` instance's current hot threshold. Notice that this value can be changed by
    /// other threads and is thus potentially stale as soon as it is read.
    pub fn hot_threshold(&self) -> HotThreshold {
        self.inner.hot_threshold.load(Ordering::Relaxed)
    }

    /// Return this meta-tracer's maximum number of worker threads. Notice that this value can be
    /// changed by other threads and is thus potentially stale as soon as it is read.
    pub fn max_worker_threads(&self) -> usize {
        self.inner.max_worker_threads.load(Ordering::Relaxed)
    }

    /// Return the kind of tracing that this meta-tracer is using. Notice that this value can be
    /// changed by other threads and is thus potentially stale as soon as it is read.
    pub fn tracing_kind(&self) -> TracingKind {
        self.inner.tracing_kind
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

    pub fn transition_location(loc: &Location, ctrlp_vars: *mut c_void) {
        let mt = MT::global();
        THREAD_MTTHREAD.with(|mtt| {
            mt.do_transition_location(mtt, loc, ctrlp_vars);
        });
    }

    fn do_transition_location(&self, mtt: &MTThread, loc: &Location, ctrlp_vars: *mut c_void) {
        let mut ls = loc.load(Ordering::Relaxed);

        if ls.is_counting() {
            debug_assert!(!ls.is_locked());
            debug_assert!(!ls.is_parked());

            let count = ls.count();
            if count < self.hot_threshold() {
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
                if mtt.tracing.get().is_some() {
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
                let new_ls = LocationInner::new().with_hotlocation(hl_ptr).with_lock();
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
                            let tid = Arc::clone(&mtt.tid);
                            #[cfg(feature = "jit_state_debug")]
                            eprintln!("jit-state: start-tracing");
                            start_tracing(self.tracing_kind());
                            *unsafe { new_ls.hot_location() } = HotLocation::Tracing(Some(tid));
                            mtt.tracing.set(Some(hl_ptr as *const ()));
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
                Some(x) => ls = x,
                None => {
                    // If this thread is tracing we need to grab the lock so that we can stop
                    // tracing, otherwise we return to the interpreter.
                    if mtt.tracing.get().is_none() {
                        return;
                    }
                    match loc.lock() {
                        Ok(x) => ls = x,
                        Err(()) => {
                            // The location transitioned back to the counting state before we'd
                            // gained a lock.
                            return;
                        }
                    }
                }
            }
            let hl = unsafe { ls.hot_location() };
            let hl_ptr = hl as *mut _ as *mut ();
            match hl {
                HotLocation::Compiled(tr) => {
                    loc.unlock();
                    // FIXME: If we want to free compiled traces, we'll need to refcount (or use
                    // a GC) to know if anyone's executing that trace at the moment.
                    //
                    // FIXME: this loop shouldn't exist. Trace stitching should be implemented in
                    // the trace itself.
                    // https://github.com/ykjit/yk/issues/442
                    loop {
                        #[cfg(feature = "jit_state_debug")]
                        eprintln!("jit-state: enter-jit-code");
                        unsafe { &**tr }.exec(ctrlp_vars);
                        #[cfg(feature = "jit_state_debug")]
                        eprintln!("jit-state: exit-jit-code");
                    }
                }
                HotLocation::Compiling => {
                    loc.unlock();
                    return;
                }
                HotLocation::Dropped => {
                    unreachable!();
                }
                HotLocation::Tracing(opt) => {
                    match mtt.tracing.get() {
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
                            if Arc::strong_count(&opt.as_ref().unwrap()) == 1 {
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
                    // tracing the loop.
                    //
                    // We must ensure that the `Arc<ThreadId>` inside `opt` is dropped so that
                    // other threads won't think this thread has died while tracing.
                    opt.take();
                    #[cfg(feature = "jit_state_debug")]
                    eprintln!("jit-state: stop-tracing");
                    match stop_tracing() {
                        Ok(ir_trace) => {
                            *hl = HotLocation::Compiling;
                            loc.unlock();
                            self.queue_compile_job(ir_trace, hl);
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
    fn queue_compile_job(&self, trace: IRTrace, hl_ptr: *const HotLocation) {
        let hl_ptr = hl_ptr as usize;

        let do_compile = move || {
            let code_ptr = trace.compile();
            let ct = Box::new(CompiledTrace::new(code_ptr));
            // We can't lock a `HotLocation` directly as the `lock` method is on `Location`. We
            // thus need to create a "fake" / "temporary" `Location` so that we can `lock` it: note
            // that we *must not* `drop` this temporary `Location`, hence the later call to
            // `forget`.
            let tmp_ls = LocationInner::new().with_hotlocation(hl_ptr as *mut HotLocation);
            let tmp_loc = unsafe { Location::from_location_inner(tmp_ls) };
            tmp_loc.lock().unwrap();
            let hl = unsafe { tmp_ls.hot_location() };
            if let HotLocation::Compiling = hl {
                // FIXME: although we've now put the compiled trace into the `HotLocation`, there's
                // no guarantee that the `Location` for which we're compiling will ever be executed
                // again. In such a case, the memory has, in essence, leaked.
                *hl = HotLocation::Compiled(Box::into_raw(ct));
            } else if let HotLocation::Dropped = hl {
                // The Location pointing to this HotLocation was dropped. There's nothing we can do
                // with the compiled trace, so we let it it be implicitly dropped.
            } else {
                unreachable!();
            }
            tmp_loc.unlock();
            forget(tmp_loc);
        };

        #[cfg(feature = "c_testing")]
        if *SERIALISE_COMPILATION {
            do_compile();
            return;
        }

        self.queue_job(Box::new(do_compile));
    }
}

/// The innards of a meta-tracer.
struct MTInner {
    hot_threshold: AtomicHotThreshold,
    /// The ordered queue of compilation worker functions.
    job_queue: (Condvar, Mutex<VecDeque<Box<dyn FnOnce() + Send>>>),
    /// The hard cap on the number of worker threads.
    max_worker_threads: AtomicUsize,
    /// How many worker threads are currently running. Note that this may temporarily be `>`
    /// [`max_worker_threads`].
    active_worker_threads: AtomicUsize,
    tracing_kind: TracingKind,
}

/// Meta-tracer per-thread state. Note that this struct is neither `Send` nor `Sync`: it can only
/// be accessed from within a single thread.
pub struct MTThread {
    /// An Arc whose pointer address uniquely identifies this thread. When a Location is traced,
    /// this Arc's strong count will be incremented. If, after this thread drops, this Arc's strong
    /// count remains > 0, it means that it was in the process of tracing a loop, implying that
    /// there is (or, at least, was at some point) a Location stuck in PHASE_TRACING.
    tid: Arc<ThreadIdInner>,
    /// If this thread is tracing, store a pointer to the `HotLocation`: this allows us to
    /// differentiate which thread is actually tracing the location.
    tracing: Cell<Option<*const ()>>,
    // Raw pointers are neither send nor sync.
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl MTThread {
    fn new() -> Self {
        MTThread {
            tid: Arc::new(ThreadIdInner),
            tracing: Cell::new(None),
            _dont_send_or_sync_me: PhantomData,
        }
    }
}
