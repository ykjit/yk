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

const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;

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
    // Create a new meta-tracer instance. Arbitrarily many of these can be created, though there
    // are no guarantees as to whether they will share resources effectively or fairly.
    pub fn new() -> Self {
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

    /// Set the threshold at which `Location`'s are considered hot.
    pub fn set_hot_threshold(&self, hot_threshold: HotThreshold) {
        self.inner
            .hot_threshold
            .store(hot_threshold, Ordering::Relaxed);
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

    pub fn control_point(&self, loc: &Location, ctrlp_vars: *mut c_void) {
        match self.transition_location(loc) {
            TransitionLocation::NoAction => (),
            TransitionLocation::Execute(ctr) => {
                // FIXME: If we want to free compiled traces, we'll need to refcount (or use
                // a GC) to know if anyone's executing that trace at the moment.
                //
                // FIXME: this loop shouldn't exist. Trace stitching should be implemented in
                // the trace itself.
                // https://github.com/ykjit/yk/issues/442
                loop {
                    #[cfg(feature = "jit_state_debug")]
                    eprintln!("jit-state: enter-jit-code");
                    unsafe { &*ctr }.exec(ctrlp_vars);
                    #[cfg(feature = "jit_state_debug")]
                    eprintln!("jit-state: exit-jit-code");
                }
            }
            TransitionLocation::StartTracing(kind) => start_tracing(kind),
            TransitionLocation::StopTracing(hl) => match stop_tracing() {
                Ok(ir_trace) => self.queue_compile_job(ir_trace, hl),
                Err(_) => todo!(),
            },
        }
    }

    /// Perform the next step to `loc` in the `Location` state-machine. If `loc` moves to the
    /// Compiled state, return a pointer to a [CompiledTrace] object.
    fn transition_location(&self, loc: &Location) -> TransitionLocation {
        let mut ls = loc.load(Ordering::Relaxed);

        if ls.is_counting() {
            debug_assert!(!ls.is_locked());
            debug_assert!(!ls.is_parked());

            let count = ls.count();
            if count < self.hot_threshold() {
                // Try incrementing this location's hot count. We make no guarantees that this will
                // succeed because under contention we can end up racing with many other threads
                // and it's not worth our time to halt execution merely to have an accurate hot
                // count.
                loc.compare_exchange_weak(
                    ls,
                    ls.with_count(count + 1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .ok();
                return TransitionLocation::NoAction;
            } else {
                return THREAD_MTTHREAD.with(|mtt| {
                    if mtt.tracing.get().is_some() {
                        // This thread is already tracing another Location, so either another
                        // thread needs to trace this Location or this thread needs to wait
                        // until the current round of tracing has completed. Either way,
                        // there's no point incrementing the hot count.
                        TransitionLocation::NoAction
                    } else {
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
                                    *unsafe { new_ls.hot_location() } =
                                        HotLocation::Tracing(Some(tid));
                                    mtt.tracing.set(Some(hl_ptr as *const ()));
                                    loc.unlock();
                                    break TransitionLocation::StartTracing(self.tracing_kind());
                                }
                                Err(x) => {
                                    if x.is_locked() {
                                        // We probably raced with another thread locking this Location in order to
                                        // start tracing. It's unlikely to be worth us spending time contending
                                        // with that other thread.
                                        unsafe { Box::from_raw(hl_ptr) };
                                        break TransitionLocation::NoAction;
                                    }
                                    ls = x;
                                }
                            }
                        }
                    }
                });
            }
        } else {
            // There's no point contending with other threads, so in general we don't want to
            // continually try grabbing the lock.
            match loc.try_lock() {
                Some(x) => ls = x,
                None => {
                    // If this thread is tracing we need to grab the lock so that we can stop
                    // tracing, otherwise we return to the interpreter.
                    if THREAD_MTTHREAD.with(|mtt| mtt.tracing.get().is_none()) {
                        return TransitionLocation::NoAction;
                    }
                    match loc.lock() {
                        Ok(x) => ls = x,
                        Err(()) => {
                            // The location transitioned back to the counting state before we'd
                            // gained a lock.
                            return TransitionLocation::NoAction;
                        }
                    }
                }
            }
            let hl = unsafe { ls.hot_location() };
            let hl_ptr = hl as *mut _ as *mut ();
            match hl {
                HotLocation::Compiled(ctr) => {
                    loc.unlock();
                    return TransitionLocation::Execute(*ctr);
                }
                HotLocation::Compiling => {
                    loc.unlock();
                    return TransitionLocation::NoAction;
                }
                HotLocation::Dropped => {
                    unreachable!();
                }
                HotLocation::Tracing(opt) => {
                    // FIXME: This is the sort of hack that keeps me awake at night: we really want
                    // to return from the outer function, and to modify `hl`, but can't because
                    // we're in a closure. The integer return value allows us to perform the
                    // control flow we want.
                    let r = THREAD_MTTHREAD.with(|mtt| {
                        if let Some(other_hl_ptr) = mtt.tracing.get() {
                            // This thread is tracing something...
                            if !ptr::eq(hl_ptr, other_hl_ptr) {
                                // but not this Location.
                                loc.unlock();
                                // Should be `return TransitionLocation::NoAction`
                                1
                            } else {
                                // Should be "do nothing"
                                2
                            }
                        } else {
                            // Should be "check if a now-dead thread was tracing this location"
                            3
                        }
                    });
                    match r {
                        1 => return TransitionLocation::NoAction,
                        2 => (),
                        3 => {
                            // This thread isn't tracing anything.
                            if Arc::strong_count(&opt.as_ref().unwrap()) == 1 {
                                // Another thread was tracing this location but it's terminated.
                                // FIXME: we should probably have some sort of occasional retry
                                // heuristic rather than simply saying "never try tracing this
                                // Location again."
                                *hl = HotLocation::DontTrace;
                            }
                            loc.unlock();
                            return TransitionLocation::NoAction;
                        }
                        _ => unreachable!(),
                    }
                    // This thread is tracing this location: we must, therefore, have finished
                    // tracing the loop.
                    //
                    // We must ensure that the `Arc<ThreadId>` inside `opt` is dropped so that
                    // other threads won't think this thread has died while tracing.
                    opt.take();
                    #[cfg(feature = "jit_state_debug")]
                    eprintln!("jit-state: stop-tracing");
                    *hl = HotLocation::Compiling;
                    loc.unlock();
                    return TransitionLocation::StopTracing(hl);
                }
                HotLocation::DontTrace => {
                    loc.unlock();
                    return TransitionLocation::NoAction;
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

/// What action should a caller of `MT::transition_location` take?
#[derive(Debug, PartialEq)]
enum TransitionLocation {
    NoAction,
    Execute(*const CompiledTrace),
    StartTracing(TracingKind),
    StopTracing(*const HotLocation),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::location::HotLocationDiscriminants;
    use std::{convert::TryFrom, thread};

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
    fn basic_transitions() {
        let hot_thrsh = 5;
        let mt = MT::new();
        mt.set_hot_threshold(hot_thrsh);
        let loc = Location::new();
        for i in 0..mt.hot_threshold() {
            assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
            assert_eq!(
                loc.load(Ordering::Relaxed),
                LocationInner::new().with_count(i + 1)
            );
        }
        assert_eq!(
            mt.transition_location(&loc),
            TransitionLocation::StartTracing(mt.tracing_kind())
        );
        assert_eq!(
            hotlocation_discriminant(&loc),
            Some(HotLocationDiscriminants::Tracing)
        );
        match mt.transition_location(&loc) {
            TransitionLocation::StopTracing(tracing_ls) => {
                let ls = loc.load(Ordering::Relaxed);
                assert!(!ls.is_counting());
                assert_eq!(unsafe { ls.hot_location() } as *const _, tracing_ls);
                unsafe {
                    *(tracing_ls as *mut HotLocation) = HotLocation::Compiling;
                }
            }
            _ => unreachable!(),
        }
        assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
        assert_eq!(
            hotlocation_discriminant(&loc),
            Some(HotLocationDiscriminants::Compiling)
        );
        let ls = loc.load(Ordering::Relaxed);
        assert!(!ls.is_counting());
        *unsafe { ls.hot_location() } = HotLocation::Compiled(std::ptr::null());
        assert_eq!(
            mt.transition_location(&loc),
            TransitionLocation::Execute(std::ptr::null())
        );
    }

    #[test]
    fn threaded_threshold() {
        // Aim for a situation where there's a lot of contention.
        let num_threads = u32::try_from(num_cpus::get() * 4).unwrap();
        let hot_thrsh = num_threads.saturating_mul(10000);
        let mt = MT::new();
        mt.set_hot_threshold(hot_thrsh);
        let loc = Arc::new(Location::new());

        let mut thrs = vec![];
        for _ in 0..num_threads {
            let mt = mt.clone();
            let loc = Arc::clone(&loc);
            let t = thread::spawn(move || {
                for _ in 0..hot_thrsh / num_threads {
                    assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
                    let c1 = loc.load(Ordering::Relaxed);
                    assert!(c1.is_counting());
                    assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
                    let c2 = loc.load(Ordering::Relaxed);
                    assert!(c2.is_counting());
                    assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
                    let c3 = loc.load(Ordering::Relaxed);
                    assert!(c3.is_counting());
                    assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
                    let c4 = loc.load(Ordering::Relaxed);
                    assert!(c4.is_counting());
                    assert!(c4.count() >= c3.count());
                    assert!(c3.count() >= c2.count());
                    assert!(c2.count() >= c1.count());
                }
            });
            thrs.push(t);
        }
        for t in thrs {
            t.join().unwrap();
        }
        // Thread contention and the use of `compare_exchange_weak` means that there is absolutely
        // no guarantee about what the location's count will be at this point other than it must be
        // at or below the threshold: it could even be (although it's rather unlikely) 0!
        assert!(loc.load(Ordering::Relaxed).is_counting());
        loop {
            match mt.transition_location(&loc) {
                TransitionLocation::NoAction => (),
                TransitionLocation::StartTracing(_) => break,
                _ => unreachable!(),
            }
        }
        assert!(matches!(
            mt.transition_location(&loc),
            TransitionLocation::StopTracing(_)
        ));
        // At this point, we have nothing to meaningfully test over the `basic_transitions` test.
    }

    #[test]
    fn locations_dont_get_stuck_tracing() {
        // If a thread starts tracing a location but terminates before finishing tracing, a
        // Location can be left in the Tracing state: this test ensures that, as soon as another
        // thread notices that the original Tracing thread has died, that the Location is updated
        // to a non-tracing state.

        const THRESHOLD: HotThreshold = 5;
        let mt = MT::new();
        mt.set_hot_threshold(THRESHOLD);
        let loc = Arc::new(Location::new());

        {
            let mt = mt.clone();
            let loc = Arc::clone(&loc);
            thread::spawn(move || {
                for _ in 0..THRESHOLD {
                    assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
                }
                assert!(matches!(
                    mt.transition_location(&loc),
                    TransitionLocation::StartTracing(_)
                ));
            })
            .join()
            .unwrap();
        }

        assert_eq!(
            hotlocation_discriminant(&loc),
            Some(HotLocationDiscriminants::Tracing)
        );
        assert_eq!(mt.transition_location(&loc), TransitionLocation::NoAction);
        assert_eq!(
            hotlocation_discriminant(&loc),
            Some(HotLocationDiscriminants::DontTrace)
        );
    }
}
