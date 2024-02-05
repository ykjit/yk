//! The main end-user interface to the meta-tracing system.

#[cfg(feature = "yk_testing")]
use std::env;
use std::{
    cell::RefCell,
    cmp,
    collections::VecDeque,
    error::Error,
    ffi::c_void,
    marker::PhantomData,
    mem,
    sync::{
        atomic::{AtomicU16, AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use parking_lot::{Condvar, Mutex, MutexGuard};
use parking_lot_core::SpinWait;
#[cfg(feature = "yk_jitstate_debug")]
use std::sync::LazyLock;

#[cfg(feature = "yk_jitstate_debug")]
use crate::print_jit_state;
use crate::{
    compile::{default_compiler, CompilationError, CompiledTrace, Compiler, GuardId},
    location::{HotLocation, HotLocationKind, Location, TraceFailed},
    trace::{default_tracer, TraceCollector, TraceIterator, Tracer},
    ykstats::{TimingState, YkStats},
};
use yktracec::promote;

// The HotThreshold must be less than a machine word wide for [`Location::Location`] to do its
// pointer tagging thing. We therefore choose a type which makes this statically clear to
// users rather than having them try to use (say) u64::max() on a 64 bit machine and get a run-time
// error.
#[cfg(target_pointer_width = "64")]
pub type HotThreshold = u32;
#[cfg(target_pointer_width = "64")]
type AtomicHotThreshold = AtomicU32;

pub type TraceFailureThreshold = u16;
pub type AtomicTraceFailureThreshold = AtomicU16;

/// How many blocks long can a trace be before we give up trying to compile it? Note that the
/// slower our compiler, the lower this will have to be in order to give the perception of
/// reasonable performance.
/// FIXME: needs to be configurable.
pub(crate) const DEFAULT_TRACE_TOO_LONG: usize = 5000;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;
const DEFAULT_SIDETRACE_THRESHOLD: HotThreshold = 5;
const DEFAULT_TRACE_FAILURE_THRESHOLD: TraceFailureThreshold = 5;

thread_local! {static THREAD_MTTHREAD: MTThread = MTThread::new();}

#[cfg(feature = "yk_testing")]
static SERIALISE_COMPILATION: LazyLock<bool> = LazyLock::new(|| {
    &env::var("YKD_SERIALISE_COMPILATION").unwrap_or_else(|_| "0".to_owned()) == "1"
});

/// Stores information required for compiling a side-trace. Passed down from a (parent) trace
/// during deoptimisation.
#[derive(Debug, Copy, Clone)]
pub(crate) struct SideTraceInfo {
    pub callstack: *const c_void,
    pub aotvalsptr: *const c_void,
    pub aotvalslen: usize,
    pub guardid: GuardId,
}

unsafe impl Send for SideTraceInfo {}

/// A meta-tracer. Note that this is conceptually a "front-end" to the actual meta-tracer akin to
/// an `Rc`: this struct can be freely `clone()`d without duplicating the underlying meta-tracer.
pub struct MT {
    hot_threshold: AtomicHotThreshold,
    sidetrace_threshold: AtomicHotThreshold,
    trace_failure_threshold: AtomicTraceFailureThreshold,
    /// The ordered queue of compilation worker functions.
    job_queue: Arc<(Condvar, Mutex<VecDeque<Box<dyn FnOnce() + Send>>>)>,
    /// The hard cap on the number of worker threads.
    max_worker_threads: AtomicUsize,
    /// How many worker threads are currently running. Note that this may temporarily be `>`
    /// [`max_worker_threads`].
    active_worker_threads: AtomicUsize,
    #[cfg(yk_llvm_sync_hack)]
    /// A temporary hack which keeps track of how many compile (etc.) jobs are active, because LLVM
    /// dies horribly if the main thread exits before those jobs are finished. This then marries up
    /// with Self::.llvm_sync_hack().
    active_worker_jobs: AtomicUsize,
    /// The [Tracer] that should be used for creating future traces. Note that this might not be
    /// the same as the tracer(s) used to create past traces.
    tracer: Mutex<Arc<dyn Tracer>>,
    /// The [Compiler] that will be used for compiling future `IRTrace`s. Note that this might not
    /// be the same as the compiler(s) used to compile past `IRTrace`s.
    compiler: Mutex<Arc<dyn Compiler>>,
    pub(crate) stats: YkStats,
}

impl std::fmt::Debug for MT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MT")
    }
}

use crate::frame::load_aot_stackmaps;

impl MT {
    // Create a new meta-tracer instance. Arbitrarily many of these can be created, though there
    // are no guarantees as to whether they will share resources effectively or fairly.
    pub fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        load_aot_stackmaps();
        Ok(Arc::new(Self {
            hot_threshold: AtomicHotThreshold::new(DEFAULT_HOT_THRESHOLD),
            sidetrace_threshold: AtomicHotThreshold::new(DEFAULT_SIDETRACE_THRESHOLD),
            trace_failure_threshold: AtomicTraceFailureThreshold::new(
                DEFAULT_TRACE_FAILURE_THRESHOLD,
            ),
            job_queue: Arc::new((Condvar::new(), Mutex::new(VecDeque::new()))),
            max_worker_threads: AtomicUsize::new(cmp::max(1, num_cpus::get() - 1)),
            active_worker_threads: AtomicUsize::new(0),
            #[cfg(yk_llvm_sync_hack)]
            active_worker_jobs: AtomicUsize::new(0),
            tracer: Mutex::new(default_tracer()?),
            compiler: Mutex::new(default_compiler()?),
            stats: YkStats::new(),
        }))
    }

    /// Return this `MT` instance's current hot threshold. Notice that this value can be changed by
    /// other threads and is thus potentially stale as soon as it is read.
    pub fn hot_threshold(self: &Arc<Self>) -> HotThreshold {
        self.hot_threshold.load(Ordering::Relaxed)
    }

    /// Set the threshold at which `Location`'s are considered hot.
    pub fn set_hot_threshold(self: &Arc<Self>, hot_threshold: HotThreshold) {
        self.hot_threshold.store(hot_threshold, Ordering::Relaxed);
    }

    /// Return this `MT` instance's current side-trace threshold. Notice that this value can be
    /// changed by other threads and is thus potentially stale as soon as it is read.
    pub fn sidetrace_threshold(self: &Arc<Self>) -> HotThreshold {
        self.sidetrace_threshold.load(Ordering::Relaxed)
    }

    /// Set the threshold at which guard failures are considered hot and side-tracing should start.
    pub fn set_sidetrace_threshold(self: &Arc<Self>, hot_threshold: HotThreshold) {
        self.sidetrace_threshold
            .store(hot_threshold, Ordering::Relaxed);
    }

    /// Return this `MT` instance's current trace failure threshold. Notice that this value can be
    /// changed by other threads and is thus potentially stale as soon as it is read.
    pub fn trace_failure_threshold(self: &Arc<Self>) -> TraceFailureThreshold {
        self.trace_failure_threshold.load(Ordering::Relaxed)
    }

    /// Set the threshold at which a `Location` from which tracing has failed multiple times is
    /// marked as "do not try tracing again".
    pub fn set_trace_failure_threshold(
        self: &Arc<Self>,
        trace_failure_threshold: TraceFailureThreshold,
    ) {
        if trace_failure_threshold < 1 {
            panic!("Trace failure threshold must be >= 1.");
        }
        self.trace_failure_threshold
            .store(trace_failure_threshold, Ordering::Relaxed);
    }

    /// Return this meta-tracer's maximum number of worker threads. Notice that this value can be
    /// changed by other threads and is thus potentially stale as soon as it is read.
    pub fn max_worker_threads(self: &Arc<Self>) -> usize {
        self.max_worker_threads.load(Ordering::Relaxed)
    }

    /// Queue `job` to be run on a worker thread.
    fn queue_job(self: &Arc<Self>, job: Box<dyn FnOnce() + Send>) {
        // We have a very simple model of worker threads. Each time a job is queued, we spin up a
        // new worker thread iff we aren't already running the maximum number of worker threads.
        // Once started, a worker thread never dies, waiting endlessly for work.

        #[cfg(yk_llvm_sync_hack)]
        self.active_worker_jobs.fetch_add(1, Ordering::Relaxed);

        let (cv, mtx) = &*self.job_queue;
        mtx.lock().push_back(job);
        cv.notify_one();

        let max_jobs = self.max_worker_threads.load(Ordering::Relaxed);
        if self.active_worker_threads.load(Ordering::Relaxed) < max_jobs {
            // At the point of the `load` on the previous line, we weren't running the maximum
            // number of worker threads. There is now a possible race condition where multiple
            // threads calling `queue_job` could try creating multiple worker threads and push us
            // over the maximum worker thread limit.
            if self.active_worker_threads.fetch_add(1, Ordering::Relaxed) > max_jobs {
                // Another thread(s) is also spinning up another worker thread and they won the
                // race.
                self.active_worker_threads.fetch_sub(1, Ordering::Relaxed);
                return;
            }

            let mt = Arc::clone(self);
            let jq = Arc::clone(&self.job_queue);
            thread::spawn(move || {
                mt.stats.timing_state(TimingState::None);
                let (cv, mtx) = &*jq;
                let mut lock = mtx.lock();
                loop {
                    match lock.pop_front() {
                        Some(x) => {
                            MutexGuard::unlocked(&mut lock, x);
                            #[cfg(yk_llvm_sync_hack)]
                            mt.active_worker_jobs.fetch_sub(1, Ordering::Relaxed);
                        }
                        None => cv.wait(&mut lock),
                    }
                }
            });
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn control_point(
        self: &Arc<Self>,
        loc: &Location,
        ctrlp_vars: *mut c_void,
        frameaddr: *mut c_void,
    ) {
        match self.transition_control_point(loc) {
            TransitionControlPoint::NoAction => (),
            TransitionControlPoint::Execute(ctr) => {
                #[cfg(feature = "yk_jitstate_debug")]
                print_jit_state("enter-jit-code");
                self.stats.trace_executed();
                self.stats.timing_state(TimingState::JitExecuting);

                unsafe {
                    #[cfg(feature = "yk_testing")]
                    assert_ne!(ctr.entry() as *const (), std::ptr::null());
                    let f = mem::transmute::<
                        _,
                        unsafe extern "C" fn(*mut c_void, *const CompiledTrace, *const c_void) -> !,
                    >(ctr.entry());
                    // FIXME: Calling this function overwrites the current (Rust) function frame,
                    // rather than unwinding it. https://github.com/ykjit/yk/issues/778.
                    // The `Arc<CompiledTrace>` passed into the trace here will be safely dropped
                    // upon deoptimisation, which is the only way to exit a trace.
                    f(ctrlp_vars, Arc::into_raw(ctr), frameaddr);
                }
            }
            TransitionControlPoint::StartTracing => {
                #[cfg(feature = "yk_jitstate_debug")]
                print_jit_state("start-tracing");
                let tracer = {
                    let lk = self.tracer.lock();
                    Arc::clone(&*lk)
                };
                match Arc::clone(&tracer).start_collector() {
                    Ok(tt) => THREAD_MTTHREAD.with(|mtt| {
                        promote::thread_record_enable(true);
                        *mtt.thread_tracer.borrow_mut() = Some(tt);
                    }),
                    Err(e) => todo!("{e:?}"),
                }
            }
            TransitionControlPoint::StopTracing(hl_arc) => {
                promote::thread_record_enable(false);
                // Assuming no bugs elsewhere, the `unwrap` cannot fail, because `StartTracing`
                // will have put a `Some` in the `Rc`.
                let thrdtrcr = THREAD_MTTHREAD.with(|mtt| mtt.thread_tracer.take().unwrap());
                match thrdtrcr.stop_collector() {
                    Ok(utrace) => {
                        #[cfg(feature = "yk_jitstate_debug")]
                        print_jit_state("stop-tracing");
                        self.queue_compile_job(utrace, hl_arc, None);
                    }
                    Err(_e) => {
                        #[cfg(feature = "yk_jitstate_debug")]
                        print_jit_state("stop-tracing-aborted");
                    }
                }
            }
            TransitionControlPoint::StopSideTracing(hl_arc, sti, parent) => {
                promote::thread_record_enable(false);
                // Assuming no bugs elsewhere, the `unwrap` cannot fail, because `StartTracing`
                // will have put a `Some` in the `Rc`.
                let thrdtrcr = THREAD_MTTHREAD.with(|mtt| mtt.thread_tracer.take().unwrap());
                match thrdtrcr.stop_collector() {
                    Ok(utrace) => {
                        #[cfg(feature = "yk_jitstate_debug")]
                        print_jit_state("stop-side-tracing");
                        self.queue_compile_job(utrace, hl_arc, Some((sti, parent)));
                    }
                    Err(_e) => {
                        #[cfg(feature = "yk_jitstate_debug")]
                        print_jit_state("stop-side-tracing-aborted");
                    }
                }
            }
        }
    }

    /// Perform the next step to `loc` in the `Location` state-machine for a control point. If
    /// `loc` moves to the Compiled state, return a pointer to a [CompiledTrace] object.
    fn transition_control_point(self: &Arc<Self>, loc: &Location) -> TransitionControlPoint {
        THREAD_MTTHREAD.with(|mtt| {
            let am_tracing = mtt.tracing.borrow().is_some();
            match loc.hot_location() {
                Some(hl) => {
                    // If this thread is tracing something, we *must* grab the [HotLocation] lock,
                    // because we need to know for sure if `loc` is the point at which we should stop
                    // tracing. If this thread is not tracing anything, however, it's not worth
                    // contending too much with other threads: we try moderately hard to grab the lock,
                    // but we don't want to park this thread.
                    let mut lk = if !am_tracing {
                        // This thread isn't tracing anything, so we try for a little while to grab the
                        // lock, before giving up and falling back to the interpreter. In general, we
                        // expect that we'll grab the lock rather quickly. However, there is one nasty
                        // use-case, which is when an army of threads all start executing the same
                        // piece of tiny code and end up thrashing away at a single Location,
                        // particularly when it's in a non-Compiled state: we can end up contending
                        // horribly for a single lock, and not making much progress. In that case, it's
                        // probably better to let some threads fall back to the interpreter for another
                        // iteration, and hopefully allow them to get sufficiently out-of-sync that
                        // they no longer contend on this one lock as much.
                        let mut sw = SpinWait::new();
                        loop {
                            if let Some(lk) = hl.try_lock() {
                                break lk;
                            }
                            if !sw.spin() {
                                return TransitionControlPoint::NoAction;
                            }
                        }
                    } else {
                        // This thread is tracing something, so we must grab the lock.
                        hl.lock()
                    };

                    match lk.kind {
                        HotLocationKind::Compiled(ref ctr) => {
                            if am_tracing {
                                // This thread is tracing something, so bail out as quickly as possible
                                TransitionControlPoint::NoAction
                            } else {
                                TransitionControlPoint::Execute(Arc::clone(ctr))
                            }
                        }
                        HotLocationKind::Compiling => TransitionControlPoint::NoAction,
                        HotLocationKind::Tracing => {
                            let hl = loc.hot_location_arc_clone().unwrap();
                            let mut thread_hl_out = mtt.tracing.borrow_mut();
                            if let Some(ref thread_hl_in) = *thread_hl_out {
                                // This thread is tracing something...
                                if !Arc::ptr_eq(thread_hl_in, &hl) {
                                    // ...but not this Location.
                                    TransitionControlPoint::NoAction
                                } else {
                                    // ...and it's this location: we have therefore finished tracing the loop.
                                    *thread_hl_out = None;
                                    lk.kind = HotLocationKind::Compiling;
                                    TransitionControlPoint::StopTracing(hl)
                                }
                            } else {
                                // This thread isn't tracing anything. Note that because we called
                                // `hot_location_arc_clone` above, the strong count of an `Arc`
                                // that's no longer being used by that thread will be 2.
                                if Arc::strong_count(&hl) == 2 {
                                    // Another thread was tracing this location but it's terminated.
                                    self.stats.trace_collected_err();
                                    match lk.trace_failed(self) {
                                        TraceFailed::KeepTrying => {
                                            *thread_hl_out = Some(hl);
                                            TransitionControlPoint::StartTracing
                                        }
                                        TraceFailed::DontTrace => TransitionControlPoint::NoAction,
                                    }
                                } else {
                                    // Another thread is tracing this location.
                                    TransitionControlPoint::NoAction
                                }
                            }
                        }
                        HotLocationKind::SideTracing(ref ctr, sti, ref parent) => {
                            let hl = loc.hot_location_arc_clone().unwrap();
                            let mut thread_hl_out = mtt.tracing.borrow_mut();
                            if let Some(ref thread_hl_in) = *thread_hl_out {
                                // This thread is tracing something...
                                if !Arc::ptr_eq(thread_hl_in, &hl) {
                                    // ...but not this Location.
                                    TransitionControlPoint::Execute(Arc::clone(ctr))
                                } else {
                                    // ...and it's this location: we have therefore finished tracing the loop.
                                    *thread_hl_out = None;
                                    let nparent = Arc::clone(parent);
                                    lk.kind = HotLocationKind::Compiled(Arc::clone(ctr));
                                    TransitionControlPoint::StopSideTracing(hl, sti, nparent)
                                }
                            } else {
                                // This thread isn't tracing anything.
                                TransitionControlPoint::Execute(Arc::clone(ctr))
                            }
                        }
                        HotLocationKind::DontTrace => TransitionControlPoint::NoAction,
                    }
                }
                None => {
                    if am_tracing {
                        // This thread is tracing something, so bail out as quickly as possible
                        return TransitionControlPoint::NoAction;
                    }
                    match loc.count() {
                        Some(x) => {
                            if x < self.hot_threshold() {
                                loc.count_set(x, x + 1);
                                TransitionControlPoint::NoAction
                            } else {
                                let hl = HotLocation {
                                    kind: HotLocationKind::Tracing,
                                    trace_failure: 0,
                                };
                                if let Some(hl) = loc.count_to_hot_location(x, hl) {
                                    debug_assert!(mtt.tracing.borrow().is_none());
                                    *mtt.tracing.borrow_mut() = Some(hl);
                                    TransitionControlPoint::StartTracing
                                } else {
                                    // We raced with another thread which has started tracing this
                                    // location. We leave it to do the tracing.
                                    TransitionControlPoint::NoAction
                                }
                            }
                        }
                        None => {
                            // `loc` is being updated by another thread and we've caught it in the
                            // middle of that. We could spin but we might as well let the other thread
                            // do its thing and go around the interpreter again.
                            TransitionControlPoint::NoAction
                        }
                    }
                }
            }
        })
    }

    /// Perform the next step to `loc` in the `Location` state-machine for a guard failure.
    pub(crate) fn transition_guard_failure(
        self: &Arc<Self>,
        hl: Arc<Mutex<HotLocation>>,
        sti: SideTraceInfo,
        parent: Arc<CompiledTrace>,
    ) -> TransitionGuardFailure {
        THREAD_MTTHREAD.with(|mtt| {
            // This thread should not be tracing anything.
            debug_assert!(!mtt.tracing.borrow().is_some());
            let mut lk = hl.lock();
            if let HotLocationKind::Compiled(ref ctr) = lk.kind {
                *mtt.tracing.borrow_mut() = Some(Arc::clone(&hl));
                lk.kind = HotLocationKind::SideTracing(Arc::clone(ctr), sti, parent);
                TransitionGuardFailure::StartSideTracing
            } else {
                TransitionGuardFailure::NoAction
            }
        })
    }

    /// Add a compilation job for to the global work queue:
    ///   * `utrace` is the trace to be compiled.
    ///   * `hl_arc` is the [HotLocation] this compilation job is related to.
    ///   * `sidetrace`, if not `None`, specifies that this is a side-trace compilation job.
    ///     The `Arc<CompiledTrace>` is the parent [CompiledTrace] for the side-trace. Because
    ///     side-traces can nest, this may or may not be the same [CompiledTrace] as contained
    ///     in the `hl_arc`.
    fn queue_compile_job(
        self: &Arc<Self>,
        trace_iter: Box<dyn TraceIterator>,
        hl_arc: Arc<Mutex<HotLocation>>,
        sidetrace: Option<(SideTraceInfo, Arc<CompiledTrace>)>,
    ) {
        self.stats.trace_collected_ok();
        let mt = Arc::clone(self);
        let do_compile = move || {
            mt.stats.timing_state(TimingState::TraceMapping);
            let irtrace = trace_iter.collect::<Vec<_>>();
            debug_assert!(
                sidetrace.is_none() || matches!(hl_arc.lock().kind, HotLocationKind::Compiled(_))
            );
            mt.stats.timing_state(TimingState::None);
            let compiler = {
                let lk = mt.compiler.lock();
                Arc::clone(&*lk)
            };
            mt.stats.timing_state(TimingState::Compiling);
            let guardid = sidetrace.as_ref().map(|x| x.0.guardid);
            match compiler.compile(
                Arc::clone(&mt),
                irtrace,
                sidetrace.as_ref().map(|x| x.0),
                Arc::clone(&hl_arc),
            ) {
                Ok(ct) => {
                    let mut hl = hl_arc.lock();
                    match &hl.kind {
                        HotLocationKind::Compiled(_) => {
                            // The `unwrap`s cannot fail because of the condition contained
                            // in the `debug_assert` above: if `sidetrace` is not-`None`
                            // then `hl_arc.kind` is `Compiled`.
                            let ctr = sidetrace.map(|x| x.1).unwrap();
                            let guard = ctr.guard(guardid.unwrap());
                            guard.setct(Arc::new(ct));
                        }
                        _ => {
                            hl.kind = HotLocationKind::Compiled(Arc::new(ct));
                        }
                    }
                    mt.stats.trace_compiled_ok();
                }
                Err(CompilationError::Temporary(_e)) => {
                    mt.stats.trace_compiled_err();
                    hl_arc.lock().trace_failed(&mt);
                    #[cfg(feature = "yk_jitstate_debug")]
                    print_jit_state(&format!("trace-compilation-aborted<reason=\"{}\">", _e));
                }
                Err(CompilationError::Unrecoverable(e)) => panic!("{}", e),
            }

            mt.stats.timing_state(TimingState::None);
        };

        #[cfg(feature = "yk_testing")]
        if *SERIALISE_COMPILATION {
            do_compile();
            return;
        }

        self.queue_job(Box::new(do_compile));
    }

    /// Start recording a side trace for a guard that failed while executing JIT compiled code in
    /// `hl`.
    pub(crate) fn side_trace(
        self: &Arc<Self>,
        hl: Arc<Mutex<HotLocation>>,
        sti: SideTraceInfo,
        parent: Arc<CompiledTrace>,
    ) {
        match self.transition_guard_failure(hl, sti, parent) {
            TransitionGuardFailure::NoAction => todo!(),
            TransitionGuardFailure::StartSideTracing => {
                #[cfg(feature = "yk_jitstate_debug")]
                print_jit_state("start-side-tracing");
                let tracer = {
                    let lk = self.tracer.lock();
                    Arc::clone(&*lk)
                };
                match Arc::clone(&tracer).start_collector() {
                    Ok(tt) => THREAD_MTTHREAD.with(|mtt| {
                        promote::thread_record_enable(true);
                        *mtt.thread_tracer.borrow_mut() = Some(tt);
                    }),
                    Err(e) => todo!("{e:?}"),
                }
            }
        }
    }

    #[cfg(yk_llvm_sync_hack)]
    pub fn llvm_sync_hack(&self) {
        while self.active_worker_jobs.load(Ordering::Relaxed) != 0 {
            // Spin, but not too much.
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

impl Drop for MT {
    fn drop(&mut self) {
        self.stats.timing_state(TimingState::None);
    }
}

/// Meta-tracer per-thread state. Note that this struct is neither `Send` nor `Sync`: it can only
/// be accessed from within a single thread.
pub(crate) struct MTThread {
    /// Is this thread currently tracing something? If so, this will be a `Some<...>`. This allows
    /// another thread to tell whether the thread that started tracing a [Location] is still alive
    /// or not by inspecting its strong count (if the strong count is equal to 1 then the thread
    /// died while tracing). Note that this relies on thread local storage dropping the [MTThread]
    /// instance and (by implication) dropping the [Arc] and decrementing its strong count.
    /// Unfortunately, there is no guarantee that thread local storage will be dropped when a
    /// thread dies (and there is also significant platform variation in regard to dropping thread
    /// locals), so this mechanism can't be fully relied upon: however, we can't monitor thread
    /// death in any other reasonable way, so this will have to do.
    tracing: RefCell<Option<Arc<Mutex<HotLocation>>>>,
    /// When tracing is active, this will be `RefCell<Some(...)>`; when tracing is inactive
    /// `RefCell<None>`. We need to keep track of the [Tracer] used to start the [ThreadTracer], as
    /// trace mapping requires a reference to the [Tracer].
    thread_tracer: RefCell<Option<Box<dyn TraceCollector>>>,
    // Raw pointers are neither send nor sync.
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl MTThread {
    fn new() -> Self {
        MTThread {
            tracing: RefCell::new(None),
            thread_tracer: RefCell::new(None),
            _dont_send_or_sync_me: PhantomData,
        }
    }
}

/// What action should a caller of [MT::transition_control_point] take?
#[derive(Debug)]
enum TransitionControlPoint {
    NoAction,
    Execute(Arc<CompiledTrace>),
    StartTracing,
    StopTracing(Arc<Mutex<HotLocation>>),
    StopSideTracing(Arc<Mutex<HotLocation>>, SideTraceInfo, Arc<CompiledTrace>),
}

/// What action should a caller of [MT::transition_guard_failure] take?
#[derive(Debug)]
pub(crate) enum TransitionGuardFailure {
    NoAction,
    StartSideTracing,
}

#[cfg(test)]
impl PartialEq for TransitionControlPoint {
    fn eq(&self, other: &Self) -> bool {
        // We only implement enough of the equality function for the tests we have.
        match (self, other) {
            (TransitionControlPoint::NoAction, TransitionControlPoint::NoAction) => true,
            (TransitionControlPoint::Execute(p1), TransitionControlPoint::Execute(p2)) => {
                std::ptr::eq(p1, p2)
            }
            (TransitionControlPoint::StartTracing, TransitionControlPoint::StartTracing) => true,
            (x, y) => todo!("{:?} {:?}", x, y),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use crate::location::HotLocationKind;
    use std::{convert::TryFrom, hint::black_box, sync::atomic::AtomicU64, thread};
    use test::bench::Bencher;

    #[test]
    fn basic_transitions() {
        let hot_thrsh = 5;
        let mt = MT::new().unwrap();
        mt.set_hot_threshold(hot_thrsh);
        let loc = Location::new();
        for i in 0..mt.hot_threshold() {
            assert_eq!(
                mt.transition_control_point(&loc),
                TransitionControlPoint::NoAction
            );
            assert_eq!(loc.count(), Some(i + 1));
        }
        assert_eq!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::StartTracing
        );
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        match mt.transition_control_point(&loc) {
            TransitionControlPoint::StopTracing(_) => {
                assert!(matches!(
                    loc.hot_location().unwrap().lock().kind,
                    HotLocationKind::Compiling
                ));
                loc.hot_location().unwrap().lock().kind =
                    HotLocationKind::Compiled(Arc::new(CompiledTrace::new_testing()));
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::Execute(_)
        ));
        let sti = SideTraceInfo {
            callstack: std::ptr::null(),
            aotvalsptr: std::ptr::null(),
            aotvalslen: 0,
            guardid: GuardId::illegal(),
        };
        assert!(matches!(
            mt.transition_guard_failure(
                loc.hot_location_arc_clone().unwrap(),
                sti,
                Arc::new(CompiledTrace::new_testing()),
            ),
            TransitionGuardFailure::StartSideTracing
        ));
        match mt.transition_control_point(&loc) {
            TransitionControlPoint::StopSideTracing(_, _, _) => {
                assert!(matches!(
                    loc.hot_location().unwrap().lock().kind,
                    HotLocationKind::Compiled(_)
                ));
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::Execute(_)
        ));
    }

    #[test]
    fn threaded_threshold() {
        // Aim for a situation where there's a lot of contention.
        let num_threads = u32::try_from(num_cpus::get() * 4).unwrap();
        let hot_thrsh = num_threads.saturating_mul(100000);
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(hot_thrsh);
        let loc = Arc::new(Location::new());

        let mut thrs = vec![];
        for _ in 0..num_threads {
            let mt = Arc::clone(&mt);
            let loc = Arc::clone(&loc);
            let t = thread::spawn(move || {
                // The "*4" is the number of times we call `transition_location` in the loop: we
                // need to make sure that this loop cannot tip the Location over the threshold,
                // otherwise tracing will start, and the assertions will fail.
                for _ in 0..hot_thrsh / (num_threads * 4) {
                    assert_eq!(
                        mt.transition_control_point(&loc),
                        TransitionControlPoint::NoAction
                    );
                    let c1 = loc.count();
                    assert!(c1.is_some());
                    assert_eq!(
                        mt.transition_control_point(&loc),
                        TransitionControlPoint::NoAction
                    );
                    let c2 = loc.count();
                    assert!(c2.is_some());
                    assert_eq!(
                        mt.transition_control_point(&loc),
                        TransitionControlPoint::NoAction
                    );
                    let c3 = loc.count();
                    assert!(c3.is_some());
                    assert_eq!(
                        mt.transition_control_point(&loc),
                        TransitionControlPoint::NoAction
                    );
                    let c4 = loc.count();
                    assert!(c4.is_some());
                    assert!(c4.unwrap() >= c3.unwrap());
                    assert!(c3.unwrap() >= c2.unwrap());
                    assert!(c2.unwrap() >= c1.unwrap());
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
        assert!(loc.count().is_some());
        loop {
            match mt.transition_control_point(&loc) {
                TransitionControlPoint::NoAction => (),
                TransitionControlPoint::StartTracing => break,
                _ => unreachable!(),
            }
        }
        assert!(matches!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::StopTracing(_)
        ));
        // At this point, we have nothing to meaningfully test over the `basic_transitions` test.
    }

    #[test]
    fn locations_dont_get_stuck_tracing() {
        // If tracing a location fails too many times (e.g. because the thread terminates before
        // tracing is complete), the location must be marked as DontTrace.

        const THRESHOLD: HotThreshold = 5;
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(THRESHOLD);
        let loc = Arc::new(Location::new());

        // Get the location to the point of being hot.
        for _ in 0..THRESHOLD {
            assert_eq!(
                mt.transition_control_point(&loc),
                TransitionControlPoint::NoAction
            );
        }

        // Start tracing in a thread and purposefully let the thread terminate before tracing is
        // complete.
        for i in 0..mt.trace_failure_threshold() + 1 {
            {
                let mt = Arc::clone(&mt);
                let loc = Arc::clone(&loc);
                thread::spawn(move || {
                    assert!(matches!(
                        mt.transition_control_point(&loc),
                        TransitionControlPoint::StartTracing
                    ));
                })
                .join()
                .unwrap();
            }
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Tracing
            ));
            assert_eq!(loc.hot_location().unwrap().lock().trace_failure, i);
        }

        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        assert_eq!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::NoAction
        );
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::DontTrace
        ));
    }

    #[test]
    fn locations_can_fail_tracing_before_succeeding() {
        // Test that a location can fail tracing multiple times before being successfully traced.

        const THRESHOLD: HotThreshold = 5;
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(THRESHOLD);
        let loc = Arc::new(Location::new());

        // Get the location to the point of being hot.
        for _ in 0..THRESHOLD {
            assert_eq!(
                mt.transition_control_point(&loc),
                TransitionControlPoint::NoAction
            );
        }

        // Start tracing in a thread and purposefully let the thread terminate before tracing is
        // complete.
        for i in 0..mt.trace_failure_threshold() {
            {
                let mt = Arc::clone(&mt);
                let loc = Arc::clone(&loc);
                thread::spawn(move || {
                    assert!(matches!(
                        mt.transition_control_point(&loc),
                        TransitionControlPoint::StartTracing
                    ));
                })
                .join()
                .unwrap();
            }
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Tracing
            ));
            assert_eq!(loc.hot_location().unwrap().lock().trace_failure, i);
        }

        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        // Start tracing again...
        assert!(matches!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::StartTracing
        ));
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        // ...and this time let tracing succeed.
        assert!(matches!(
            mt.transition_control_point(&loc),
            TransitionControlPoint::StopTracing(_)
        ));
        // If tracing succeeded, we'll now be in the Compiling state.
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling
        ));
    }

    #[test]
    fn dont_trace_two_locations_simultaneously_in_one_thread() {
        // A thread can only trace one Location at a time: if, having started tracing, it
        // encounters another Location which has reached its hot threshold, it just ignores it.
        // Once the first location is compiled, the second location can then be compiled.

        const THRESHOLD: HotThreshold = 5;
        let mt = MT::new().unwrap();
        mt.set_hot_threshold(THRESHOLD);
        let loc1 = Location::new();
        let loc2 = Location::new();

        for _ in 0..THRESHOLD {
            assert_eq!(
                mt.transition_control_point(&loc1),
                TransitionControlPoint::NoAction
            );
            assert_eq!(
                mt.transition_control_point(&loc2),
                TransitionControlPoint::NoAction
            );
        }
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::StartTracing
        ));
        assert_eq!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::NoAction
        );
        assert!(matches!(
            loc1.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        assert_eq!(loc2.count(), Some(THRESHOLD));
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::StopTracing(_)
        ));
        assert!(matches!(
            loc1.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling
        ));
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StartTracing
        ));
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StopTracing(_)
        ));
    }

    #[test]
    fn only_one_thread_starts_tracing() {
        // If multiple threads hammer away at a location, only one of them can win the race to
        // trace it (and then compile it etc.).

        // We need to set a high enough threshold that the threads are likely to meaningfully
        // interleave when interacting with the location.
        const THRESHOLD: HotThreshold = 100000;
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(THRESHOLD);
        let loc = Arc::new(Location::new());

        let mut thrs = Vec::new();
        let num_starts = Arc::new(AtomicU64::new(0));
        for _ in 0..num_cpus::get() {
            let loc = Arc::clone(&loc);
            let mt = Arc::clone(&mt);
            let num_starts = Arc::clone(&num_starts);
            thrs.push(thread::spawn(move || {
                for _ in 0..THRESHOLD {
                    match mt.transition_control_point(&loc) {
                        TransitionControlPoint::NoAction => (),
                        TransitionControlPoint::Execute(_) => (),
                        TransitionControlPoint::StartTracing => {
                            num_starts.fetch_add(1, Ordering::Relaxed);
                            assert!(matches!(
                                loc.hot_location().unwrap().lock().kind,
                                HotLocationKind::Tracing
                            ));

                            match mt.transition_control_point(&loc) {
                                TransitionControlPoint::StopTracing(_) => {
                                    assert!(matches!(
                                        loc.hot_location().unwrap().lock().kind,
                                        HotLocationKind::Compiling
                                    ));
                                    assert_eq!(
                                        mt.transition_control_point(&loc),
                                        TransitionControlPoint::NoAction
                                    );
                                    assert!(matches!(
                                        loc.hot_location().unwrap().lock().kind,
                                        HotLocationKind::Compiling
                                    ));
                                    loc.hot_location().unwrap().lock().kind =
                                        HotLocationKind::Compiled(Arc::new(
                                            CompiledTrace::new_testing(),
                                        ));
                                }
                                x => unreachable!("Reached incorrect state {:?}", x),
                            }
                            loop {
                                if let TransitionControlPoint::Execute(_) =
                                    mt.transition_control_point(&loc)
                                {
                                    break;
                                }
                            }
                            break;
                        }
                        TransitionControlPoint::StopTracing(_)
                        | TransitionControlPoint::StopSideTracing(_, _, _) => unreachable!(),
                    }
                }
            }));
        }

        for t in thrs {
            t.join().unwrap();
        }

        assert_eq!(num_starts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn two_tracing_threads_must_not_stop_each_others_tracing_location() {
        // A tracing thread can only stop tracing when it encounters the specific Location that
        // caused it to start tracing. If it encounters another Location that also happens to be
        // tracing, it must ignore it.

        const THRESHOLD: HotThreshold = 5;
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(THRESHOLD);
        let loc1 = Arc::new(Location::new());
        let loc2 = Location::new();

        for _ in 0..THRESHOLD {
            assert_eq!(
                mt.transition_control_point(&loc1),
                TransitionControlPoint::NoAction
            );
            assert_eq!(
                mt.transition_control_point(&loc2),
                TransitionControlPoint::NoAction
            );
        }

        {
            let mt = Arc::clone(&mt);
            let loc1 = Arc::clone(&loc1);
            thread::spawn(move || {
                assert!(matches!(
                    mt.transition_control_point(&loc1),
                    TransitionControlPoint::StartTracing
                ));
            })
            .join()
            .unwrap();
        }

        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StartTracing
        ));
        assert_eq!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::NoAction
        );
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StopTracing(_)
        ));
    }

    #[test]
    fn two_sidetracing_threads_must_not_stop_each_others_tracing_location() {
        // A side-tracing thread can only stop tracing when it encounters the specific Location
        // that caused it to start tracing. If it encounters another Location that also happens to
        // be tracing, it must ignore it.

        const THRESHOLD: HotThreshold = 5;
        let mt = MT::new().unwrap();
        mt.set_hot_threshold(THRESHOLD);
        let loc1 = Arc::new(Location::new());
        let loc2 = Location::new();

        fn to_compiled(mt: &Arc<MT>, loc: &Location) {
            for _ in 0..THRESHOLD {
                assert_eq!(
                    mt.transition_control_point(&loc),
                    TransitionControlPoint::NoAction
                );
            }

            assert_eq!(
                mt.transition_control_point(&loc),
                TransitionControlPoint::StartTracing
            );
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Tracing
            ));
            match mt.transition_control_point(&loc) {
                TransitionControlPoint::StopTracing(_) => {
                    assert!(matches!(
                        loc.hot_location().unwrap().lock().kind,
                        HotLocationKind::Compiling
                    ));
                    loc.hot_location().unwrap().lock().kind =
                        HotLocationKind::Compiled(Arc::new(CompiledTrace::new_testing()));
                }
                _ => unreachable!(),
            }
        }

        to_compiled(&mt, &loc1);
        to_compiled(&mt, &loc2);

        {
            let mt = Arc::clone(&mt);
            let loc1 = Arc::clone(&loc1);
            thread::spawn(move || {
                let sti = SideTraceInfo {
                    callstack: std::ptr::null(),
                    aotvalsptr: std::ptr::null(),
                    aotvalslen: 0,
                    guardid: GuardId::illegal(),
                };
                assert!(matches!(
                    mt.transition_guard_failure(
                        loc1.hot_location_arc_clone().unwrap(),
                        sti,
                        Arc::new(CompiledTrace::new_testing()),
                    ),
                    TransitionGuardFailure::StartSideTracing
                ));
            })
            .join()
            .unwrap();
        }

        let sti = SideTraceInfo {
            callstack: std::ptr::null(),
            aotvalsptr: std::ptr::null(),
            aotvalslen: 0,
            guardid: GuardId::illegal(),
        };
        assert!(matches!(
            mt.transition_guard_failure(
                loc2.hot_location_arc_clone().unwrap(),
                sti,
                Arc::new(CompiledTrace::new_testing()),
            ),
            TransitionGuardFailure::StartSideTracing
        ));
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::Execute(_)
        ));
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StopSideTracing(_, _, _)
        ));
    }

    #[bench]
    fn bench_single_threaded_control_point(b: &mut Bencher) {
        let mt = MT::new().unwrap();
        let loc = Location::new();
        b.iter(|| {
            for _ in 0..100000 {
                black_box(mt.transition_control_point(&loc));
            }
        });
    }

    #[bench]
    fn bench_multi_threaded_control_point(b: &mut Bencher) {
        let mt = Arc::new(MT::new().unwrap());
        let loc = Arc::new(Location::new());
        b.iter(|| {
            let mut thrs = vec![];
            for _ in 0..4 {
                let loc = Arc::clone(&loc);
                let mt = Arc::clone(&mt);
                thrs.push(thread::spawn(move || {
                    for _ in 0..100 {
                        black_box(mt.transition_control_point(&loc));
                    }
                }));
            }
            for t in thrs {
                t.join().unwrap();
            }
        });
    }

    #[test]
    fn dont_trace_execution_of_a_trace() {
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(0);
        let loc1 = Location::new();
        let loc2 = Location::new();

        // Get `loc1` to the point where there's a compiled trace for it.
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::StartTracing
        ));
        if let TransitionControlPoint::StopTracing(_) = mt.transition_control_point(&loc1) {
            loc1.hot_location().unwrap().lock().kind =
                HotLocationKind::Compiled(Arc::new(CompiledTrace::new_testing()));
        } else {
            panic!();
        }

        // If we transition `loc2` into `StartTracing`, then (for now) we should not execute the
        // trace for `loc1`, as another location is being traced and we don't want to trace the
        // execution of the trace!
        //
        // FIXME: this behaviour will need to change in the future:
        // https://github.com/ykjit/yk/issues/519
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StartTracing
        ));
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::NoAction
        ));

        // But once we stop tracing for `loc2`, we should be able to execute the trace for `loc1`.
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StopTracing(_)
        ));
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::Execute(_)
        ));
    }
}
