//! The main end-user interface to the meta-tracing system.

use std::{
    assert_matches::debug_assert_matches,
    cell::RefCell,
    cmp,
    collections::VecDeque,
    env,
    error::Error,
    ffi::c_void,
    marker::PhantomData,
    sync::{
        atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use parking_lot::{Condvar, Mutex, MutexGuard};
#[cfg(not(all(feature = "yk_testing", not(test))))]
use parking_lot_core::SpinWait;

use crate::{
    aotsmp::{load_aot_stackmaps, AOT_STACKMAPS},
    compile::{default_compiler, CompilationError, CompiledTrace, Compiler, GuardIdx},
    location::{HotLocation, HotLocationKind, Location, TraceFailed},
    log::{
        log_jit_state,
        stats::{Stats, TimingState},
    },
    trace::{default_tracer, AOTTraceIterator, TraceRecorder, Tracer},
};

// The HotThreshold must be less than a machine word wide for [`Location::Location`] to do its
// pointer tagging thing. We therefore choose a type which makes this statically clear to
// users rather than having them try to use (say) u64::max() on a 64 bit machine and get a run-time
// error.
#[cfg(target_pointer_width = "64")]
pub type HotThreshold = u32;
#[cfg(target_pointer_width = "64")]
type AtomicHotThreshold = AtomicU32;

/// How often can a [HotLocation] or [Guard] lead to an error in tracing or compilation before we
/// give up trying to trace (or compile...) it?
pub type TraceCompilationErrorThreshold = u16;
pub type AtomicTraceCompilationErrorThreshold = AtomicU16;

/// How many basic blocks long can a trace be before we give up trying to compile it? Note that the
/// slower our compiler, the lower this will have to be in order to give the perception of
/// reasonable performance.
/// FIXME: needs to be configurable.
pub(crate) const DEFAULT_TRACE_TOO_LONG: usize = 5000;
const DEFAULT_HOT_THRESHOLD: HotThreshold = 50;
const DEFAULT_SIDETRACE_THRESHOLD: HotThreshold = 5;
/// How often can a [HotLocation] or [Guard] lead to an error in tracing or compilation before we
/// give up trying to trace (or compile...) it?
const DEFAULT_TRACECOMPILATION_ERROR_THRESHOLD: TraceCompilationErrorThreshold = 5;
static REG64_SIZE: usize = 8;

thread_local! {
    static THREAD_MTTHREAD: MTThread = MTThread::new();
}

/// A meta-tracer. Note that this is conceptually a "front-end" to the actual meta-tracer akin to
/// an `Rc`: this struct can be freely `clone()`d without duplicating the underlying meta-tracer.
pub struct MT {
    hot_threshold: AtomicHotThreshold,
    sidetrace_threshold: AtomicHotThreshold,
    trace_failure_threshold: AtomicTraceCompilationErrorThreshold,
    /// The ordered queue of compilation worker functions.
    job_queue: Arc<(Condvar, Mutex<VecDeque<Box<dyn FnOnce() + Send>>>)>,
    /// The hard cap on the number of worker threads.
    max_worker_threads: AtomicUsize,
    /// How many worker threads are currently running. Note that this may temporarily be `>`
    /// [`max_worker_threads`].
    active_worker_threads: AtomicUsize,
    /// The [Tracer] that should be used for creating future traces. Note that this might not be
    /// the same as the tracer(s) used to create past traces.
    tracer: Mutex<Arc<dyn Tracer>>,
    /// The [Compiler] that will be used for compiling future `IRTrace`s. Note that this might not
    /// be the same as the compiler(s) used to compile past `IRTrace`s.
    compiler: Mutex<Arc<dyn Compiler>>,
    /// A monotonically increasing integer that semi-uniquely identifies each compiled trace. This
    /// is only useful for general debugging purposes, and must not be relied upon for semantic
    /// correctness, because the IDs can repeat when the underlying `u64` overflows/wraps.
    compiled_trace_id: AtomicU64,
    pub(crate) stats: Stats,
}

impl std::fmt::Debug for MT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MT")
    }
}

impl MT {
    // Create a new meta-tracer instance. Arbitrarily many of these can be created, though there
    // are no guarantees as to whether they will share resources effectively or fairly.
    pub fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        load_aot_stackmaps();
        let hot_threshold = match env::var("YK_HOT_THRESHOLD") {
            Ok(s) => s
                .parse::<HotThreshold>()
                .map_err(|e| format!("Invalid hot threshold '{s}': {e}"))?,
            Err(_) => DEFAULT_HOT_THRESHOLD,
        };
        Ok(Arc::new(Self {
            hot_threshold: AtomicHotThreshold::new(hot_threshold),
            sidetrace_threshold: AtomicHotThreshold::new(DEFAULT_SIDETRACE_THRESHOLD),
            trace_failure_threshold: AtomicTraceCompilationErrorThreshold::new(
                DEFAULT_TRACECOMPILATION_ERROR_THRESHOLD,
            ),
            job_queue: Arc::new((Condvar::new(), Mutex::new(VecDeque::new()))),
            max_worker_threads: AtomicUsize::new(cmp::max(1, num_cpus::get() - 1)),
            active_worker_threads: AtomicUsize::new(0),
            tracer: Mutex::new(default_tracer()?),
            compiler: Mutex::new(default_compiler()?),
            compiled_trace_id: AtomicU64::new(0),
            stats: Stats::new(),
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
    pub fn trace_failure_threshold(self: &Arc<Self>) -> TraceCompilationErrorThreshold {
        self.trace_failure_threshold.load(Ordering::Relaxed)
    }

    /// Set the threshold at which a `Location` from which tracing has failed multiple times is
    /// marked as "do not try tracing again".
    pub fn set_trace_failure_threshold(
        self: &Arc<Self>,
        trace_failure_threshold: TraceCompilationErrorThreshold,
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

    /// Return the semi-unique ID for the next compiled trace. Note: this is only useful for
    /// general debugging purposes, and must not be relied upon for semantic correctness, because
    /// the IDs will wrap when the underlying `u64` overflows.
    pub(crate) fn next_compiled_trace_id(self: &Arc<Self>) -> u64 {
        // Note: fetch_add is documented to wrap on overflow.
        self.compiled_trace_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Queue `job` to be run on a worker thread.
    fn queue_job(self: &Arc<Self>, job: Box<dyn FnOnce() + Send>) {
        // We have a very simple model of worker threads. Each time a job is queued, we spin up a
        // new worker thread iff we aren't already running the maximum number of worker threads.
        // Once started, a worker thread never dies, waiting endlessly for work.

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

            self.stats.timing_state(TimingState::None);
            // We only keep a weak reference alive to `self`, as otherwise an active compiler job
            // causes `self` to never be dropped.
            let mt = Arc::downgrade(self);
            let jq = Arc::clone(&self.job_queue);
            thread::spawn(move || {
                let (cv, mtx) = &*jq;
                let mut lock = mtx.lock();
                // If the strong count for `mt` is 0 then it has been dropped and there is no
                // point trying to do further work, even if there is work in the queue.
                while mt.upgrade().is_some() {
                    match lock.pop_front() {
                        Some(x) => {
                            MutexGuard::unlocked(&mut lock, x);
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
        smid: u64,
    ) {
        match self.transition_control_point(loc) {
            TransitionControlPoint::NoAction => (),
            TransitionControlPoint::Execute(ctr) => {
                log_jit_state("enter-jit-code");
                self.stats.trace_executed();

                // Compute the rsp of the control_point frame.
                let (rec, pinfo) = AOT_STACKMAPS
                    .as_ref()
                    .unwrap()
                    .get(usize::try_from(smid).unwrap());
                let mut rsp = unsafe { frameaddr.byte_sub(usize::try_from(rec.size).unwrap()) };
                if pinfo.hasfp {
                    rsp = unsafe { rsp.byte_add(REG64_SIZE) };
                }
                let trace_addr = ctr.entry();
                MTThread::with(|mtt| {
                    mtt.set_running_trace(Some(ctr));
                });
                self.stats.timing_state(TimingState::JitExecuting);

                // FIXME: Calling this function overwrites the current (Rust) function frame,
                // rather than unwinding it. https://github.com/ykjit/yk/issues/778.
                unsafe { exec_trace(ctrlp_vars, frameaddr, rsp, trace_addr) };
            }
            TransitionControlPoint::StartTracing(hl) => {
                log_jit_state("start-tracing");
                let tracer = {
                    let lk = self.tracer.lock();
                    Arc::clone(&*lk)
                };
                match Arc::clone(&tracer).start_recorder() {
                    Ok(tt) => MTThread::with(|mtt| {
                        *mtt.tstate.borrow_mut() = MTThreadState::Tracing {
                            hl,
                            thread_tracer: tt,
                            promotions: Vec::new(),
                        };
                    }),
                    Err(e) => todo!("{e:?}"),
                }
            }
            TransitionControlPoint::StopTracing => {
                // Assuming no bugs elsewhere, the `unwrap`s cannot fail, because `StartTracing`
                // will have put a `Some` in the `Rc`.
                let (hl, thread_tracer, promotions) =
                    MTThread::with(
                        |mtt| match mtt.tstate.replace(MTThreadState::Interpreting) {
                            MTThreadState::Tracing {
                                hl,
                                thread_tracer,
                                promotions,
                            } => (hl, thread_tracer, promotions),
                            _ => unreachable!(),
                        },
                    );
                match thread_tracer.stop() {
                    Ok(utrace) => {
                        self.stats.timing_state(TimingState::None);
                        log_jit_state("stop-tracing");
                        self.queue_compile_job((utrace, promotions.into_boxed_slice()), hl, None);
                    }
                    Err(_e) => {
                        self.stats.timing_state(TimingState::None);
                        self.stats.trace_recorded_err();
                        log_jit_state(&format!("stop-tracing-aborted: {_e}"));
                    }
                }
            }
            TransitionControlPoint::StopSideTracing {
                gidx: guardid,
                parent_ctr,
            } => {
                // Assuming no bugs elsewhere, the `unwrap`s cannot fail, because
                // `StartSideTracing` will have put a `Some` in the `Rc`.
                let (hl, thread_tracer, promotions) =
                    MTThread::with(
                        |mtt| match mtt.tstate.replace(MTThreadState::Interpreting) {
                            MTThreadState::Tracing {
                                hl,
                                thread_tracer,
                                promotions,
                            } => (hl, thread_tracer, promotions),
                            _ => unreachable!(),
                        },
                    );
                self.stats.timing_state(TimingState::TraceMapping);
                match thread_tracer.stop() {
                    Ok(utrace) => {
                        self.stats.timing_state(TimingState::None);
                        log_jit_state("stop-side-tracing");
                        self.queue_compile_job(
                            (utrace, promotions.into_boxed_slice()),
                            hl,
                            Some((guardid, parent_ctr)),
                        );
                    }
                    Err(_e) => {
                        self.stats.timing_state(TimingState::None);
                        self.stats.trace_recorded_err();
                        log_jit_state(&format!("stop-side-tracing-aborted: {_e}"));
                    }
                }
            }
        }
    }

    /// Perform the next step to `loc` in the `Location` state-machine for a control point. If
    /// `loc` moves to the Compiled state, return a pointer to a [CompiledTrace] object.
    fn transition_control_point(self: &Arc<Self>, loc: &Location) -> TransitionControlPoint {
        MTThread::with(|mtt| {
            let is_tracing = mtt.is_tracing();
            match loc.hot_location() {
                Some(hl) => {
                    // If this thread is tracing something, we *must* grab the [HotLocation] lock,
                    // because we need to know for sure if `loc` is the point at which we should
                    // stop tracing. In most compilation modes, we are willing to give up trying to
                    // lock and return if it looks like it will take too long. When `yk_testing` is
                    // enabled, however, this introduces non-determinism, so in that compilation
                    // mode only we guarantee to grab the lock.
                    let mut lk;

                    #[cfg(not(all(feature = "yk_testing", not(test))))]
                    {
                        // If this thread is not tracing anything, however, it's not worth
                        // contending too much with other threads: we try moderately hard to grab
                        // the lock, but we don't want to park this thread.
                        if !is_tracing {
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
                                if let Some(x) = hl.try_lock() {
                                    lk = x;
                                    break;
                                }
                                if !sw.spin() {
                                    return TransitionControlPoint::NoAction;
                                }
                            }
                        } else {
                            // This thread is tracing something, so we must grab the lock.
                            lk = hl.lock();
                        };
                    }

                    #[cfg(all(feature = "yk_testing", not(test)))]
                    {
                        lk = hl.lock();
                    }

                    match lk.kind {
                        HotLocationKind::Compiled(ref ctr) => {
                            if is_tracing {
                                // This thread is tracing something, so bail out as quickly as possible
                                TransitionControlPoint::NoAction
                            } else {
                                TransitionControlPoint::Execute(Arc::clone(ctr))
                            }
                        }
                        HotLocationKind::Compiling => TransitionControlPoint::NoAction,
                        HotLocationKind::Tracing => {
                            let hl = loc.hot_location_arc_clone().unwrap();
                            match &*mtt.tstate.borrow() {
                                MTThreadState::Tracing { hl: thread_hl, .. } => {
                                    // This thread is tracing something...
                                    if !Arc::ptr_eq(thread_hl, &hl) {
                                        // ...but not this Location.
                                        TransitionControlPoint::NoAction
                                    } else {
                                        // ...and it's this location: we have therefore finished
                                        // tracing the loop.
                                        lk.kind = HotLocationKind::Compiling;
                                        TransitionControlPoint::StopTracing
                                    }
                                }
                                _ => {
                                    // FIXME: This branch is also used by side tracing. That's not
                                    // necessarily wrong, but it wasn't what was intended. We
                                    // should at least explicitly think about whether this is the
                                    // best way of doing things or not.

                                    // This thread isn't tracing anything. Note that because we called
                                    // `hot_location_arc_clone` above, the strong count of an `Arc`
                                    // that's no longer being used by that thread will be 2.
                                    if Arc::strong_count(&hl) == 2 {
                                        // Another thread was tracing this location but it's terminated.
                                        self.stats.trace_recorded_err();
                                        match lk.tracecompilation_error(self) {
                                            TraceFailed::KeepTrying => {
                                                TransitionControlPoint::StartTracing(hl)
                                            }
                                            TraceFailed::DontTrace => {
                                                TransitionControlPoint::NoAction
                                            }
                                        }
                                    } else {
                                        // Another thread is tracing this location.
                                        TransitionControlPoint::NoAction
                                    }
                                }
                            }
                        }
                        HotLocationKind::SideTracing {
                            ref root_ctr,
                            gidx,
                            ref parent_ctr,
                        } => {
                            let hl = loc.hot_location_arc_clone().unwrap();
                            match &*mtt.tstate.borrow() {
                                MTThreadState::Tracing { hl: thread_hl, .. } => {
                                    // This thread is tracing something...
                                    if !Arc::ptr_eq(thread_hl, &hl) {
                                        // ...but not this Location.
                                        TransitionControlPoint::Execute(Arc::clone(root_ctr))
                                    } else {
                                        // ...and it's this location: we have therefore finished
                                        // tracing the loop.
                                        let parent_ctr = Arc::clone(parent_ctr);
                                        lk.kind = HotLocationKind::Compiled(Arc::clone(root_ctr));
                                        drop(lk);
                                        TransitionControlPoint::StopSideTracing { gidx, parent_ctr }
                                    }
                                }
                                _ => {
                                    // This thread isn't tracing anything.
                                    TransitionControlPoint::Execute(Arc::clone(root_ctr))
                                }
                            }
                        }
                        HotLocationKind::DontTrace => TransitionControlPoint::NoAction,
                    }
                }
                None => {
                    if is_tracing {
                        // This thread is tracing something, so bail out as quickly as possible
                        return TransitionControlPoint::NoAction;
                    }
                    match loc.inc_count() {
                        Some(x) => {
                            debug_assert!(self.hot_threshold() < HotThreshold::MAX);
                            if x < self.hot_threshold() + 1 {
                                TransitionControlPoint::NoAction
                            } else {
                                let hl = HotLocation {
                                    kind: HotLocationKind::Tracing,
                                    tracecompilation_errors: 0,
                                };
                                if let Some(hl) = loc.count_to_hot_location(x, hl) {
                                    debug_assert!(!is_tracing);
                                    TransitionControlPoint::StartTracing(hl)
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
        gidx: GuardIdx,
        parent_ctr: Arc<dyn CompiledTrace>,
    ) -> TransitionGuardFailure {
        if let Some(hl) = parent_ctr.hl().upgrade() {
            MTThread::with(|mtt| {
                // This thread should not be tracing anything.
                debug_assert!(!mtt.is_tracing());
                let mut lk = hl.lock();
                if let HotLocationKind::Compiled(ref root_ctr) = lk.kind {
                    lk.kind = HotLocationKind::SideTracing {
                        root_ctr: Arc::clone(root_ctr),
                        gidx,
                        parent_ctr,
                    };
                    drop(lk);
                    TransitionGuardFailure::StartSideTracing(hl)
                } else {
                    // The top-level trace's [HotLocation] might have changed to another state while
                    // the associated trace was executing; or we raced with another thread (which is
                    // most likely to have started side tracing itself).
                    TransitionGuardFailure::NoAction
                }
            })
        } else {
            // The parent [HotLocation] has been garbage collected.
            TransitionGuardFailure::NoAction
        }
    }

    /// Start recording a side trace for a guard that failed in `ctr`.
    pub(crate) fn side_trace(self: &Arc<Self>, gidx: GuardIdx, parent: Arc<dyn CompiledTrace>) {
        match self.transition_guard_failure(gidx, parent) {
            TransitionGuardFailure::NoAction => todo!(),
            TransitionGuardFailure::StartSideTracing(hl) => {
                log_jit_state("start-side-tracing");
                let tracer = {
                    let lk = self.tracer.lock();
                    Arc::clone(&*lk)
                };
                match Arc::clone(&tracer).start_recorder() {
                    Ok(tt) => MTThread::with(|mtt| {
                        *mtt.tstate.borrow_mut() = MTThreadState::Tracing {
                            hl,
                            thread_tracer: tt,
                            promotions: Vec::new(),
                        };
                    }),
                    Err(e) => todo!("{e:?}"),
                }
            }
        }
    }

    /// Add a compilation job for to the global work queue:
    ///   * `utrace` is the trace to be compiled.
    ///   * `hl_arc` is the [HotLocation] this compilation job is related to.
    ///   * `sidetrace`, if not `None`, specifies that this is a side-trace compilation job.
    ///     The `Arc<dyn CompiledTrace>` is the parent [CompiledTrace] for the side-trace. Because
    ///     side-traces can nest, this may or may not be the same [CompiledTrace] as contained
    ///     in the `hl_arc`.
    fn queue_compile_job(
        self: &Arc<Self>,
        trace_iter: (Box<dyn AOTTraceIterator>, Box<[usize]>),
        hl_arc: Arc<Mutex<HotLocation>>,
        sidetrace: Option<(GuardIdx, Arc<dyn CompiledTrace>)>,
    ) {
        self.stats.trace_recorded_ok();
        let mt = Arc::clone(self);
        let do_compile = move || {
            let compiler = {
                let lk = mt.compiler.lock();
                Arc::clone(&*lk)
            };
            mt.stats.timing_state(TimingState::Compiling);
            let (sti, guardid) = if let Some((guardid, ctr)) = &sidetrace {
                (Some(ctr.sidetraceinfo(*guardid)), Some(*guardid))
            } else {
                (None, None)
            };
            match compiler.compile(Arc::clone(&mt), trace_iter, sti, Arc::clone(&hl_arc)) {
                Ok(ct) => {
                    if let Some((_, parent_ctr)) = sidetrace {
                        parent_ctr.guard(guardid.unwrap()).set_ctr(ct);
                    } else {
                        let mut hl = hl_arc.lock();
                        debug_assert_matches!(hl.kind, HotLocationKind::Compiling);
                        hl.kind = HotLocationKind::Compiled(ct);
                    }
                    mt.stats.trace_compiled_ok();
                }
                Err(e) => {
                    mt.stats.trace_compiled_err();
                    hl_arc.lock().tracecompilation_error(&mt);
                    match e {
                        CompilationError::General(_reason)
                        | CompilationError::LimitExceeded(_reason) => {
                            log_jit_state(&format!("trace-compilation-aborted: {_reason}"));
                        }
                        CompilationError::InternalError(reason) => {
                            #[cfg(feature = "ykd")]
                            panic!("{reason}");
                            #[cfg(not(feature = "ykd"))]
                            {
                                eprintln!("yk error (trying to continue): {reason}");
                            }
                        }
                        CompilationError::ResourceExhausted(e) => {
                            eprintln!("yk warning: {e}");
                            log_jit_state(&format!("trace-compilation-aborted: {e}"));
                        }
                    }
                }
            }

            mt.stats.timing_state(TimingState::None);
        };

        #[cfg(feature = "yk_testing")]
        if let Ok(true) = env::var("YKD_SERIALISE_COMPILATION").map(|x| x.as_str() == "1") {
            // To ensure that we properly test that compilation can occur in another thread, we
            // spin up a new thread for each compilation. This is only acceptable because a)
            // `SERIALISE_COMPILATION` is an internal yk testing feature b) when we use it we're
            // checking correctness, not performance.
            thread::spawn(do_compile).join().unwrap();
            return;
        }

        self.queue_job(Box::new(do_compile));
    }
}

impl Drop for MT {
    fn drop(&mut self) {
        self.stats.timing_state(TimingState::None);
    }
}

/// Execute a trace. Note: this overwrites the current (Rust) function frame.
#[cfg(target_arch = "x86_64")]
#[naked]
unsafe extern "C" fn exec_trace(
    ctrlp_vars: *mut c_void,
    frameaddr: *const c_void,
    rsp: *const c_void,
    trace: *const c_void,
) -> ! {
    std::arch::asm!(
        // Reset RSP to the end of the control point frame (this doesn't include the
        // return address)
        "mov rsp, rdx",
        // Call the trace function.
        "call rcx",
        "ret",
        options(noreturn)
    )
}

/// [MTThread]'s major job is to record what state in the "interpreting/tracing/executing"
/// state-machine this thread is in. This enum contains the states.
enum MTThreadState {
    /// This thread is executing in the normal interpreter: it is not executing a trace or
    /// recording a trace.
    Interpreting,
    /// This thread is recording a trace.
    Tracing {
        /// The [HotLocation] the trace will end at. For a top-level trace, this will be the same
        /// [HotLocation] the trace started at; for a side-trace, tracing started elsewhere.
        hl: Arc<Mutex<HotLocation>>,
        /// What tracer is being used to record this trace? Needed for trace mapping.
        thread_tracer: Box<dyn TraceRecorder>,
        /// Records the content of data recorded via `yk_promote`.
        promotions: Vec<usize>,
    },
    /// This thread is executing a trace. Note that the `dyn CompiledTrace` serves two different purposes:
    ///
    /// 1. It is needed for side traces and the like.
    /// 2. It allows another thread to tell whether the thread that started tracing a [Location] is
    ///    still alive or not by inspecting its strong count (if the strong count is equal to 1
    ///    then the thread died while tracing). Note that this relies on thread local storage
    ///    dropping the [MTThread] instance and (by implication) dropping the [Arc] and
    ///    decrementing its strong count. Unfortunately, there is no guarantee that thread local
    ///    storage will be dropped when a thread dies (and there is also significant platform
    ///    variation in regard to dropping thread locals), so this mechanism can't be fully relied
    ///    upon: however, we can't monitor thread death in any other reasonable way, so this will
    ///    have to do.
    Executing(Arc<dyn CompiledTrace>),
}

/// Meta-tracer per-thread state. Note that this struct is neither `Send` nor `Sync`: it can only
/// be accessed from within a single thread.
pub(crate) struct MTThread {
    /// Where in the "interpreting/tracing/executing" is this thread?
    tstate: RefCell<MTThreadState>,
    // Raw pointers are neither send nor sync.
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl MTThread {
    fn new() -> Self {
        MTThread {
            tstate: RefCell::new(MTThreadState::Interpreting),
            _dont_send_or_sync_me: PhantomData,
        }
    }

    /// Call `f` with a reference to this thread's [MTThread] instance.
    ///
    /// # Panics
    ///
    /// For the same reasons as [thread::local::LocalKey::with].
    pub fn with<F, R>(f: F) -> R
    where
        F: FnOnce(&MTThread) -> R,
    {
        THREAD_MTTHREAD.with(|mtt| f(mtt))
    }

    /// Is this thread currently tracing something?
    pub(crate) fn is_tracing(&self) -> bool {
        matches!(&*self.tstate.borrow(), &MTThreadState::Tracing { .. })
    }

    /// If a trace is currently running, return a reference to its `CompiledTrace`.
    pub(crate) fn running_trace(&self) -> Option<Arc<dyn CompiledTrace>> {
        match &*self.tstate.borrow() {
            MTThreadState::Executing(ctr_arc) => Some(Arc::clone(ctr_arc)),
            _ => None,
        }
    }

    /// Update the currently running trace: `None` means that no trace is running.
    pub(crate) fn set_running_trace(&self, ctr: Option<Arc<dyn CompiledTrace>>) {
        *self.tstate.borrow_mut() = match ctr {
            Some(ctr) => MTThreadState::Executing(ctr),
            None => MTThreadState::Interpreting,
        };
    }

    /// Records `val` as a value to be promoted. Returns `true` if either: no trace is being
    /// recorded; or recording the promotion succeeded.
    ///
    /// If `false` is returned, the current trace is unable to record the promotion successfully
    /// and further calls are probably pointless, though they will not cause the tracer to enter
    /// undefined behaviour territory.
    pub(crate) fn promote_usize(&self, val: usize) -> bool {
        if let MTThreadState::Tracing {
            ref mut promotions, ..
        } = *self.tstate.borrow_mut()
        {
            promotions.push(val);
        }
        true
    }
}

/// What action should a caller of [MT::transition_control_point] take?
#[derive(Debug)]
enum TransitionControlPoint {
    NoAction,
    Execute(Arc<dyn CompiledTrace>),
    StartTracing(Arc<Mutex<HotLocation>>),
    StopTracing,
    StopSideTracing {
        gidx: GuardIdx,
        parent_ctr: Arc<dyn CompiledTrace>,
    },
}

/// What action should a caller of [MT::transition_guard_failure] take?
#[derive(Debug)]
pub(crate) enum TransitionGuardFailure {
    NoAction,
    StartSideTracing(Arc<Mutex<HotLocation>>),
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use crate::{
        compile::{CompiledTraceTesting, CompiledTraceTestingWithHl},
        trace::TraceRecorderError,
    };
    use std::hint::black_box;
    use test::bench::Bencher;

    // We only implement enough of the equality function for the tests we have.
    impl PartialEq for TransitionControlPoint {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (TransitionControlPoint::NoAction, TransitionControlPoint::NoAction) => true,
                (TransitionControlPoint::Execute(p1), TransitionControlPoint::Execute(p2)) => {
                    std::ptr::eq(p1, p2)
                }
                (
                    TransitionControlPoint::StartTracing(_),
                    TransitionControlPoint::StartTracing(_),
                ) => true,
                (x, y) => todo!("{:?} {:?}", x, y),
            }
        }
    }

    struct DummyTraceRecorder;

    impl TraceRecorder for DummyTraceRecorder {
        fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
            todo!();
        }
    }

    fn expect_start_tracing(mt: &Arc<MT>, loc: &Location) {
        let TransitionControlPoint::StartTracing(hl) = mt.transition_control_point(loc) else {
            panic!()
        };
        MTThread::with(|mtt| {
            *mtt.tstate.borrow_mut() = MTThreadState::Tracing {
                hl,
                thread_tracer: Box::new(DummyTraceRecorder),
                promotions: Vec::new(),
            };
        });
    }

    fn expect_stop_tracing(mt: &Arc<MT>, loc: &Location) {
        let TransitionControlPoint::StopTracing = mt.transition_control_point(loc) else {
            panic!()
        };
        MTThread::with(|mtt| {
            *mtt.tstate.borrow_mut() = MTThreadState::Interpreting;
        });
    }

    fn expect_start_side_tracing(mt: &Arc<MT>, loc: &Location) {
        let TransitionGuardFailure::StartSideTracing(hl) = mt.transition_guard_failure(
            GuardIdx::from(0),
            Arc::new(CompiledTraceTestingWithHl::new(Arc::downgrade(
                &loc.hot_location_arc_clone().unwrap(),
            ))),
        ) else {
            panic!()
        };
        MTThread::with(|mtt| {
            *mtt.tstate.borrow_mut() = MTThreadState::Tracing {
                hl,
                thread_tracer: Box::new(DummyTraceRecorder),
                promotions: Vec::new(),
            };
        });
    }

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
        expect_start_tracing(&mt, &loc);
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        expect_stop_tracing(&mt, &loc);
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling
        ));
        loc.hot_location().unwrap().lock().kind =
            HotLocationKind::Compiled(Arc::new(CompiledTraceTesting::new()));
        assert!(matches!(
            dbg!(mt.transition_control_point(&loc)),
            TransitionControlPoint::Execute(_)
        ));
        expect_start_side_tracing(&mt, &loc);

        match mt.transition_control_point(&loc) {
            TransitionControlPoint::StopSideTracing { .. } => {
                MTThread::with(|mtt| {
                    *mtt.tstate.borrow_mut() = MTThreadState::Interpreting;
                });
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
        let hot_thrsh = num_threads.saturating_mul(2500);
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
                TransitionControlPoint::StartTracing(hl) => {
                    MTThread::with(|mtt| {
                        *mtt.tstate.borrow_mut() = MTThreadState::Tracing {
                            hl,
                            thread_tracer: Box::new(DummyTraceRecorder),
                            promotions: Vec::new(),
                        };
                    });
                    break;
                }
                _ => unreachable!(),
            }
        }
        expect_stop_tracing(&mt, &loc);
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
                    expect_start_tracing(&mt, &loc);
                })
                .join()
                .unwrap();
            }
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Tracing
            ));
            assert_eq!(
                loc.hot_location().unwrap().lock().tracecompilation_errors,
                i
            );
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
                thread::spawn(move || expect_start_tracing(&mt, &loc))
                    .join()
                    .unwrap();
            }
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Tracing
            ));
            assert_eq!(
                loc.hot_location().unwrap().lock().tracecompilation_errors,
                i
            );
        }

        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        // Start tracing again...
        expect_start_tracing(&mt, &loc);
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        // ...and this time let tracing succeed.
        expect_stop_tracing(&mt, &loc);
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
        expect_start_tracing(&mt, &loc1);
        assert_eq!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::NoAction
        );
        assert!(matches!(
            loc1.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing
        ));
        assert_eq!(loc2.count(), Some(THRESHOLD));
        expect_stop_tracing(&mt, &loc1);
        assert!(matches!(
            loc1.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling
        ));
        expect_start_tracing(&mt, &loc2);
        expect_stop_tracing(&mt, &loc2);
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
                        TransitionControlPoint::StartTracing(hl) => {
                            num_starts.fetch_add(1, Ordering::Relaxed);
                            MTThread::with(|mtt| {
                                *mtt.tstate.borrow_mut() = MTThreadState::Tracing {
                                    hl,
                                    thread_tracer: Box::new(DummyTraceRecorder),
                                    promotions: Vec::new(),
                                };
                            });
                            assert!(matches!(
                                loc.hot_location().unwrap().lock().kind,
                                HotLocationKind::Tracing
                            ));
                            expect_stop_tracing(&mt, &loc);
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
                                HotLocationKind::Compiled(Arc::new(CompiledTraceTesting::new()));
                            loop {
                                if let TransitionControlPoint::Execute(_) =
                                    mt.transition_control_point(&loc)
                                {
                                    break;
                                }
                            }
                            break;
                        }
                        TransitionControlPoint::StopTracing
                        | TransitionControlPoint::StopSideTracing { .. } => unreachable!(),
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
            thread::spawn(move || expect_start_tracing(&mt, &loc1))
                .join()
                .unwrap();
        }

        expect_start_tracing(&mt, &loc2);
        assert_eq!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::NoAction
        );
        expect_stop_tracing(&mt, &loc2);
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
                    mt.transition_control_point(loc),
                    TransitionControlPoint::NoAction
                );
            }

            expect_start_tracing(mt, loc);
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Tracing
            ));
            expect_stop_tracing(mt, loc);
            assert!(matches!(
                loc.hot_location().unwrap().lock().kind,
                HotLocationKind::Compiling
            ));
            loc.hot_location().unwrap().lock().kind =
                HotLocationKind::Compiled(Arc::new(CompiledTraceTesting::new()));
        }

        to_compiled(&mt, &loc1);
        to_compiled(&mt, &loc2);

        {
            let mt = Arc::clone(&mt);
            let loc1 = Arc::clone(&loc1);
            thread::spawn(move || expect_start_side_tracing(&mt, &loc1))
                .join()
                .unwrap();
        }

        expect_start_side_tracing(&mt, &loc2);
        assert!(matches!(
            dbg!(mt.transition_control_point(&loc1)),
            TransitionControlPoint::Execute(_)
        ));
        assert!(matches!(
            mt.transition_control_point(&loc2),
            TransitionControlPoint::StopSideTracing { .. }
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
        expect_start_tracing(&mt, &loc1);
        expect_stop_tracing(&mt, &loc1);
        loc1.hot_location().unwrap().lock().kind =
            HotLocationKind::Compiled(Arc::new(CompiledTraceTesting::new()));

        // If we transition `loc2` into `StartTracing`, then (for now) we should not execute the
        // trace for `loc1`, as another location is being traced and we don't want to trace the
        // execution of the trace!
        //
        // FIXME: this behaviour will need to change in the future:
        // https://github.com/ykjit/yk/issues/519
        expect_start_tracing(&mt, &loc2);
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::NoAction
        ));

        // But once we stop tracing for `loc2`, we should be able to execute the trace for `loc1`.
        expect_stop_tracing(&mt, &loc2);
        assert!(matches!(
            mt.transition_control_point(&loc1),
            TransitionControlPoint::Execute(_)
        ));
    }
}
