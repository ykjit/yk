//! The main end-user interface to the meta-tracing system.

use std::{
    assert_matches::debug_assert_matches,
    cell::RefCell,
    collections::{HashMap, HashSet},
    env,
    error::Error,
    ffi::c_void,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering},
    },
};

use parking_lot::Mutex;
#[cfg(not(all(feature = "yk_testing", not(test))))]
use parking_lot_core::SpinWait;

use crate::{
    aotsmp::{AOT_STACKMAPS, load_aot_stackmaps},
    compile::{
        CompilationError, CompiledTrace, Compiler, GuardId, default_compiler,
        jitc_yk::jit_ir::TraceEndFrame,
    },
    job_queue::{Job, JobQueue},
    location::{HotLocation, HotLocationKind, Location, TraceFailed},
    log::{
        Log, Verbosity,
        stats::{Stats, TimingState},
    },
    profile::{PlatformTraceProfiler, profiler_for_current_platform},
    trace::{AOTTraceIterator, TraceRecorder, Tracer, default_tracer},
};

// Emit a log entry with hot location debug information if present and support is compiled in.
macro_rules! yklog {
    ($logger:expr, $level:expr, $msg:expr, $opt_hl:expr) => {
        #[cfg(feature = "ykd")]
        if let Some(hl) = $opt_hl {
            $logger.log_with_hl_debug($level, $msg, hl);
        } else {
            $logger.log($level, $msg);
        }
        #[cfg(not(feature = "ykd"))]
        $logger.log($level, $msg);
    };
}

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

const DEFAULT_HOT_THRESHOLD: HotThreshold = 131;
const DEFAULT_SIDETRACE_THRESHOLD: HotThreshold = 5;
/// How often can a [HotLocation] or [Guard] lead to an error in tracing or compilation before we
/// give up trying to trace (or compile...) it?
const DEFAULT_TRACECOMPILATION_ERROR_THRESHOLD: TraceCompilationErrorThreshold = 5;
static REG64_SIZE: usize = 8;

thread_local! {
    /// This thread's [MTThread]. Do not access this directly: use [MTThread::with_borrow] or
    /// [MTThread::with_borrow_mut].
    static THREAD_MTTHREAD: RefCell<MTThread> = RefCell::new(MTThread::new());
    /// Is this thread tracing something? Do not access this directly: use [MTThread::is_tracing]
    /// and friends.
    static THREAD_IS_TRACING: RefCell<IsTracing> = const { RefCell::new(IsTracing::None) };
}

/// A meta-tracer. This is always passed around stored in an [Arc].
///
/// When you are finished with this meta-tracer, it is best to explicitly call [MT::shutdown] to
/// perform shutdown tasks (though the correctness of the system is not impacted if you do not call
/// this function).
pub struct MT {
    /// Have we been requested to shutdown this meta-tracer? In a sense this is merely advisory:
    /// since [MT] is contained within an [Arc], if we're able to read this value, then the
    /// meta-tracer is still working and it may go on doing so for an arbitrary period of time.
    /// However, it means that some "shutdown" activities, such as printing statistics and checking
    /// for failed compilation threads, have already occurred, and should not be repeated.
    shutdown: AtomicBool,
    hot_threshold: AtomicHotThreshold,
    sidetrace_threshold: AtomicHotThreshold,
    trace_failure_threshold: AtomicTraceCompilationErrorThreshold,
    /// The requested optimisation level, where 0 is the lowest possible level. How these levels
    /// are interpreted is up to a given optimiser. Default: 1.
    opt_level: AtomicU8,
    /// The queue of compiling jobs.
    job_queue: Arc<JobQueue>,
    /// The [Tracer] that should be used for creating future traces. Note that this might not be
    /// the same as the tracer(s) used to create past traces.
    tracer: Mutex<Arc<dyn Tracer>>,
    /// The [Compiler] that will be used for compiling future `IRTrace`s. Note that this might not
    /// be the same as the compiler(s) used to compile past `IRTrace`s.
    compiler: Mutex<Arc<dyn Compiler>>,
    /// A monotonically increasing integer that uniquely identifies each compiled trace.
    compiled_trace_id: AtomicU64,
    /// The currently available compiled traces. This is a [HashMap] because it is potentially a
    /// sparse mapping due to (1) (one day!) we might garbage collect traces (2) some
    /// [TraceId]s that we hand out are "lost" because a trace failed to compile.
    pub(crate) compiled_traces: Mutex<HashMap<TraceId, Arc<dyn CompiledTrace>>>,
    pub(crate) log: Log,
    pub(crate) stats: Stats,
    /// The trace profiler implementation to use.
    trace_profiler: Arc<dyn PlatformTraceProfiler>,
    /// Whether JIT compilation is enabled. Can be disabled with YK_JITC=none.
    jit_enabled: AtomicBool,
}

impl std::fmt::Debug for MT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MT")
    }
}

impl MT {
    pub(crate) fn trace_profiler(&self) -> &Arc<dyn PlatformTraceProfiler> {
        &self.trace_profiler
    }

    // Create a new meta-tracer instance. Arbitrarily many of these can be created, though there
    // are no guarantees as to whether they will share resources effectively or fairly.
    pub fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        load_aot_stackmaps();
        let opt_level = match env::var("YKD_OPT") {
            Ok(s) => s
                .parse::<u8>()
                .map_err(|e| format!("Invalid optimisation level '{s}': {e}"))?,
            Err(_) => 1,
        };
        let hot_threshold = match env::var("YK_HOT_THRESHOLD") {
            Ok(s) => s
                .parse::<HotThreshold>()
                .map_err(|e| format!("Invalid hot threshold '{s}': {e}"))?,
            Err(_) => DEFAULT_HOT_THRESHOLD,
        };
        let sidetrace_threshold = match env::var("YK_SIDETRACE_THRESHOLD") {
            Ok(s) => s
                .parse::<HotThreshold>()
                .map_err(|e| format!("Invalid sidetrace threshold '{s}': {e}"))?,
            Err(_) => DEFAULT_SIDETRACE_THRESHOLD,
        };
        let jit_enabled = match env::var("YK_JITC") {
            Ok(s) => s != "none",
            Err(_) => true, // Default to enabled
        };
        Ok(Arc::new(Self {
            shutdown: AtomicBool::new(false),
            hot_threshold: AtomicHotThreshold::new(hot_threshold),
            sidetrace_threshold: AtomicHotThreshold::new(sidetrace_threshold),
            trace_failure_threshold: AtomicTraceCompilationErrorThreshold::new(
                DEFAULT_TRACECOMPILATION_ERROR_THRESHOLD,
            ),
            opt_level: AtomicU8::new(opt_level),
            job_queue: JobQueue::new(),
            tracer: Mutex::new(default_tracer()?),
            compiler: Mutex::new(default_compiler()?),
            compiled_trace_id: AtomicU64::new(0),
            compiled_traces: Mutex::new(HashMap::new()),
            log: Log::new()?,
            stats: Stats::new(),
            trace_profiler: profiler_for_current_platform(),
            jit_enabled: AtomicBool::new(jit_enabled),
        }))
    }

    /// Put this meta-tracer into shutdown mode, panicking if any problems are discovered. This
    /// will perform actions such as printing summary statistics and checking whether any worker
    /// threads have caused an error. The best place to do this is likely to be on the main thread,
    /// though this is not mandatory.
    ///
    /// Note: this method does not stop all of the meta-tracer's activities. For example, -- but
    /// not only! -- other threads will continue compiling and executing traces.
    ///
    /// Only the first call of this method performs meaningful actions: any subsequent calls will
    /// note the previous shutdown and immediately return.
    pub fn shutdown(&self) {
        if !self.shutdown.swap(true, Ordering::Relaxed) {
            self.stats.timing_state(TimingState::None);
            self.stats.output();
            self.job_queue.shutdown();
        }
    }

    /// Check the integrity of the job queue: if any job queue thread has panicked, this function
    /// will itself panic. This should only be used for testing purposes.
    #[cfg(feature = "yk_testing")]
    pub(crate) fn check_job_queue_integrity(&self) {
        self.job_queue.check_integrity();
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

    /// Return the requested trace optimisation level, where 0 is the lowest possible level. How
    /// these levels are interpreted is up to a given optimiser.
    pub fn opt_level(self: &Arc<Self>) -> u8 {
        self.opt_level.load(Ordering::Relaxed)
    }

    /// Return whether JIT compilation is enabled. Can be controlled with YK_JITC.
    pub(crate) fn jit_enabled(self: &Arc<Self>) -> bool {
        self.jit_enabled.load(Ordering::Relaxed)
    }

    /// Return the unique ID for the next trace.
    pub(crate) fn next_trace_id(self: &Arc<Self>) -> TraceId {
        // Note: fetch_add is documented to wrap on overflow.
        let ctr_id = self.compiled_trace_id.fetch_add(1, Ordering::Relaxed);
        if ctr_id == u64::MAX {
            // OK, OK, technically we have 1 ID left that we could use, but if we've actually
            // managed to compile u64::MAX traces, it's probable that something's gone wrong.
            panic!("Ran out of trace IDs");
        }
        TraceId(ctr_id)
    }

    /// Add a compilation job for a root trace where:
    ///   * `hl_arc` is the [HotLocation] this compilation job is related to.
    ///   * `ctrid` is the trace ID to be given to the new compiled trace.
    ///   * `connector_tid` is the optional trace ID of the trace the new compiled trace will
    ///     connect to.
    fn queue_root_compile_job(
        self: &Arc<Self>,
        trace_iter: (Box<dyn AOTTraceIterator>, Box<[u8]>, Vec<String>),
        hl_arc: Arc<Mutex<HotLocation>>,
        trid: TraceId,
        connector_tid: Option<TraceId>,
        endframe: TraceEndFrame,
    ) {
        self.stats.trace_recorded_ok();

        let hl_arc_cl = Arc::clone(&hl_arc);
        let mt = Arc::clone(self);
        let main = move || {
            let compiler = {
                let lk = mt.compiler.lock();
                Arc::clone(&*lk)
            };
            mt.stats.timing_state(TimingState::Compiling);
            let connector_ctr = connector_tid.map(|x| Arc::clone(&mt.compiled_traces.lock()[&x]));
            match compiler.root_compile(
                Arc::clone(&mt),
                trace_iter.0,
                trid,
                Arc::clone(&hl_arc),
                trace_iter.1,
                trace_iter.2,
                connector_ctr,
                endframe,
            ) {
                Ok(ctr) => {
                    assert_eq!(ctr.ctrid(), trid);
                    mt.compiled_traces
                        .lock()
                        .insert(ctr.ctrid(), Arc::clone(&ctr));
                    let mut hl = hl_arc.lock();
                    debug_assert_matches!(hl.kind, HotLocationKind::Compiling(_));
                    hl.kind = HotLocationKind::Compiled(ctr);
                    mt.stats.trace_compiled_ok();
                    mt.job_queue.notify_success(trid);
                }
                Err(e) => {
                    mt.stats.trace_compiled_err();
                    let mut hl = hl_arc.lock();
                    debug_assert_matches!(hl.kind, HotLocationKind::Compiling(_));
                    if let TraceFailed::DontTrace = hl.tracecompilation_error(&mt) {
                        hl.kind = HotLocationKind::DontTrace;
                    } else {
                        hl.kind = HotLocationKind::Counting(0);
                    }
                    match e {
                        CompilationError::General(e) | CompilationError::LimitExceeded(e) => {
                            mt.log.log(
                                Verbosity::Warning,
                                &format!("trace-compilation-aborted: {e}"),
                            );
                        }
                        CompilationError::InternalError(e) => {
                            #[cfg(feature = "ykd")]
                            panic!("{e}");
                            #[cfg(not(feature = "ykd"))]
                            {
                                mt.log.log(
                                    Verbosity::Error,
                                    &format!("trace-compilation-aborted: {e}"),
                                );
                            }
                        }
                        CompilationError::ResourceExhausted(e) => {
                            mt.log
                                .log(Verbosity::Error, &format!("trace-compilation-aborted: {e}"));
                        }
                    }
                    mt.job_queue.notify_failure(&mt, trid);
                }
            }

            mt.stats.timing_state(TimingState::None);
        };

        let mt = Arc::clone(self);
        let failure = move || {
            let mut hl = hl_arc_cl.lock();
            debug_assert_matches!(hl.kind, HotLocationKind::Compiling(_));
            if let TraceFailed::DontTrace = hl.tracecompilation_error(&mt) {
                hl.kind = HotLocationKind::DontTrace;
            } else {
                hl.kind = HotLocationKind::Counting(0);
            }
            mt.job_queue.notify_failure(&mt, trid);
        };

        self.job_queue.push(
            self,
            Job::new(Box::new(main), connector_tid, Box::new(failure)),
        );
    }

    /// Add a compilation job for a sidetrace where: `hl_arc` is the [HotLocation] this compilation
    ///   * `hl_arc` is the [HotLocation] this compilation job is related to.
    ///   * `root_ctr` is the root [CompiledTrace].
    ///   * `parent_ctr` is the parent [CompiledTrace] of the side-trace that's about to be
    ///     compiled. Because side-traces can nest, this may or may not be the same [CompiledTrace]
    ///     as `root_ctr`.
    ///   * `guardid` is the ID of the guard in `parent_ctr` which failed.
    ///   * `connector_tid` is the optional trace ID of the trace the new compiled trace will
    ///     connect to.
    fn queue_sidetrace_compile_job(
        self: &Arc<Self>,
        trace_iter: (Box<dyn AOTTraceIterator>, Box<[u8]>, Vec<String>),
        hl_arc: Arc<Mutex<HotLocation>>,
        trid: TraceId,
        parent_ctr: Arc<dyn CompiledTrace>,
        gid: GuardId,
        connector_tid: TraceId,
        endframe: TraceEndFrame,
    ) {
        self.stats.trace_recorded_ok();
        let mt = Arc::clone(self);
        let parent_ctr_cl = Arc::clone(&parent_ctr);
        let main = move || {
            let compiler = {
                let lk = mt.compiler.lock();
                Arc::clone(&*lk)
            };
            mt.stats.timing_state(TimingState::Compiling);
            let target_ctr = Arc::clone(&mt.compiled_traces.lock()[&connector_tid]);
            // FIXME: Can we pass in the root trace address, root trace entry variable locations,
            // and the base stack-size from here, rather than spreading them out via
            // DeoptInfo/SideTraceInfo, and CompiledTrace?
            match compiler.sidetrace_compile(
                Arc::clone(&mt),
                trace_iter.0,
                trid,
                Arc::clone(&parent_ctr),
                gid,
                target_ctr,
                Arc::clone(&hl_arc),
                trace_iter.1,
                trace_iter.2,
                endframe,
            ) {
                Ok(ctr) => {
                    assert_eq!(ctr.ctrid(), trid);
                    mt.compiled_traces
                        .lock()
                        .insert(ctr.ctrid(), Arc::clone(&ctr));
                    parent_ctr.guard(gid).set_ctr(ctr, &parent_ctr, gid);
                    mt.stats.trace_compiled_ok();
                }
                Err(e) => {
                    parent_ctr.guard(gid).trace_or_compile_failed(&mt);
                    mt.stats.trace_compiled_err();
                    match e {
                        CompilationError::General(e) | CompilationError::LimitExceeded(e) => {
                            mt.log.log(
                                Verbosity::Warning,
                                &format!("sidetrace-compilation-aborted: {e}"),
                            );
                        }
                        CompilationError::InternalError(e) => {
                            #[cfg(feature = "ykd")]
                            panic!("{e}");
                            #[cfg(not(feature = "ykd"))]
                            {
                                mt.log.log(
                                    Verbosity::Error,
                                    &format!("sidetrace-compilation-aborted: {e}"),
                                );
                            }
                        }
                        CompilationError::ResourceExhausted(e) => {
                            mt.log.log(
                                Verbosity::Error,
                                &format!("sidetrace-compilation-aborted: {e}"),
                            );
                        }
                    }
                }
            }

            mt.stats.timing_state(TimingState::None);
        };

        let mt = Arc::clone(self);
        let failure = move || {
            parent_ctr_cl.guard(gid).trace_or_compile_failed(&mt);
        };
        self.job_queue.push(
            self,
            Job::new(Box::new(main), Some(connector_tid), Box::new(failure)),
        );
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn control_point(self: &Arc<Self>, loc: &Location, frameaddr: *mut c_void, smid: u64) {
        match self.transition_control_point(loc, frameaddr) {
            TransitionControlPoint::NoAction => (),
            TransitionControlPoint::AbortTracing(ak) => {
                let thread_tracer = MTThread::with_borrow_mut(|mtt| match mtt.pop_tstate() {
                    MTThreadState::Tracing { thread_tracer, .. } => thread_tracer,
                    _ => unreachable!(),
                });
                thread_tracer.stop().ok();
                MTThread::set_tracing(IsTracing::None);
                yklog!(
                    self.log,
                    Verbosity::Warning,
                    &format!("tracing-aborted: {ak}"),
                    loc.hot_location()
                );
                self.stats.timing_state(TimingState::OutsideYk);
            }
            TransitionControlPoint::Execute(ctr) => {
                yklog!(
                    self.log,
                    Verbosity::Execution,
                    "enter-jit-code",
                    loc.hot_location()
                );
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
                MTThread::with_borrow_mut(|mtt| {
                    mtt.push_tstate(MTThreadState::Executing {
                        mt: Arc::clone(self),
                    });
                });
                self.stats.timing_state(TimingState::JitExecuting);
                unsafe { __yk_exec_trace(frameaddr, rsp, trace_addr) };
            }
            TransitionControlPoint::StartTracing(hl, trid) => {
                self.start_tracing(frameaddr, loc, hl, trid);
            }
            TransitionControlPoint::StopTracing(trid, connector_tid) => {
                self.stop_tracing(frameaddr, loc, trid, connector_tid);
            }
            TransitionControlPoint::StopSideTracing {
                trid,
                gid,
                parent_ctr,
                connector_tid,
                start,
            } => {
                // Assuming no bugs elsewhere, the `unwrap`s cannot fail, because
                // `StartSideTracing` will have put a `Some` in the `Rc`.
                let (hl, thread_tracer, promotions, debug_strs) =
                    MTThread::with_borrow_mut(|mtt| match mtt.pop_tstate() {
                        MTThreadState::Tracing {
                            trid: _,
                            hl,
                            thread_tracer,
                            promotions,
                            debug_strs,
                            frameaddr: tracing_frameaddr,
                            seen_hls: _,
                            gtrace: _,
                        } => {
                            assert_eq!(frameaddr, tracing_frameaddr);
                            (hl, thread_tracer, promotions, debug_strs)
                        }
                        _ => unreachable!(),
                    });
                self.stats.timing_state(TimingState::TraceMapping);
                match thread_tracer.stop() {
                    Ok(utrace) => {
                        MTThread::set_tracing(IsTracing::None);
                        self.stats.timing_state(TimingState::None);
                        yklog!(
                            self.log,
                            Verbosity::Tracing,
                            "stop-tracing",
                            loc.hot_location()
                        );
                        self.queue_sidetrace_compile_job(
                            (utrace, promotions.into_boxed_slice(), debug_strs),
                            hl,
                            trid,
                            parent_ctr,
                            gid,
                            connector_tid,
                            TraceEndFrame::Same,
                        );
                        if start {
                            self.start_tracing(
                                frameaddr,
                                loc,
                                loc.hot_location_arc_clone().unwrap(),
                                connector_tid,
                            );
                        }
                    }
                    Err(e) => {
                        MTThread::set_tracing(IsTracing::None);
                        self.job_queue.notify_failure(self, trid);
                        parent_ctr.guard(gid).trace_or_compile_failed(self);
                        self.stats.trace_recorded_err();
                        yklog!(
                            self.log,
                            Verbosity::Warning,
                            &format!("stop-tracing-aborted: {e}"),
                            loc.hot_location()
                        );
                        self.stats.timing_state(TimingState::None);
                    }
                }
                self.stats.timing_state(TimingState::OutsideYk);
            }
        }
    }

    /// Start tracing at `loc` / `hl` (i.e. `hl` must be the [HotLocation] for `loc`) for a trace
    /// with ID `trid`.
    fn start_tracing(
        self: &Arc<Self>,
        frameaddr: *mut c_void,
        _loc: &Location,
        hl: Arc<Mutex<HotLocation>>,
        trid: TraceId,
    ) {
        self.stats
            .timing_state(crate::log::stats::TimingState::Tracing);
        yklog!(
            self.log,
            Verbosity::Tracing,
            "start-tracing",
            _loc.hot_location()
        );
        let tracer = {
            let lk = self.tracer.lock();
            Arc::clone(&*lk)
        };
        MTThread::set_tracing(IsTracing::Loop);
        MTThread::with_borrow_mut(|mtt| {
            match Arc::clone(&tracer).start_recorder() {
                Ok(tt) => {
                    mtt.push_tstate(MTThreadState::Tracing {
                        hl,
                        trid,
                        thread_tracer: tt,
                        promotions: Vec::new(),
                        debug_strs: Vec::new(),
                        frameaddr,
                        seen_hls: HashSet::new(),
                        gtrace: None,
                    });
                }
                Err(e) => {
                    MTThread::set_tracing(IsTracing::None);
                    mtt.pop_tstate();
                    // FIXME: start_recorder needs a way of signalling temporary errors.
                    #[cfg(tracer_hwt)]
                    match e.downcast::<hwtracer::HWTracerError>() {
                        Ok(e) => {
                            if let hwtracer::HWTracerError::Temporary(_) = *e {
                                let mut lk = hl.lock();
                                debug_assert_matches!(lk.kind, HotLocationKind::Tracing(_));
                                lk.tracecompilation_error(self);
                                // FIXME: This is stupidly brutal.
                                lk.kind = HotLocationKind::DontTrace;
                                drop(lk);
                                yklog!(
                                    self.log,
                                    Verbosity::Warning,
                                    "start-tracing-abort",
                                    _loc.hot_location()
                                );
                            } else {
                                todo!("{e:?}");
                            }
                        }
                        Err(e) => todo!("{e:?}"),
                    }
                    #[cfg(not(tracer_hwt))]
                    todo!("{e:?}");
                }
            }
        });
    }

    /// Stop tracing of the trace with id `trid` at `loc`. If `connector_tid` is `Some`, the
    /// resulting trace will be a connector trace.
    fn stop_tracing(
        self: &Arc<Self>,
        frameaddr: *mut c_void,
        _loc: &Location,
        trid: TraceId,
        connector_tid: Option<TraceId>,
    ) {
        // Assuming no bugs elsewhere, the `unwrap`s cannot fail, because `StartTracing`
        // will have put a `Some` in the `Rc`.
        let (hl, thread_tracer, promotions, debug_strs, endframe) =
            MTThread::with_borrow_mut(|mtt| match mtt.pop_tstate() {
                MTThreadState::Tracing {
                    trid: _,
                    hl,
                    thread_tracer,
                    promotions,
                    debug_strs,
                    frameaddr: tracing_frameaddr,
                    seen_hls: _,
                    gtrace: _,
                } => {
                    // If this assert fails then the code in `transition_control_point`,
                    // which rejects traces that end in another frame, didn't work.
                    let endframe = TraceEndFrame::from_frames(tracing_frameaddr, frameaddr);
                    (hl, thread_tracer, promotions, debug_strs, endframe)
                }
                _ => unreachable!(),
            });
        match thread_tracer.stop() {
            Ok(utrace) => {
                MTThread::set_tracing(IsTracing::None);
                self.stats.timing_state(TimingState::None);
                yklog!(
                    self.log,
                    Verbosity::Tracing,
                    "stop-tracing",
                    _loc.hot_location()
                );
                self.queue_root_compile_job(
                    (utrace, promotions.into_boxed_slice(), debug_strs),
                    hl,
                    trid,
                    connector_tid,
                    endframe,
                );
            }
            Err(e) => {
                MTThread::set_tracing(IsTracing::None);
                self.job_queue.notify_failure(self, trid);
                self.stats.timing_state(TimingState::None);
                self.stats.trace_recorded_err();
                yklog!(
                    self.log,
                    Verbosity::Warning,
                    &format!("stop-tracing-aborted: {e}"),
                    _loc.hot_location()
                );
            }
        }
        self.stats.timing_state(TimingState::OutsideYk);
    }

    /// Perform the next step to `loc` in the `Location` state-machine for a control point. If
    /// `loc` moves to the Compiled state, return a pointer to a [CompiledTrace] object.
    fn transition_control_point(
        self: &Arc<Self>,
        loc: &Location,
        frameaddr: *mut c_void,
    ) -> TransitionControlPoint {
        match MTThread::tracing_kind() {
            IsTracing::None => self.transition_control_point_not_tracing(loc),
            IsTracing::Loop => MTThread::with_borrow_mut(|mtt| {
                self.transition_control_point_tracing_loop(loc, frameaddr, mtt)
            }),
            IsTracing::Guard => MTThread::with_borrow_mut(|mtt| {
                self.transition_control_point_tracing_guard(loc, frameaddr, mtt)
            }),
        }
    }

    fn transition_control_point_not_tracing(
        self: &Arc<Self>,
        loc: &Location,
    ) -> TransitionControlPoint {
        // If JIT is disabled, don't increment the location count.
        if !self.jit_enabled() {
            return TransitionControlPoint::NoAction;
        }
        match loc.hot_location() {
            Some(hl) => {
                let mut lk;
                #[cfg(not(all(feature = "yk_testing", not(test))))]
                {
                    // Since this thread isn't tracing, it's not worth contending too much with
                    // other threads: we'd rather fall back to the interpreter and try again later.
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
                }
                #[cfg(all(feature = "yk_testing", not(test)))]
                {
                    // When `yk_testing` is enabled, the spinlock above introduces non-determinism,
                    // so instead we forcibly grab the lock.
                    lk = hl.lock();
                }

                match lk.kind {
                    HotLocationKind::Compiled(ref ctr) => {
                        TransitionControlPoint::Execute(Arc::clone(ctr))
                    }
                    HotLocationKind::Compiling(_) => TransitionControlPoint::NoAction,
                    HotLocationKind::Counting(c) => {
                        if c < self.hot_threshold() {
                            lk.kind = HotLocationKind::Counting(c + 1);
                            TransitionControlPoint::NoAction
                        } else {
                            let hl = loc.hot_location_arc_clone().unwrap();
                            let trid = self.next_trace_id();
                            lk.kind = HotLocationKind::Tracing(trid);
                            TransitionControlPoint::StartTracing(hl, trid)
                        }
                    }
                    HotLocationKind::Tracing(trid) => {
                        let hl = loc.hot_location_arc_clone().unwrap();
                        // This thread isn't tracing anything. Note that because we called
                        // `hot_location_arc_clone` above, the strong count of an `Arc`
                        // that's no longer being used by that thread will be 2.
                        if Arc::strong_count(&hl) == 2 {
                            // Another thread was tracing this location but it's terminated.
                            self.stats.trace_recorded_err();
                            self.job_queue.notify_failure(self, trid);
                            match lk.tracecompilation_error(self) {
                                TraceFailed::KeepTrying => {
                                    let trid = self.next_trace_id();
                                    lk.kind = HotLocationKind::Tracing(trid);
                                    TransitionControlPoint::StartTracing(hl, trid)
                                }
                                TraceFailed::DontTrace => {
                                    // FIXME: This is stupidly brutal.
                                    lk.kind = HotLocationKind::DontTrace;
                                    TransitionControlPoint::NoAction
                                }
                            }
                        } else {
                            // Another thread is tracing this location.
                            TransitionControlPoint::NoAction
                        }
                    }
                    HotLocationKind::DontTrace => TransitionControlPoint::NoAction,
                }
            }
            None => {
                match loc.inc_count() {
                    Some(x) => {
                        debug_assert!(self.hot_threshold() < HotThreshold::MAX);
                        if x < self.hot_threshold() + 1 {
                            TransitionControlPoint::NoAction
                        } else {
                            let trid = self.next_trace_id();
                            let hl = HotLocation {
                                kind: HotLocationKind::Tracing(trid),
                                tracecompilation_errors: 0,
                                debug_str: None,
                            };
                            if let Some(hl) = loc.count_to_hot_location(x, hl) {
                                TransitionControlPoint::StartTracing(hl, trid)
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
    }

    /// Perform a transition when we're tracing what we hope will be a loop trace (but which might
    /// end up being a coupler trace if we're unlucky).
    fn transition_control_point_tracing_loop(
        self: &Arc<Self>,
        loc: &Location,
        frameaddr: *mut c_void,
        mtt: &mut MTThread,
    ) -> TransitionControlPoint {
        let MTThreadState::Tracing {
            trid: tracing_trid,
            frameaddr: tracing_frameaddr,
            hl: tracing_hl,
            seen_hls,
            ..
        } = mtt.peek_mut_tstate()
        else {
            panic!()
        };
        if frameaddr != *tracing_frameaddr {
            // We traced into or out of a recursive interpreter call.
            let mut lk = tracing_hl.lock();
            match lk.kind {
                HotLocationKind::Compiled(_)
                | HotLocationKind::Compiling(_)
                | HotLocationKind::Counting(_)
                | HotLocationKind::DontTrace => (),
                HotLocationKind::Tracing(trid) => {
                    lk.kind = HotLocationKind::Compiling(trid);
                    return TransitionControlPoint::StopTracing(trid, None);
                }
            }
        }

        match loc.hot_location() {
            Some(hl) => {
                let mut akind = None;
                assert!(std::ptr::eq(frameaddr, *tracing_frameaddr));

                if let Some(x) = loc.hot_location().map(|x| x as *const Mutex<HotLocation>)
                    && !seen_hls.insert(x)
                {
                    // We have traced this location more than once.
                    akind = Some(AbortKind::Unrolled);
                }

                if let Some(akind) = akind {
                    self.stats.trace_recorded_err();
                    let mut lk = tracing_hl.lock();
                    match &lk.kind {
                        HotLocationKind::Compiled(_)
                        | HotLocationKind::Compiling(_)
                        | HotLocationKind::Counting(_)
                        | HotLocationKind::DontTrace => (),
                        HotLocationKind::Tracing(trid) => {
                            let trid = *trid;
                            match lk.tracecompilation_error(self) {
                                TraceFailed::KeepTrying => {
                                    lk.kind = HotLocationKind::Counting(0);
                                }
                                TraceFailed::DontTrace => {
                                    lk.kind = HotLocationKind::DontTrace;
                                }
                            }
                            self.job_queue.notify_failure(self, trid);
                        }
                    }

                    return TransitionControlPoint::AbortTracing(akind);
                }

                let mut lk = hl.lock();

                match lk.kind {
                    HotLocationKind::Compiled(_) | HotLocationKind::Compiling(_) => {
                        let compiled_trid = match lk.kind {
                            HotLocationKind::Compiled(ref ctr) => ctr.ctrid(),
                            HotLocationKind::Compiling(ref ctrid) => *ctrid,
                            _ => unreachable!(),
                        };
                        drop(lk);
                        let mut lk = tracing_hl.lock();
                        lk.kind = HotLocationKind::Compiling(*tracing_trid);
                        TransitionControlPoint::StopTracing(*tracing_trid, Some(compiled_trid))
                    }
                    HotLocationKind::Counting(_) => TransitionControlPoint::NoAction,
                    HotLocationKind::Tracing(hl_trid) => {
                        // This thread is tracing something...
                        if hl_trid != *tracing_trid {
                            // ...but it's not this Location.
                            TransitionControlPoint::NoAction
                        } else {
                            // ...and it's this location...
                            lk.kind = HotLocationKind::Compiling(hl_trid);
                            TransitionControlPoint::StopTracing(hl_trid, None)
                        }
                    }
                    HotLocationKind::DontTrace => TransitionControlPoint::NoAction,
                }
            }
            None => {
                let hl_ptr = match loc.inc_count() {
                    Some(count) => {
                        let hl = HotLocation {
                            kind: HotLocationKind::Counting(count),
                            tracecompilation_errors: 0,
                            debug_str: None,
                        };
                        loc.count_to_hot_location(count, hl)
                            .map(|x| Arc::as_ptr(&x))
                    }
                    None => loc.hot_location().map(|x| x as *const Mutex<HotLocation>),
                };
                if let Some(hl_ptr) = hl_ptr {
                    assert!(std::ptr::eq(frameaddr, *tracing_frameaddr));
                    if !seen_hls.insert(hl_ptr) {
                        return TransitionControlPoint::AbortTracing(AbortKind::Unrolled);
                    }
                }
                TransitionControlPoint::NoAction
            }
        }
    }

    /// Perform a transition when we're tracing a guard trace.
    fn transition_control_point_tracing_guard(
        self: &Arc<Self>,
        loc: &Location,
        frameaddr: *mut c_void,
        mtt: &mut MTThread,
    ) -> TransitionControlPoint {
        let MTThreadState::Tracing {
            trid: tracing_trid,
            frameaddr: tracing_frameaddr,
            hl: tracing_hl,
            seen_hls,
            gtrace,
            ..
        } = mtt.peek_mut_tstate()
        else {
            panic!()
        };

        match loc.hot_location() {
            Some(hl) => {
                if !std::ptr::eq(frameaddr, *tracing_frameaddr) {
                    // We're tracing but no longer in the frame we started in, so we
                    // need to stop tracing and report the original [HotLocation] as
                    // having failed to trace properly.
                    self.stats.trace_recorded_err();
                    let mut lk = tracing_hl.lock();
                    match &lk.kind {
                        HotLocationKind::Compiled(_)
                        | HotLocationKind::Compiling(_)
                        | HotLocationKind::Counting(_)
                        | HotLocationKind::DontTrace => (),
                        HotLocationKind::Tracing(trid) => {
                            let trid = *trid;
                            match lk.tracecompilation_error(self) {
                                TraceFailed::KeepTrying => {
                                    lk.kind = HotLocationKind::Counting(0);
                                }
                                TraceFailed::DontTrace => {
                                    lk.kind = HotLocationKind::DontTrace;
                                }
                            }
                            self.job_queue.notify_failure(self, trid);
                        }
                    }

                    return TransitionControlPoint::AbortTracing(AbortKind::OutOfFrame);
                }

                let mut lk = hl.lock();
                match lk.kind {
                    HotLocationKind::Compiled(_)
                    | HotLocationKind::Compiling(_)
                    | HotLocationKind::Tracing(_) => {
                        let connector_tid = match lk.kind {
                            HotLocationKind::Compiled(ref ctr) => ctr.ctrid(),
                            HotLocationKind::Compiling(ref ctrid) => *ctrid,
                            HotLocationKind::Tracing(tid) => tid,
                            _ => unreachable!(),
                        };
                        drop(lk);
                        let Some((parent_ctr, gid)) = gtrace else {
                            panic!()
                        };
                        let gid = *gid;
                        let parent_ctr = Arc::clone(parent_ctr);
                        TransitionControlPoint::StopSideTracing {
                            trid: *tracing_trid,
                            gid,
                            parent_ctr,
                            connector_tid,
                            start: false,
                        }
                    }
                    HotLocationKind::Counting(_) => {
                        let next_tid = self.next_trace_id();
                        lk.kind = HotLocationKind::Tracing(next_tid);
                        drop(lk);
                        let Some((parent_ctr, gid)) = gtrace else {
                            panic!()
                        };
                        let gid = *gid;
                        let parent_ctr = Arc::clone(parent_ctr);
                        TransitionControlPoint::StopSideTracing {
                            trid: *tracing_trid,
                            gid,
                            parent_ctr,
                            connector_tid: next_tid,
                            start: true,
                        }
                    }
                    HotLocationKind::DontTrace => TransitionControlPoint::NoAction,
                }
            }
            None => {
                let hl_ptr = match loc.inc_count() {
                    Some(count) => {
                        let hl = HotLocation {
                            kind: HotLocationKind::Counting(count),
                            tracecompilation_errors: 0,
                            debug_str: None,
                        };
                        loc.count_to_hot_location(count, hl)
                            .map(|x| Arc::as_ptr(&x))
                    }
                    None => loc.hot_location().map(|x| x as *const Mutex<HotLocation>),
                };
                if let Some(hl_ptr) = hl_ptr {
                    if !std::ptr::eq(frameaddr, *tracing_frameaddr) {
                        // We're tracing but no longer in the frame we started in, so we
                        // need to stop tracing and report the original [HotLocation] as
                        // having failed to trace properly.
                        return TransitionControlPoint::AbortTracing(AbortKind::OutOfFrame);
                    }
                    if !seen_hls.insert(hl_ptr) {
                        return TransitionControlPoint::AbortTracing(AbortKind::Unrolled);
                    }
                }
                TransitionControlPoint::NoAction
            }
        }
    }

    /// Perform the next step in the guard failure statemachine.
    pub(crate) fn transition_guard_failure(
        self: &Arc<Self>,
        parent_ctr: Arc<dyn CompiledTrace>,
        gid: GuardId,
    ) -> TransitionGuardFailure {
        if parent_ctr.guard(gid).inc_failed(self) {
            if let Some(hl) = parent_ctr.hl().upgrade() {
                // This thread should not be tracing anything.
                debug_assert!(!MTThread::is_tracing());
                TransitionGuardFailure::StartSideTracing(hl, self.next_trace_id())
            } else {
                // The parent [HotLocation] has been garbage collected.
                TransitionGuardFailure::NoAction
            }
        } else {
            // We're side-tracing
            TransitionGuardFailure::NoAction
        }
    }

    /// Inform this `MT` instance that `deopt` has occurred: this updates the stack of
    /// [MTThreadState]s.
    pub(crate) fn deopt(self: &Arc<Self>) {
        loop {
            let st = MTThread::with_borrow_mut(|mtt| mtt.pop_tstate());
            match st {
                MTThreadState::Interpreting => todo!(),
                MTThreadState::Tracing {
                    trid,
                    hl,
                    thread_tracer,
                    gtrace,
                    ..
                } => {
                    let mut lk = hl.lock();
                    match &lk.kind {
                        HotLocationKind::Compiled(_) => {
                            if let Some((parent_ctr, gidx)) = gtrace {
                                // An inner trace has started side-tracing, then returned to the
                                // outer trace, which deopts.
                                let mt = Arc::clone(self);
                                parent_ctr.guard(gidx).trace_or_compile_failed(&mt);
                                mt.stats.trace_recorded_err();
                                self.job_queue.notify_failure(self, trid);
                            } else {
                                todo!();
                            }
                        }
                        HotLocationKind::Compiling(_) => {
                            todo!();
                        }
                        HotLocationKind::Counting(_) => {
                            todo!();
                        }
                        HotLocationKind::DontTrace => {}
                        HotLocationKind::Tracing(trid) => {
                            let trid = *trid;
                            match lk.tracecompilation_error(self) {
                                TraceFailed::KeepTrying => {
                                    lk.kind = HotLocationKind::Counting(0);
                                }
                                TraceFailed::DontTrace => {
                                    lk.kind = HotLocationKind::DontTrace;
                                }
                            }
                            self.job_queue.notify_failure(self, trid);
                        }
                    }
                    drop(lk);
                    thread_tracer.stop().ok();
                    MTThread::set_tracing(IsTracing::None);
                    yklog!(
                        self.log,
                        Verbosity::Warning,
                        &format!("tracing-aborted: {}", AbortKind::BackIntoExecution),
                        Some(&*hl)
                    );
                }
                MTThreadState::Executing { .. } => return,
            }
        }
    }

    /// Inform this meta-tracer that guard `gid` has failed.
    ///
    // FIXME: Don't side trace the last guard of a side-trace as this guard always fails.
    // FIXME: Don't side-trace after switch instructions: not every guard failure is equal
    // and a trace compiled for case A won't work for case B.
    pub(crate) fn guard_failure(
        self: &Arc<Self>,
        parent: Arc<dyn CompiledTrace>,
        gid: GuardId,
        frameaddr: *mut c_void,
    ) {
        match self.transition_guard_failure(Arc::clone(&parent), gid) {
            TransitionGuardFailure::NoAction => {
                self.stats
                    .timing_state(crate::log::stats::TimingState::OutsideYk);
            }
            TransitionGuardFailure::StartSideTracing(hl, trid) => {
                self.stats
                    .timing_state(crate::log::stats::TimingState::Tracing);
                yklog!(
                    self.log,
                    Verbosity::Tracing,
                    "start-side-tracing",
                    Some(&*hl)
                );
                let tracer = {
                    let lk = self.tracer.lock();
                    Arc::clone(&*lk)
                };
                MTThread::set_tracing(IsTracing::Guard);
                MTThread::with_borrow_mut(|mtt| match Arc::clone(&tracer).start_recorder() {
                    Ok(tt) => mtt.push_tstate(MTThreadState::Tracing {
                        trid,
                        hl,
                        thread_tracer: tt,
                        promotions: Vec::new(),
                        debug_strs: Vec::new(),
                        frameaddr,
                        seen_hls: HashSet::new(),
                        gtrace: Some((parent, gid)),
                    }),
                    Err(e) => {
                        MTThread::set_tracing(IsTracing::None);
                        todo!("{e:?}");
                    }
                });
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
#[unsafe(no_mangle)]
unsafe extern "C" fn __yk_exec_trace(
    frameaddr: *const c_void,
    rsp: *const c_void,
    trace: *const c_void,
) {
    std::arch::naked_asm!(
        // Write trace address over the return address of the control point call.
        "mov [rsi-8], rdx",
        "ret",
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
        /// The ID of this trace (and which will be -- if everything is successful! -- the ID of
        /// the subsequent [CompiledTrace]).
        trid: TraceId,
        /// Which [Location]s have we seen so far in this trace? If we see a [Location] twice
        /// then we know we've got an undesirable trace (e.g. we started tracing an outer loop and
        /// have started to unroll an inner loop).
        ///
        /// Tracking [Location]s directly is tricky as they have no inherent ID. To solve that, for
        /// the time being we force every `Location` that we encounter in a trace to become a
        /// [HotLocation] (with kind [HotLocationKind::Counting]) if it is not already. We can then
        /// use the (unmoving) pointer to a [HotLocation]'s inner [Mutex] as an ID.
        seen_hls: HashSet<*const Mutex<HotLocation>>,
        /// The [HotLocation] the trace will end at. For a top-level trace, this will be the same
        /// [HotLocation] the trace started at; for a side-trace, tracing started elsewhere.
        hl: Arc<Mutex<HotLocation>>,
        /// What tracer is being used to record this trace? Needed for trace mapping.
        thread_tracer: Box<dyn TraceRecorder>,
        /// Records the content of data recorded via `yk_promote_*` and `yk_idempotent_promote_*`.
        promotions: Vec<u8>,
        /// Records the content of data recorded via `yk_debug_str`.
        debug_strs: Vec<String>,
        /// The `frameaddr` when tracing started. This allows us to tell if we're finishing tracing
        /// at the same point that we started.
        frameaddr: *mut c_void,
        /// If we're tracing from a guard, this will be `Some(parent_ctr,
        /// guard_idx_in_parent_ctr)`; for loop traces this will be `None`.
        gtrace: Option<(Arc<dyn CompiledTrace>, GuardId)>,
    },
    Executing {
        mt: Arc<MT>,
    },
}

impl std::fmt::Debug for MTThreadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interpreting => write!(f, "Interpreting"),
            Self::Tracing { .. } => write!(f, "Tracing"),
            Self::Executing { .. } => write!(f, "Executing"),
        }
    }
}

/// Meta-tracer per-thread state. Note that this struct is neither `Send` nor `Sync`: it can only
/// be accessed from within a single thread.
pub struct MTThread {
    /// Where in the "interpreting/tracing/executing" is this thread? This `Vec` always has at
    /// least 1 element in it. It should not be access directly: use the `*_tstate` methods.
    tstate: Vec<MTThreadState>,
    // Raw pointers are neither send nor sync.
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl MTThread {
    fn new() -> Self {
        MTThread {
            tstate: vec![MTThreadState::Interpreting],
            _dont_send_or_sync_me: PhantomData,
        }
    }

    /// Is this thread currently tracing something?
    pub(crate) fn is_tracing() -> bool {
        THREAD_IS_TRACING.with(|x| *x.borrow() != IsTracing::None)
    }

    /// What kind of tracing (if any!) is this thread undertaking?
    fn tracing_kind() -> IsTracing {
        THREAD_IS_TRACING.with(|x| x.borrow().clone())
    }

    /// Mark this thread as currently tracing something.
    fn set_tracing(kind: IsTracing) {
        THREAD_IS_TRACING.with(|x| *x.borrow_mut() = kind);
    }

    /// Call `f` with a `&` reference to this thread's [MTThread] instance.
    ///
    /// # Panics
    ///
    /// For the same reasons as [thread::local::LocalKey::with].
    pub(crate) fn with_borrow<F, R>(f: F) -> R
    where
        F: FnOnce(&MTThread) -> R,
    {
        THREAD_MTTHREAD.with_borrow(|mtt| f(mtt))
    }

    /// Call `f` with a `&mut` reference to this thread's [MTThread] instance.
    ///
    /// # Panics
    ///
    /// For the same reasons as [thread::local::LocalKey::with].
    pub(crate) fn with_borrow_mut<F, R>(f: F) -> R
    where
        F: FnOnce(&mut MTThread) -> R,
    {
        THREAD_MTTHREAD.with_borrow_mut(|mtt| f(mtt))
    }

    /// Return a reference to the [CompiledTrace] with ID `ctrid`.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    pub(crate) fn compiled_trace(&self, ctrid: TraceId) -> Arc<dyn CompiledTrace> {
        for tstate in self.tstate.iter().rev() {
            if let MTThreadState::Executing { mt } = tstate {
                return Arc::clone(&mt.compiled_traces.lock()[&ctrid]);
            }
        }
        panic!();
    }

    /// Return a mutable reference to the last element on the stack of [MTThreadState]s.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    fn peek_mut_tstate(&mut self) -> &mut MTThreadState {
        self.tstate.last_mut().unwrap()
    }

    /// Pop the last element from the stack of [MTThreadState]s and return it.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    fn pop_tstate(&mut self) -> MTThreadState {
        debug_assert!(self.tstate.len() > 1);
        self.tstate.pop().unwrap()
    }

    /// Push `tstate` to the end of the stack of [MTThreadState]s.
    fn push_tstate(&mut self, tstate: MTThreadState) {
        self.tstate.push(tstate);
    }

    /// Records `val` as a value to be promoted. Returns `true` if either: no trace is being
    /// recorded; or recording the promotion succeeded.
    ///
    /// If `false` is returned, the current trace is unable to record the promotion successfully
    /// and further calls are probably pointless, though they will not cause the tracer to enter
    /// undefined behaviour territory.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    pub(crate) fn promote_i32(&mut self, val: i32) -> bool {
        if let MTThreadState::Tracing { promotions, .. } = self.peek_mut_tstate() {
            promotions.extend_from_slice(&val.to_ne_bytes());
        }
        true
    }

    /// Records `val` as a value to be promoted. Returns `true` if either: no trace is being
    /// recorded; or recording the promotion succeeded.
    ///
    /// If `false` is returned, the current trace is unable to record the promotion successfully
    /// and further calls are probably pointless, though they will not cause the tracer to enter
    /// undefined behaviour territory.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    pub(crate) fn promote_u32(&mut self, val: u32) -> bool {
        if let MTThreadState::Tracing { promotions, .. } = self.peek_mut_tstate() {
            promotions.extend_from_slice(&val.to_ne_bytes());
        }
        true
    }

    /// Records `val` as a value to be promoted. Returns `true` if either: no trace is being
    /// recorded; or recording the promotion succeeded.
    ///
    /// If `false` is returned, the current trace is unable to record the promotion successfully
    /// and further calls are probably pointless, though they will not cause the tracer to enter
    /// undefined behaviour territory.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    pub(crate) fn promote_i64(&mut self, val: i64) -> bool {
        if let MTThreadState::Tracing { promotions, .. } = self.peek_mut_tstate() {
            promotions.extend_from_slice(&val.to_ne_bytes());
        }
        true
    }

    /// Records `val` as a value to be promoted. Returns `true` if either: no trace is being
    /// recorded; or recording the promotion succeeded.
    ///
    /// If `false` is returned, the current trace is unable to record the promotion successfully
    /// and further calls are probably pointless, though they will not cause the tracer to enter
    /// undefined behaviour territory.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    pub(crate) fn promote_usize(&mut self, val: usize) -> bool {
        if let MTThreadState::Tracing { promotions, .. } = self.peek_mut_tstate() {
            promotions.extend_from_slice(&val.to_ne_bytes());
        }
        true
    }

    /// Record a debug string.
    ///
    /// # Panics
    ///
    /// If the stack is empty. There should always be at least one element on the stack, so a panic
    /// here means that something has gone wrong elsewhere.
    pub(crate) fn insert_debug_str(&mut self, msg: String) -> bool {
        if let MTThreadState::Tracing { debug_strs, .. } = self.peek_mut_tstate() {
            debug_strs.push(msg);
        }
        true
    }
}

#[derive(Clone, Eq, PartialEq)]
enum IsTracing {
    None,
    Loop,
    Guard,
}

/// What action should a caller of [MT::transition_control_point] take?
#[derive(Debug)]
enum TransitionControlPoint {
    NoAction,
    AbortTracing(AbortKind),
    Execute(Arc<dyn CompiledTrace>),
    /// Start tracing: in a sense the `Arc<Mutex<HotLocation>>` could be seen as an optimisation
    /// because it can always be derived from the [Location] that was encountered. However, we also
    /// use the `Arc` to detect tracing issues in other threads, and we need to keep it alive for
    /// the duration of the transition calls for that to work properly.
    StartTracing(Arc<Mutex<HotLocation>>, TraceId),
    /// Stop tracing. If `Option<TraceId>` is not-`None`, we have a connector trace.
    StopTracing(TraceId, Option<TraceId>),
    /// Stop side tracing.
    StopSideTracing {
        trid: TraceId,
        gid: GuardId,
        parent_ctr: Arc<dyn CompiledTrace>,
        connector_tid: TraceId,
        // Should a new trace be immediately started after the guard trace?
        start: bool,
    },
}

/// Why did we abort tracing?
#[derive(Debug)]
enum AbortKind {
    /// While tracing we fell back from an interpreter to a JIT frame.
    BackIntoExecution,
    /// Tracing continued while the interpreter frame address changed.
    OutOfFrame,
    /// We unrolled a loop (i.e. we traced a [Location] more than once).
    Unrolled,
}

impl std::fmt::Display for AbortKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            AbortKind::BackIntoExecution => write!(f, "tracing continued into a JIT frame"),
            AbortKind::OutOfFrame => write!(f, "tracing went outside of starting frame"),
            AbortKind::Unrolled => write!(f, "tracing unrolled a loop"),
        }
    }
}

/// What action should a caller of [MT::transition_guard_failure] take?
#[derive(Debug)]
pub(crate) enum TransitionGuardFailure {
    NoAction,
    StartSideTracing(Arc<Mutex<HotLocation>>, TraceId),
}

/// The unique identifier of a trace.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TraceId(u64);

impl TraceId {
    /// Create a [TraceId] from a `u64`. This function should only be used by deopt
    /// modules, which have to take a value from a register.
    pub(crate) fn from_u64(ctrid: u64) -> Self {
        Self(ctrid)
    }

    /// Create a dummy [TraceId] for testing purposes. Note: duplicate IDs can, and
    /// probably will, be returned!
    #[cfg(test)]
    pub(crate) fn testing() -> Self {
        Self(0)
    }

    /// Return a `u64` which can later be turned back into a `TraceId`. This should only be
    /// used by code gen when creating guard/deopt code.
    pub(crate) fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use crate::{
        compile::{CompiledTraceTestingBasicTransitions, CompiledTraceTestingMinimal},
        trace::TraceRecorderError,
    };
    use std::{assert_matches::assert_matches, hint::black_box, ptr, thread};
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
                    TransitionControlPoint::StartTracing(_, _),
                    TransitionControlPoint::StartTracing(_, _),
                ) => true,
                (x, y) => todo!("{:?} {:?}", x, y),
            }
        }
    }

    #[derive(Debug)]
    struct DummyTraceRecorder;

    impl TraceRecorder for DummyTraceRecorder {
        fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
            todo!();
        }
    }

    fn expect_start_tracing(mt: &Arc<MT>, loc: &Location) {
        let TransitionControlPoint::StartTracing(hl, trid) =
            mt.transition_control_point(loc, ptr::null_mut())
        else {
            panic!()
        };
        MTThread::set_tracing(IsTracing::Loop);
        MTThread::with_borrow_mut(|mtt| {
            mtt.push_tstate(MTThreadState::Tracing {
                trid,
                hl,
                thread_tracer: Box::new(DummyTraceRecorder),
                promotions: Vec::new(),
                debug_strs: Vec::new(),
                frameaddr: ptr::null_mut(),
                seen_hls: HashSet::new(),
                gtrace: None,
            });
        });
    }

    fn expect_stop_tracing(mt: &Arc<MT>, loc: &Location) {
        let TransitionControlPoint::StopTracing(_, _) =
            mt.transition_control_point(loc, ptr::null_mut())
        else {
            panic!()
        };
        MTThread::set_tracing(IsTracing::None);
        MTThread::with_borrow_mut(|mtt| {
            mtt.pop_tstate();
            mtt.push_tstate(MTThreadState::Interpreting);
        });
    }

    fn expect_start_side_tracing(mt: &Arc<MT>, ctr: Arc<dyn CompiledTrace>) {
        let TransitionGuardFailure::StartSideTracing(hl, trid) =
            mt.transition_guard_failure(Arc::clone(&ctr), GuardId::from(0))
        else {
            panic!()
        };
        MTThread::set_tracing(IsTracing::Guard);
        MTThread::with_borrow_mut(|mtt| {
            mtt.push_tstate(MTThreadState::Tracing {
                trid,
                hl,
                thread_tracer: Box::new(DummyTraceRecorder),
                promotions: Vec::new(),
                debug_strs: Vec::new(),
                frameaddr: ptr::null_mut(),
                seen_hls: HashSet::new(),
                gtrace: Some((ctr, GuardId::from(0))),
            });
        });
    }

    #[test]
    fn basic_transitions() {
        let hot_thrsh = 5;
        let mt = MT::new().unwrap();
        mt.set_hot_threshold(hot_thrsh);
        mt.set_sidetrace_threshold(1);
        let loc = Location::new();
        for i in 0..mt.hot_threshold() {
            assert_eq!(
                mt.transition_control_point(&loc, ptr::null_mut()),
                TransitionControlPoint::NoAction
            );
            assert_eq!(loc.count(), Some(i + 1));
        }
        expect_start_tracing(&mt, &loc);
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing(_)
        ));
        expect_stop_tracing(&mt, &loc);
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling(_)
        ));
        let ctr = Arc::new(CompiledTraceTestingBasicTransitions::new(Arc::downgrade(
            &loc.hot_location_arc_clone().unwrap(),
        )));
        loc.hot_location().unwrap().lock().kind = HotLocationKind::Compiled(ctr.clone());
        assert!(matches!(
            mt.transition_control_point(&loc, ptr::null_mut()),
            TransitionControlPoint::Execute(_)
        ));
        expect_start_side_tracing(&mt, ctr);

        match mt.transition_control_point(&loc, ptr::null_mut()) {
            TransitionControlPoint::StopSideTracing { .. } => {
                MTThread::set_tracing(IsTracing::None);
                MTThread::with_borrow_mut(|mtt| {
                    mtt.pop_tstate();
                    mtt.push_tstate(MTThreadState::Interpreting);
                });
                assert!(matches!(
                    loc.hot_location().unwrap().lock().kind,
                    HotLocationKind::Compiled(_)
                ));
            }
            _ => unreachable!(),
        }
        assert!(matches!(
            mt.transition_control_point(&loc, ptr::null_mut()),
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
                        mt.transition_control_point(&loc, ptr::null_mut()),
                        TransitionControlPoint::NoAction
                    );
                    let c1 = loc.count();
                    assert!(c1.is_some());
                    assert_eq!(
                        mt.transition_control_point(&loc, ptr::null_mut()),
                        TransitionControlPoint::NoAction
                    );
                    let c2 = loc.count();
                    assert!(c2.is_some());
                    assert_eq!(
                        mt.transition_control_point(&loc, ptr::null_mut()),
                        TransitionControlPoint::NoAction
                    );
                    let c3 = loc.count();
                    assert!(c3.is_some());
                    assert_eq!(
                        mt.transition_control_point(&loc, ptr::null_mut()),
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
            match mt.transition_control_point(&loc, ptr::null_mut()) {
                TransitionControlPoint::NoAction => (),
                TransitionControlPoint::StartTracing(hl, trid) => {
                    MTThread::set_tracing(IsTracing::Loop);
                    MTThread::with_borrow_mut(|mtt| {
                        mtt.push_tstate(MTThreadState::Tracing {
                            trid,
                            hl,
                            thread_tracer: Box::new(DummyTraceRecorder),
                            promotions: Vec::new(),
                            debug_strs: Vec::new(),
                            frameaddr: ptr::null_mut(),
                            seen_hls: HashSet::new(),
                            gtrace: None,
                        });
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
                mt.transition_control_point(&loc, ptr::null_mut()),
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
                HotLocationKind::Tracing(_)
            ));
            assert_eq!(
                loc.hot_location().unwrap().lock().tracecompilation_errors,
                i
            );
        }

        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing(_)
        ));
        assert_eq!(
            mt.transition_control_point(&loc, ptr::null_mut()),
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
                mt.transition_control_point(&loc, ptr::null_mut()),
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
                HotLocationKind::Tracing(_)
            ));
            assert_eq!(
                loc.hot_location().unwrap().lock().tracecompilation_errors,
                i
            );
        }

        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing(_)
        ));
        // Start tracing again...
        expect_start_tracing(&mt, &loc);
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing(_)
        ));
        // ...and this time let tracing succeed.
        expect_stop_tracing(&mt, &loc);
        // If tracing succeeded, we'll now be in the Compiling state.
        assert!(matches!(
            loc.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling(_)
        ));
    }

    #[test]
    fn locations_can_fail_multiple_times() {
        // Test that a location can fail tracing/compiling multiple times before we give up.

        let hot_thrsh = 5;
        let mt = MT::new().unwrap();
        mt.set_hot_threshold(hot_thrsh);
        let loc = Location::new();
        for i in 0..mt.hot_threshold() {
            assert_eq!(
                mt.transition_control_point(&loc, ptr::null_mut()),
                TransitionControlPoint::NoAction
            );
            assert_eq!(loc.count(), Some(i + 1));
        }
        expect_start_tracing(&mt, &loc);
        expect_stop_tracing(&mt, &loc);

        for _ in 0..mt.trace_failure_threshold() {
            assert_matches!(
                loc.hot_location()
                    .unwrap()
                    .lock()
                    .tracecompilation_error(&mt),
                TraceFailed::KeepTrying
            );
        }
        assert_matches!(
            loc.hot_location()
                .unwrap()
                .lock()
                .tracecompilation_error(&mt),
            TraceFailed::DontTrace
        );
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
                mt.transition_control_point(&loc1, ptr::null_mut()),
                TransitionControlPoint::NoAction
            );
            assert_eq!(
                mt.transition_control_point(&loc2, ptr::null_mut()),
                TransitionControlPoint::NoAction
            );
        }
        expect_start_tracing(&mt, &loc1);
        assert_eq!(
            mt.transition_control_point(&loc2, ptr::null_mut()),
            TransitionControlPoint::NoAction
        );
        assert!(matches!(
            loc1.hot_location().unwrap().lock().kind,
            HotLocationKind::Tracing(_)
        ));
        assert_eq!(loc2.count(), None);
        assert_matches!(
            loc2.hot_location().unwrap().lock().kind,
            HotLocationKind::Counting(6)
        );
        expect_stop_tracing(&mt, &loc1);
        assert!(matches!(
            loc1.hot_location().unwrap().lock().kind,
            HotLocationKind::Compiling(_)
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
                    match mt.transition_control_point(&loc, ptr::null_mut()) {
                        TransitionControlPoint::NoAction => (),
                        TransitionControlPoint::AbortTracing(_) => panic!(),
                        TransitionControlPoint::Execute(_) => (),
                        TransitionControlPoint::StartTracing(hl, trid) => {
                            num_starts.fetch_add(1, Ordering::Relaxed);
                            MTThread::set_tracing(IsTracing::Loop);
                            MTThread::with_borrow_mut(|mtt| {
                                mtt.push_tstate(MTThreadState::Tracing {
                                    trid,
                                    hl,
                                    thread_tracer: Box::new(DummyTraceRecorder),
                                    promotions: Vec::new(),
                                    debug_strs: Vec::new(),
                                    frameaddr: ptr::null_mut(),
                                    seen_hls: HashSet::new(),
                                    gtrace: None,
                                });
                            });
                            assert!(matches!(
                                loc.hot_location().unwrap().lock().kind,
                                HotLocationKind::Tracing(_)
                            ));
                            expect_stop_tracing(&mt, &loc);
                            assert!(matches!(
                                loc.hot_location().unwrap().lock().kind,
                                HotLocationKind::Compiling(_)
                            ));
                            assert_eq!(
                                mt.transition_control_point(&loc, ptr::null_mut()),
                                TransitionControlPoint::NoAction
                            );
                            assert!(matches!(
                                loc.hot_location().unwrap().lock().kind,
                                HotLocationKind::Compiling(_)
                            ));
                            loc.hot_location().unwrap().lock().kind = HotLocationKind::Compiled(
                                Arc::new(CompiledTraceTestingMinimal::new()),
                            );
                            loop {
                                if let TransitionControlPoint::Execute(_) =
                                    mt.transition_control_point(&loc, ptr::null_mut())
                                {
                                    break;
                                }
                            }
                            break;
                        }
                        TransitionControlPoint::StopTracing(_, _)
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
                mt.transition_control_point(&loc1, ptr::null_mut()),
                TransitionControlPoint::NoAction
            );
            assert_eq!(
                mt.transition_control_point(&loc2, ptr::null_mut()),
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
            mt.transition_control_point(&loc1, ptr::null_mut()),
            TransitionControlPoint::NoAction
        );
        expect_stop_tracing(&mt, &loc2);
    }

    #[bench]
    fn bench_single_threaded_control_point(b: &mut Bencher) {
        let mt = MT::new().unwrap();
        let loc = Location::new();
        b.iter(|| {
            for _ in 0..100000 {
                black_box(mt.transition_control_point(&loc, ptr::null_mut()));
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
                        black_box(mt.transition_control_point(&loc, ptr::null_mut()));
                    }
                }));
            }
            for t in thrs {
                t.join().unwrap();
            }
        });
    }

    #[test]
    fn traces_can_be_executed_during_tracing() {
        let mt = Arc::new(MT::new().unwrap());
        mt.set_hot_threshold(0);
        let loc1 = Location::new();
        let loc2 = Location::new();

        // Get `loc1` to the point where there's a compiled trace for it.
        expect_start_tracing(&mt, &loc1);
        expect_stop_tracing(&mt, &loc1);
        loc1.hot_location().unwrap().lock().kind =
            HotLocationKind::Compiled(Arc::new(CompiledTraceTestingMinimal::new()));

        expect_start_tracing(&mt, &loc2);
        assert_matches!(
            mt.transition_control_point(&loc1, ptr::null_mut()),
            TransitionControlPoint::StopTracing(_, _)
        );

        expect_stop_tracing(&mt, &loc2);
        assert_matches!(
            mt.transition_control_point(&loc1, ptr::null_mut()),
            TransitionControlPoint::Execute(_)
        );
    }
}
