//! This module records statistics about yk and the VM it is part of. The accuracy of the
//! statistics varies: for example, "durations" are wall-clock time, which inevitably fail to
//! account for context switches and the like. Thus the statistics are very much in "best effort"
//! territory -- but it's better than nothing!

#[cfg(not(test))]
use std::env;
#[cfg(feature = "yk_testing")]
use std::sync::Condvar;
use std::{
    cell::Cell,
    fs,
    ops::DerefMut,
    sync::Mutex,
    time::{Duration, Instant},
};
use strum::{Display, EnumCount, EnumIter, IntoEnumIterator};

/// Record yk statistics if enabled. In non-testing mode, this is only enabled if the end user
/// defines the environment variable `YKD_LOG_STATS`. In testing mode, this is always enabled, with
/// output being sent to `stderr`.
pub(crate) struct Stats {
    // On most runs of yk we anticipate that the end user won't want to be recording JIT
    // statistics, so we want to able to do the quickest possible check for "are any stats to be
    // recorded?" The outer `Option` means thus becomes a simple `if (NULL) { return}` check: only
    // if stats are to be recorded do we have to go to the expense of locking a `Mutex`.
    inner: Option<Mutex<StatsInner>>,
    // In `yk_testing`, this [CondVar] allows threads to wait until a certain set of events have
    // happened with the [Self::wait_until] function.
    #[cfg(feature = "yk_testing")]
    wait_until_condvar: Option<Condvar>,
}

struct StatsInner {
    /// The path to write output. If exactly equal to `-`, output will be written to stderr.
    output_path: String,
    /// How many traces were recorded successfully?
    traces_recorded_ok: u64,
    /// How many traces were recorded unsuccessfully?
    traces_recorded_err: u64,
    /// How many traces were compiled successfully?
    traces_compiled_ok: u64,
    /// How many traces were compiled unsuccessfully?
    traces_compiled_err: u64,
    /// How many times have traces been executed? Note that the same trace can count arbitrarily
    /// many times to this.
    trace_executions: u64,
    /// The time spent in each [TimingState].
    durations: [Duration; TimingState::COUNT],
}

impl Stats {
    #[cfg(not(test))]
    pub fn new() -> Self {
        if let Ok(p) = env::var("YKD_LOG_STATS") {
            Self {
                inner: Some(Mutex::new(StatsInner::new(p))),
                #[cfg(feature = "yk_testing")]
                wait_until_condvar: Some(Condvar::new()),
            }
        } else {
            Self {
                inner: None,
                #[cfg(feature = "yk_testing")]
                wait_until_condvar: None,
            }
        }
    }

    #[cfg(test)]
    pub fn new() -> Self {
        Self {
            inner: Some(Mutex::new(StatsInner::new("-".to_string()))),
            #[cfg(feature = "yk_testing")]
            wait_until_condvar: None,
        }
    }

    /// If `YKD_LOG_STATS` was specified, update `inner` by running the function `f`, otherwise return
    /// immediately without calling `f`.
    fn update_with<F>(&self, f: F)
    where
        F: FnOnce(&mut StatsInner),
    {
        if let Some(mtx) = &self.inner {
            let mut lk = mtx.lock().unwrap();
            f(lk.deref_mut());
            #[cfg(feature = "yk_testing")]
            {
                drop(lk);
                if let Some(x) = self.wait_until_condvar.as_ref() {
                    x.notify_all()
                }
            }
        }
    }

    /// Iff `YKD_LOG_STATS` is set, suspend this thread's execution until `test(StatsInner)` returns
    /// true. Note that a lock is held on yk's statistics while `test` is called, so `test` should
    /// not perform lengthy calculations (if it does, it may block other threads).
    ///
    /// # Panics
    ///
    /// If `YKD_LOG_STATS` is not set.
    #[cfg(feature = "yk_testing")]
    fn wait_until<F>(&self, test: F)
    where
        F: Fn(&mut StatsInner) -> bool,
    {
        match &self.inner {
            Some(mtx) => {
                let mut lk = mtx.lock().unwrap();
                while !test(lk.deref_mut()) {
                    lk = self
                        .wait_until_condvar
                        .as_ref()
                        .expect("Can't call wait_until unless YKD_LOG_STATS is set")
                        .wait(lk)
                        .unwrap();
                }
            }
            None => panic!("Can't call wait_until unless YKD_LOG_STATS is set"),
        }
    }

    /// Increment the "a trace has been recorded successfully" count.
    pub fn trace_recorded_ok(&self) {
        self.update_with(|inner| inner.traces_recorded_ok += 1);
    }

    /// Increment the "a trace has been recorded unsuccessfully" count.
    pub fn trace_recorded_err(&self) {
        self.update_with(|inner| inner.traces_recorded_err += 1);
    }

    /// Increment the "a trace has been compiled successfully" count.
    pub fn trace_compiled_ok(&self) {
        self.update_with(|inner| inner.traces_compiled_ok += 1);
    }

    /// Increment the "a trace has been compiled unsuccessfully" count.
    pub fn trace_compiled_err(&self) {
        self.update_with(|inner| inner.traces_compiled_err += 1);
    }

    /// Increment the "a compiled trace has started execution" count.
    pub fn trace_executed(&self) {
        self.update_with(|inner| inner.trace_executions += 1);
    }

    /// Change the [TimingState] the current thread is in.
    pub fn timing_state(&self, new_state: TimingState) {
        self.update_with(|inner| {
            let now = Instant::now();
            let (prev_state, then) = VM_STATE.replace((new_state, now));
            let d = now.saturating_duration_since(then);
            inner.durations[prev_state as usize] =
                inner.durations[prev_state as usize].saturating_add(d);
        });
    }

    /// Output these statistics to the appropriate output path.
    pub(crate) fn output(&self) {
        self.update_with(|inner| inner.output());
    }
}

impl StatsInner {
    fn new(output_path: String) -> Self {
        Self {
            output_path,
            traces_recorded_ok: 0,
            traces_recorded_err: 0,
            traces_compiled_ok: 0,
            traces_compiled_err: 0,
            trace_executions: 0,
            durations: [Duration::new(0, 0); TimingState::COUNT],
        }
    }

    /// Output these statistics to the appropriate output path.
    fn output(&self) {
        let json = self.to_json();
        if self.output_path == "-" {
            eprintln!("{json}");
        } else {
            fs::write(&self.output_path, json).ok();
        }
    }

    /// Turn these statistics into JSON. The output is guaranteed to be sorted by field name so
    /// that textual matching of the JSON string (e.g. in lang_tester) is possible.
    fn to_json(&self) -> String {
        fn fmt_duration(d: Duration) -> String {
            format!("{}.{:03}", d.as_secs(), d.subsec_millis())
        }

        let mut fields = vec![
            (
                "traces_recorded_ok".to_owned(),
                self.traces_recorded_ok.to_string(),
            ),
            (
                "traces_recorded_err".to_owned(),
                self.traces_recorded_err.to_string(),
            ),
            (
                "traces_compiled_ok".to_owned(),
                self.traces_compiled_ok.to_string(),
            ),
            (
                "traces_compiled_err".to_owned(),
                self.traces_compiled_err.to_string(),
            ),
            (
                "trace_executions".to_owned(),
                self.trace_executions.to_string(),
            ),
        ];
        for v in TimingState::iter() {
            let s = v.to_string();
            if !s.is_empty() {
                fields.push((s, fmt_duration(self.durations[v as usize])));
            }
        }
        // We sort the output fields so that tests can match the output with a simple text match.
        fields.sort_unstable_by(|(k1, _), (k2, _)| k1.cmp(k2));
        format!(
            r#"{{
    {}
}}"#,
            fields
                .iter()
                .map(|(x, y)| format!(r#""{x}": {y}"#))
                .collect::<Vec<_>>()
                .join(",\n    ")
        )
    }
}

/// The different timing states a VM can go through.
#[repr(u8)]
#[derive(Copy, Clone, Display, EnumCount, EnumIter)]
// You can add new states to this with the following notes:
//   1. `TimingState` must be `repr(T)` where `T` is an integer that can be convert with `as usize`
//      without loss of information.
//   2. The variants range from `0..TimingState::COUNT`. In other words, don't assign numbers to
//      any of the variants with `= <int>`!
//   2. Any new state has a `strum` `to_string` that produces the name of the key that will appear
//      in the JSON stats. If `to_string` produces the empty string, that value will not appear in
//      the JSON stats.
pub(crate) enum TimingState {
    /// The "we don't know what this thread is doing" state. Time spent in this state is not
    /// counted towards anything and is not displayed to the user.
    #[strum(to_string = "")]
    None,
    /// This thread is compiling a mapped trace.
    #[strum(to_string = "duration_compiling")]
    Compiling,
    /// This thread is deoptimising from a guard failure.
    #[strum(to_string = "duration_deopting")]
    Deopting,
    /// This thread is executing machine code compiled by yk.
    #[strum(to_string = "duration_jit_executing")]
    JitExecuting,
    /// This thread is executing code outside yk (roughly "in the interpreter").
    #[strum(to_string = "duration_outside_yk")]
    OutsideYk,
    /// This thread is mapping a trace into IR.
    #[strum(to_string = "duration_trace_mapping")]
    TraceMapping,
}

thread_local! {
    static VM_STATE: Cell<(TimingState, Instant)> = Cell::new((TimingState::OutsideYk, Instant::now()));
}

/// Various functions for the C testing suite.
#[cfg(feature = "yk_testing")]
mod yk_testing {
    use crate::mt::MT;

    /// This struct *must* stay in sync with `YkCStats` in `yk_testing.h`.
    #[repr(C)]
    pub struct YkCStats {
        traces_recorded_ok: u64,
        traces_recorded_err: u64,
        traces_compiled_ok: u64,
        traces_compiled_err: u64,
        trace_executions: u64,
    }

    #[no_mangle]
    pub extern "C" fn __ykstats_wait_until(
        mt: *const MT,
        test: unsafe extern "C" fn(YkCStats) -> bool,
    ) {
        let mt = unsafe { &*mt };
        if mt.stats.inner.is_none() {
            panic!("Statistics collection not enabled");
        }
        mt.stats.wait_until(|inner| {
            let cstats = YkCStats {
                traces_recorded_ok: inner.traces_recorded_ok,
                traces_recorded_err: inner.traces_recorded_err,
                traces_compiled_ok: inner.traces_compiled_ok,
                traces_compiled_err: inner.traces_compiled_err,
                trace_executions: inner.trace_executions,
            };
            unsafe { test(cstats) }
        });
    }
}
