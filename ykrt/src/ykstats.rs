/// This module records statistics about yk and the VM it is part of. The accuracy of the
/// statistics varies: for example, "durations" are wall-clock time, which inevitably fail to
/// account for context switches and the like. Thus the statistics are very much in "best effort"
/// territory -- but it's better than nothing!

#[cfg(not(test))]
use std::env;
use std::{
    cell::Cell,
    fs,
    ops::DerefMut,
    sync::Mutex,
    time::{Duration, Instant},
};
use strum::{Display, EnumCount, EnumIter, IntoEnumIterator};

/// Record yk statistics if enabled. In non-testing mode, this is only enabled if the end user
/// defines the environment variable `YKD_STATS`. In testing mode, this is always enabled, with
/// output being sent to `stderr`.
pub(crate) struct YkStats {
    // On most runs of yk we anticipate that the end user won't want to be recording JIT
    // statistics, so we want to able to do the quickest possible check for "are any stats to be
    // recorded?" The outer `Option` means thus becomes a simple `if (NULL) { return}` check: only
    // if stats are to be recorded do we have to go to the expense of locking a `Mutex`.
    inner: Option<Mutex<YkStatsInner>>,
}

struct YkStatsInner {
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

impl YkStats {
    #[cfg(not(test))]
    pub fn new() -> Self {
        if let Ok(p) = env::var("YKD_STATS") {
            Self {
                inner: Some(Mutex::new(YkStatsInner::new(p))),
            }
        } else {
            Self { inner: None }
        }
    }

    #[cfg(test)]
    pub fn new() -> Self {
        Self {
            inner: Some(Mutex::new(YkStatsInner::new("-".to_string()))),
        }
    }

    fn lock<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&mut YkStatsInner) -> T,
    {
        self.inner
            .as_ref()
            .map(|x| f(x.lock().unwrap().deref_mut()))
    }

    /// Increment the "a trace has been recorded successfully" count.
    pub fn trace_recorded_ok(&self) {
        self.lock(|inner| inner.traces_recorded_ok += 1);
    }

    /// Increment the "a trace has been recorded unsuccessfully" count.
    pub fn trace_recorded_err(&self) {
        self.lock(|inner| inner.traces_recorded_err += 1);
    }

    /// Increment the "a trace has been compiled successfully" count.
    pub fn trace_compiled_ok(&self) {
        self.lock(|inner| inner.traces_compiled_ok += 1);
    }

    /// Increment the "a trace has been compiled unsuccessfully" count.
    pub fn trace_compiled_err(&self) {
        self.lock(|inner| inner.traces_compiled_err += 1);
    }

    /// Increment the "a compiled trace has started execution" count.
    pub fn trace_executed(&self) {
        self.lock(|inner| inner.trace_executions += 1);
    }

    /// Change the [TimingState] the current thread is in.
    pub fn timing_state(&self, new_state: TimingState) {
        self.lock(|inner| {
            let now = Instant::now();
            let (prev_state, then) = VM_STATE.replace((new_state, now));
            let d = now.saturating_duration_since(then);
            inner.durations[prev_state as usize] =
                inner.durations[prev_state as usize].saturating_add(d);
        });
    }
}

impl YkStatsInner {
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

impl Drop for YkStatsInner {
    fn drop(&mut self) {
        let json = self.to_json();
        if self.output_path == "-" {
            eprintln!("{json}");
        } else {
            fs::write(&self.output_path, json).ok();
        }
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
