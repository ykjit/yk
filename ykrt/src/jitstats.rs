/// This module records statistics about Yk and the VM it is part of. The accuracy of the
/// statistics varies: in general, the statistics about yk's internals are accurate, but
/// interactions with the wider interpreter are likely to be approximations, because we can only
/// guess at what the interpreter is doing. For example, when an interpreter spins up a thread,
/// does that count as "time spent interpreting"? What happens if it's a job queue that immediately
/// goes to sleep? In such cases, we are very much in "best effort" territory -- but it's better
/// than nothing!

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

pub(crate) struct JitStats {
    inner: Option<Mutex<JitStatsInner>>,
}

struct JitStatsInner {
    /// The path to write output. If exactly equal to `-`, output will be written to stderr.
    output_path: String,
    /// How many traces were collected successfully?
    traces_collected_ok: u64,
    /// How many traces were collected unsuccessfully?
    traces_collected_err: u64,
    /// How many traces were compiled successfully?
    traces_compiled_ok: u64,
    /// How many traces were compiled unsuccessfully?
    traces_compiled_err: u64,
    durations: [Duration; TimingState::COUNT],
}

impl JitStats {
    #[cfg(not(test))]
    pub fn new() -> Self {
        if let Ok(p) = env::var("YKD_JITSTATS") {
            Self {
                inner: Some(Mutex::new(JitStatsInner::new(p))),
            }
        } else {
            Self { inner: None }
        }
    }

    #[cfg(test)]
    pub fn new() -> Self {
        Self {
            inner: Some(Mutex::new(JitStatsInner::new("-".to_string()))),
        }
    }

    fn lock<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&mut JitStatsInner) -> T,
    {
        self.inner
            .as_ref()
            .map(|x| f(x.lock().unwrap().deref_mut()))
    }

    pub fn trace_collected_ok(&self) {
        self.lock(|inner| inner.traces_collected_ok += 1);
    }

    pub fn trace_collected_err(&self) {
        self.lock(|inner| inner.traces_collected_err += 1);
    }

    pub fn trace_compiled_ok(&self) {
        self.lock(|inner| inner.traces_compiled_ok += 1);
    }

    pub fn trace_compiled_err(&self) {
        self.lock(|inner| inner.traces_compiled_err += 1);
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

impl JitStatsInner {
    fn new(output_path: String) -> Self {
        Self {
            output_path,
            traces_collected_ok: 0,
            traces_collected_err: 0,
            traces_compiled_ok: 0,
            traces_compiled_err: 0,
            durations: [Duration::new(0, 0); TimingState::COUNT],
        }
    }

    fn to_json(&self) -> String {
        fn fmt_duration(d: Duration) -> String {
            format!("{}.{:03}", d.as_secs(), d.subsec_millis())
        }

        let mut fields = vec![
            (
                "traces_collected_ok".to_owned(),
                self.traces_collected_ok.to_string(),
            ),
            (
                "traces_collected_err".to_owned(),
                self.traces_collected_err.to_string(),
            ),
            (
                "traces_compiled_ok".to_owned(),
                self.traces_compiled_ok.to_string(),
            ),
            (
                "traces_compiled_err".to_owned(),
                self.traces_compiled_err.to_string(),
            ),
        ];
        for v in TimingState::iter() {
            let s = v.to_string();
            if !s.is_empty() {
                fields.push((s, fmt_duration(self.durations[v as usize])));
            }
        }
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

impl Drop for JitStatsInner {
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
