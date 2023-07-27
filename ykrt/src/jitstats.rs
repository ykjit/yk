#[cfg(not(test))]
use std::env;
use std::{fs, ops::DerefMut, sync::Mutex};

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
}

impl JitStatsInner {
    fn new(output_path: String) -> Self {
        Self {
            output_path,
            traces_collected_ok: 0,
            traces_collected_err: 0,
            traces_compiled_ok: 0,
            traces_compiled_err: 0,
        }
    }

    fn to_json(&self) -> String {
        let traces_collected_ok = self.traces_collected_ok;
        let traces_collected_err = self.traces_collected_err;
        let traces_compiled_ok = self.traces_compiled_ok;
        let traces_compiled_err = self.traces_compiled_err;
        format!(
            r#"{{
    "traces_collected_ok": {traces_collected_ok},
    "traces_collected_err": {traces_collected_err},
    "traces_compiled_ok": {traces_compiled_ok},
    "traces_compiled_err": {traces_compiled_err}
}}"#
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
