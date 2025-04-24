//! The implementation of the `YKD_LOG_*` environment variables.
//!
//! Note that some of these features are only meaningfully available when the `ykd` feature is
//! available: otherwise we expose no-op functions.

#[cfg(feature = "ykd")]
use parking_lot::Mutex;
use std::{env, error::Error, fs::File, io::Write, path::PathBuf};
use strum::{EnumCount, FromRepr};

#[cfg(feature = "ykd")]
use crate::location::HotLocation;

pub(crate) mod stats;

/// How verbose should yk's normal logging be?
#[repr(u8)]
#[derive(Copy, Clone, Debug, EnumCount, FromRepr, PartialEq, PartialOrd)]
pub(crate) enum Verbosity {
    /// Disable logging entirely.
    Disabled,
    /// Log errors.
    Error,
    /// Log warnings.
    Warning,
    /// Log transitions of a [Location].
    LocationTransition,
    /// Log JIT events (e.g. start/stop tracing).
    JITEvent,
}

pub(crate) struct Log {
    /// The requested [Verbosity] level for logging.
    level: Verbosity,
    /// The path to write to. A value of `None` should default to the platform specific standard
    /// for logging (e.g. stderr).
    path: Option<PathBuf>,
}

impl Log {
    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        match env::var("YKD_LOG") {
            Ok(s) => {
                let (path, level) = match s.split(':').collect::<Vec<_>>()[..] {
                    [path, level] => {
                        if path == "-" {
                            (None, level)
                        } else {
                            let path = PathBuf::from(path);
                            // If there's an existing log file, truncate (i.e. empty it), so that later
                            // appends to the log aren't appending to a previous log run.
                            File::create(&path).ok();
                            (Some(path), level)
                        }
                    }
                    [level] => (None, level),
                    [..] => return Err("YKD_LOG must be of the format `[<path|->:]<level>".into()),
                };
                let level = level
                    .parse::<u8>()
                    .map_err(|e| format!("Invalid YKD_LOG level '{s}': {e}"))?;
                // This unwrap can only fail dynamically if we've got the types wrong statically
                // (i.e. it'll fail as soon as this code is executed for the first time).
                let max_level = u8::try_from(Verbosity::COUNT).unwrap() - 1;
                let level = Verbosity::from_repr(level)
                    .ok_or_else(|| format!("YKD_LOG level {level} exceeds maximum {max_level}"))?;
                Ok(Self { path, level })
            }
            Err(_) => Ok(Self {
                path: None,
                level: Verbosity::Error,
            }),
        }
    }

    #[cfg(feature = "ykd")]
    pub(crate) fn log_with_hl_debug(&self, level: Verbosity, msg: &str, hl: &Mutex<HotLocation>) {
        // If the hot location has a debug string, append it to the log message.
        let msg = if let Some(dstr) = hl.lock().debug_str.as_ref() {
            &format!("{msg}: {dstr}")
        } else {
            msg
        };
        self.log(level, msg);
    }

    /// Log `msg` with the [Verbosity] level `verbosity`.
    ///
    /// # Panics
    ///
    /// If `level == Verbosity::None`.
    pub(crate) fn log(&self, level: Verbosity, msg: &str) {
        if level <= self.level {
            let prefix = match level {
                Verbosity::Disabled => panic!(),
                Verbosity::Error => "yk-error",
                Verbosity::Warning => "yk-warning",
                Verbosity::JITEvent => "yk-jit-event",
                Verbosity::LocationTransition => "yk-location-transition",
            };
            match &self.path {
                Some(p) => {
                    let s = format!("{prefix}: {msg}\n");
                    File::options()
                        .append(true)
                        .open(p)
                        .map(|mut x| x.write(s.as_bytes()))
                        .ok();
                }
                None => {
                    eprintln!("{prefix}: {msg}");
                }
            }
        }
    }
}

#[derive(Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub(crate) enum IRPhase {
    /// The AOT IR.
    AOT,
    /// Debug strings only
    DebugStrs,
    /// The JIT IR before it has been optimised.
    PreOpt,
    /// The JIT IR after it has been optimised.
    PostOpt,
    /// The assembler code of the compiled trace.
    Asm,
    /// The assembler code of the compiled trace, including instruction offsets and virtual
    /// addresses.
    AsmFull,
}

#[cfg(not(feature = "ykd"))]
mod internals {
    use super::IRPhase;
    pub(crate) fn should_log_ir(_: IRPhase) -> bool {
        false
    }
    pub(crate) fn log_ir(_: &str) {}
}

#[cfg(feature = "ykd")]
mod internals {
    use super::IRPhase;
    use std::{collections::HashSet, env, error::Error, fs::File, io::Write, sync::LazyLock};

    static LOG_IR: LazyLock<Option<(String, HashSet<IRPhase>)>> = LazyLock::new(|| {
        let mut log_phases = HashSet::new();
        if let Ok(x) = env::var("YKD_LOG_IR") {
            let (path, phases) = match x.split(':').collect::<Vec<_>>().as_slice() {
                [path, phases] => (*path, *phases),
                [phases] => ("-", *phases),
                _ => panic!(
                    "YKD_LOG_IR must be of the format '[<path>:]<irstage_1>[,...,<irstage_n>]'"
                ),
            };
            for x in phases.split(',') {
                log_phases.insert(IRPhase::from_str(x).unwrap());
            }
            if path != "-" {
                // If there's an existing log file, truncate (i.e. empty it), so that later
                // appends to the log aren't appending to a previous log run.
                File::create(path).ok();
            }
            Some((path.to_string(), log_phases))
        } else {
            None
        }
    });

    impl IRPhase {
        fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
            match s {
                "aot" => Ok(Self::AOT),
                "debugstrs" => Ok(Self::DebugStrs),
                "jit-pre-opt" => Ok(Self::PreOpt),
                "jit-post-opt" => Ok(Self::PostOpt),
                "jit-asm" => Ok(Self::Asm),
                "jit-asm-full" => Ok(Self::AsmFull),
                _ => Err(format!("Invalid YKD_LOG_IR value: {s}").into()),
            }
        }
    }

    pub(crate) fn should_log_ir(phase: IRPhase) -> bool {
        if let Some(true) = LOG_IR.as_ref().map(|(_, phases)| phases.contains(&phase)) {
            return true;
        }
        false
    }

    pub(crate) fn log_ir(s: &str) {
        match LOG_IR.as_ref().map(|(p, _)| p.as_str()) {
            Some("-") => eprint!("{s}"),
            Some(x) => {
                File::options()
                    .append(true)
                    .open(x)
                    .map(|mut x| x.write(s.as_bytes()))
                    .ok();
            }
            None => (),
        }
    }
}

pub(crate) use internals::{log_ir, should_log_ir};
