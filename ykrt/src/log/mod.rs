//! The implementation of the `YKD_LOG_*` environment variables.
//!
//! Note that some of these features are only meaningfully available when the `ykd` feature is
//! available: otherwise we expose no-op functions.

use strum::{EnumCount, FromRepr};

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

#[derive(Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub(crate) enum IRPhase {
    AOT,
    PreOpt,
    PostOpt,
    Asm,
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

    // YKD_LOG_IR

    static LOG_IR: LazyLock<Option<(String, HashSet<IRPhase>)>> = LazyLock::new(|| {
        let mut log_phases = HashSet::new();
        if let Ok(x) = env::var("YKD_LOG_IR") {
            match x.split(':').collect::<Vec<_>>().as_slice() {
                [p, phases] => {
                    for x in phases.split(',') {
                        log_phases.insert(IRPhase::from_str(x).unwrap());
                    }
                    if *p != "-" {
                        // If there's an existing log file, truncate (i.e. empty it), so that later
                        // appends to the log aren't appending to a previous log run.
                        File::create(p).ok();
                    }
                    Some((p.to_string(), log_phases))
                }
                _ => panic!(
                    "YKD_LOG_IR must be of the format '<path>:<irstage_1>[,...,<irstage_n>]'"
                ),
            }
        } else {
            None
        }
    });

    impl IRPhase {
        fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
            match s {
                "aot" => Ok(Self::AOT),
                "jit-pre-opt" => Ok(Self::PreOpt),
                "jit-post-opt" => Ok(Self::PostOpt),
                "jit-asm" => Ok(Self::Asm),
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
            Some("-") => eprint!("{}", s),
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
