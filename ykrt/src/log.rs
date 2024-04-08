//! The implementation of the `YKD_LOG_*` environment variables.
//!
//! When the `ykd` feature is not enabled, this module exposes no-op functions.

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
    pub(crate) fn log_jit_state(_: &str) {}
    pub(crate) fn should_log_ir(_: IRPhase) -> bool {
        false
    }
    pub(crate) fn log_ir(_: &str) {}
}

#[cfg(feature = "ykd")]
mod internals {
    use super::IRPhase;
    use std::{collections::HashSet, env, error::Error, fs::File, io::Write, sync::LazyLock};

    // YKD_LOG_JITSTATE

    static JITSTATE_DEBUG: LazyLock<Option<String>> =
        LazyLock::new(|| env::var("YKD_LOG_JITSTATE").ok());

    /// Log select JIT events to stderr for testing/debugging purposes.
    pub fn log_jit_state(state: &str) {
        match JITSTATE_DEBUG.as_ref().map(|x| x.as_str()) {
            Some("-") => eprintln!("jitstate: {}", state),
            Some(x) => {
                File::options()
                    .append(true)
                    .open(x)
                    .map(|mut x| x.write(state.as_bytes()))
                    .ok();
            }
            None => (),
        }
    }

    // YKD_LOG_IR

    static LOG_IR: LazyLock<Option<(String, HashSet<IRPhase>)>> = LazyLock::new(|| {
        let mut log_phases = HashSet::new();
        if let Ok(x) = env::var("YKD_LOG_IR") {
            match x.split(":").collect::<Vec<_>>().as_slice() {
                [p, phases] => {
                    for x in phases.split(',') {
                        log_phases.insert(IRPhase::from_str(x).unwrap());
                    }
                    Some((p.to_string(), log_phases))
                }
                _ => panic!("YKD_LOG_IR must be of the format '<path|->:stage_1,...,stage_n'"),
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
            Some("-") => eprintln!("{}", s),
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

pub(crate) use internals::{log_ir, log_jit_state, should_log_ir};
