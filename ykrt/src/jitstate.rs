//! The implementation of the `YKD_LOG_JITSTATE` environment variable.
//!
//! Note: the `print_jit_state` function exposed here is always defined but is guaranteed to be a
//! no-op unless the `ykd` feature is enabled.

#[cfg(not(feature = "ykd"))]
mod jitstate {
    pub(crate) fn print_jit_state(_: &str) {}
}

#[cfg(feature = "ykd")]
mod jitstate {
    use std::{env, fs, sync::LazyLock};

    static JITSTATE_DEBUG: LazyLock<Option<String>> =
        LazyLock::new(|| env::var("YKD_LOG_JITSTATE").ok());

    /// Print select JIT events to stderr for testing/debugging purposes.
    pub fn print_jit_state(state: &str) {
        match JITSTATE_DEBUG.as_ref().map(|x| x.as_str()) {
            Some("-") => eprintln!("jit-state: {}", state),
            Some(x) => {
                fs::write(x, state).ok();
            }
            None => (),
        }
    }
}

pub(crate) use jitstate::print_jit_state;
