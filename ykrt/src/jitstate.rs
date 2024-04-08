#[cfg(feature = "ykd")]
mod jitstate {
    use std::{env, sync::LazyLock};

    static JITSTATE_DEBUG: LazyLock<bool> =
        LazyLock::new(|| env::var("YKD_PRINT_JITSTATE").is_ok());

    /// Print select JIT events to stderr for testing/debugging purposes.
    pub fn print_jit_state(state: &str) {
        if *JITSTATE_DEBUG {
            eprintln!("jit-state: {}", state);
        }
    }
}

pub(crate) use jitstate::print_jit_state;
