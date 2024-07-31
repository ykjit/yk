use rerun_except::rerun_except;
use std::env;

pub fn main() {
    // Don't rebuild the whole crate when only test inputs change.
    rerun_except(&["c", "extra_linkage", "trace_compiler", "benches/*.c"]).unwrap();

    // Expose the cargo profile to run.rs so that it can set the right link flags.
    if let Ok(profile) = env::var("PROFILE") {
        println!("cargo::rustc-cfg=cargo_profile=\"{}\"", profile);
        println!(
            r#"cargo::rustc-check-cfg=cfg(cargo_profile, values("debug", "release", "release-with-debug"))"#
        );
    }
}
