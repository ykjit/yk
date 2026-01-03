use rerun_except::rerun_except;
use std::env;

pub fn main() {
    // Don't rebuild the whole crate when only test inputs change.
    rerun_except(&[
        "c",
        "extra_linkage",
        "lua",
        "trace_compiler",
        "benches/*.c",
        "yklua",
    ])
    .unwrap();

    // Expose the cargo profile to run.rs so that it can set the right link flags.
    let profile = env::var("PROFILE").unwrap();
    println!("cargo::rustc-cfg=cargo_profile=\"{profile}\"");
    println!(r#"cargo::rustc-check-cfg=cfg(cargo_profile, values("debug", "release"))"#);

    println!("cargo::rerun-if-env-changed=YKB_TRACER");
    println!("cargo::rustc-check-cfg=cfg(tracer_swt)");
    match env::var("YKB_TRACER") {
        Ok(ref tracer) if tracer == "swt" => println!("cargo::rustc-cfg=tracer_swt"),
        Err(env::VarError::NotPresent) => println!("cargo::rustc-cfg=tracer_swt"),
        Ok(x) => panic!("Unknown tracer {x}"),
        Err(_) => panic!("Invalid value for YKB_TRACER"),
    }
}
