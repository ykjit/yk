//! Custom build system for the Yorick meta-tracer.
//!
//! This is required because we need to separately compile parts of the codebase with different
//! configurations. To that end we have two workspaces.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

// FIXME make `cargo xtask fmt` and `cargo audit` work.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let mut args = env::args().skip(1);
    let target = args.next().unwrap();
    let extra_args = args.collect::<Vec<_>>();
    let cargo = env::var("CARGO").unwrap();
    let rflags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());

    // Change into the internal workspace.
    let this_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let int_dir = [&this_dir, "..", "internal_ws"].iter().collect::<PathBuf>();
    env::set_current_dir(&int_dir).unwrap();

    let build_internal = |target: &str, with_extra_args: bool| {
        let mut int_rflags = rflags.clone();
        int_rflags.push_str(" --cfg tracermode=\"hw\"");
        let mut cmd = Command::new(&cargo);
        cmd.arg(&target).arg("--release");
        if with_extra_args {
            cmd.args(&extra_args);
        }
        let status = cmd
            .env_remove("RUSTFLAGS")
            .env("RUSTFLAGS", int_rflags)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        if !status.success() {
            panic!("internal build failed");
        }
    };

    eprintln!("Building internal (optimised) workspace...");
    if target == "test" {
        // FIXME
        // Running `cargo xtask test` won't rebuild ykshim when it has changed, so force it.
        build_internal("build", false);
    }
    build_internal(&target, true);

    let mut ext_rflags = rflags;
    ext_rflags.push_str(" -C tracer=hw");

    eprintln!("Building external (unoptimised) workspace...");
    let ext_dir = [&this_dir, ".."].iter().collect::<PathBuf>();
    let int_target_dir = [int_dir.to_str().unwrap(), "target", "release"]
        .iter()
        .collect::<PathBuf>();
    env::set_current_dir(ext_dir).unwrap();
    let status = Command::new(cargo)
        .arg(&target)
        .args(&extra_args)
        .env_remove("RUSTFLAGS")
        .env("RUSTFLAGS", ext_rflags)
        .env("LD_LIBRARY_PATH", int_target_dir)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    if !status.success() {
        panic!("external build failed");
    }
}
