//! Custom build system for the Yorick meta-tracer.
//!
//! This is required because we need to separately compile parts of the codebase with different
//! configurations. To that end we have two workspaces.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

use regex::Regex;
use std::{
    env,
    path::PathBuf,
    process::{exit, Command},
};

include!("../../build_aux.rs");

#[derive(PartialEq)]
enum Workspace {
    Internal,
    External,
}

impl Workspace {
    fn dir(&self) -> PathBuf {
        let this_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        match self {
            Self::Internal => [&this_dir, "..", "internal_ws", "ykshim"]
                .iter()
                .collect::<PathBuf>(),
            Self::External => [&this_dir, ".."].iter().collect::<PathBuf>(),
        }
    }
}

fn run_action(workspace: Workspace, target: &str, extra_args: &[String]) {
    // The external workspace depends on libykshim.so produced by the internal workspace
    if workspace == Workspace::External {
        run_action(Workspace::Internal, target, extra_args);
    }

    let mut tool = env::var("CARGO").unwrap();
    let mut target_args = vec![target.to_string()];
    let mut tool_args: Vec<&str> = Vec::new();
    let mut rust_flags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());

    match target {
        "audit" => (),
        "clean" => {
            if workspace == Workspace::Internal {
                // The internal workspace is optimized.
                target_args.push("--release".to_string());
            }
        }
        "build" | "check" | "test" => {
            if workspace == Workspace::Internal {
                let tracing_kind = find_tracing_kind(&rust_flags);
                rust_flags = make_internal_rustflags(&rust_flags);

                // Optimise the internal workspace.
                target_args.push("--release".to_string());
                // Set the tracermode cfg macro, but without changing anything relating to code
                // generation. We can't use `-C tracer=hw` as this would turn off optimisations
                // and emit SIR for stuff we will never trace.
                target_args.push("--features".to_string());
                target_args.push(format!("yktrace/trace_{}", tracing_kind));

                // `cargo test` in the internal workspace won't build libykshim.so, so we have
                // to force-build it to avoid linkage problems for the external workspace.
                if target == "test" {
                    run_action(Workspace::Internal, "build", extra_args);
                }
            }
        }
        "fmt" => {
            // There is currently a bug where `cargo fmt` doesn't work for linked toolchains:
            // https://github.com/rust-lang/rust/issues/81431
            //
            // As a workaround we fall back on the nightly toolchain installed via rustup. This
            // is confusing. Normally when we run `cargo`, having used `rustup` to install, it
            // is not actually `cargo` that we run, but a wrapper provided by `rustup`. It is
            // this wrapper which understands the `+nightly` argument. The binary pointed to by
            // by $CARGO (in the environment) is a real cargo (not a wrapper) which won't
            // understand `+nightly`. So the easiest way to run `cargo fmt` for the nightly
            // toolchain is to use `rustup run nightly cargo fmt`.
            rust_flags.clear();
            tool = "rustup".to_owned();
            tool_args.extend(&["run", "nightly", "cargo"]);
        }
        _ => bail(format!(
            "the build system does not support the {} target",
            target
        )),
    }

    let mut cmd = Command::new(&tool);
    let status = cmd
        .current_dir(workspace.dir())
        .args(tool_args)
        .args(target_args)
        .args(extra_args)
        .env("RUSTFLAGS", rust_flags)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    if !status.success() {
        let pb = PathBuf::from(tool);
        let base = pb.iter().last().unwrap().to_str().unwrap();
        bail(format!("{} failed with exit code {}", base, status));
    }
}

fn bail(err_str: String) -> ! {
    eprintln!("xtask: {}", err_str);
    exit(1);
}

fn main() {
    env::set_var("YK_XTASK", "YES");
    let mut args = env::args().skip(1);
    let target = args.next().unwrap();
    let extra_args = args.collect::<Vec<_>>();

    run_action(Workspace::External, &target, &extra_args);
}
