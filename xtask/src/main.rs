//! Custom build system for the Yorick meta-tracer.
//!
//! This is required because we need to separately compile parts of the codebase with different
//! configurations. To that end we have two workspaces.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

use std::{
    env,
    path::PathBuf,
    process::{exit, Command},
};

include!("../../build_aux.rs");

#[derive(PartialEq, Debug, Clone, Copy)]
enum Workspace {
    Untraced,
    Traced,
    Xtask,
}

impl Workspace {
    /// Get the path to the workspace relative to the xtask crate's path.
    fn path(&self) -> PathBuf {
        let xtask_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        match self {
            Self::Untraced => [&xtask_dir, "..", "untraced"].iter().collect::<PathBuf>(),
            Self::Traced => [&xtask_dir, "..", "traced"].iter().collect::<PathBuf>(),
            Self::Xtask => PathBuf::from(&xtask_dir),
        }
    }
}

fn run_action(workspace: Workspace, target: &str, extra_args: &[String], features: &[String]) {
    // The traced workspace depends on libuntraced_api.so produced by the untraced workspace
    if workspace == Workspace::Traced {
        run_action(Workspace::Untraced, target, extra_args, &[]);
    }

    let mut cmd = if ["fmt", "clippy"].contains(&target) {
        // There is currently a bug where `cargo fmt` and `cargo clippy` doesn't work for linked
        // toolchains:
        // https://github.com/rust-lang/rust/issues/81431
        //
        // As a workaround we fall back on the nightly toolchain installed via rustup. This
        // is confusing. Normally when we run `cargo`, having used `rustup` to install, it
        // is not actually `cargo` that we run, but a wrapper provided by `rustup`. It is
        // this wrapper which understands the `+nightly` argument. The binary pointed to by
        // by $CARGO (in the environment) is a real cargo (not a wrapper) which won't
        // understand `+nightly`. So the easiest way to run `cargo fmt` for the nightly
        // toolchain is to use `rustup run nightly cargo fmt`.
        let mut cmd = Command::new("rustup");
        cmd.args(&["run", "nightly", "cargo", target])
            .current_dir(workspace.path());
        cmd
    } else {
        let mut cmd = Command::new(env::var("CARGO").unwrap());
        cmd.arg(target).current_dir(workspace.path());
        cmd
    };

    let mut rust_flags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());

    match target {
        "audit" | "clean" => {}
        "fmt" => {
            rust_flags.clear();
        }
        "bench" | "build" | "check" | "clippy" | "test" => {
            // Ensure that the whole workspace is tested and not just the base crate in the
            // workspace.
            if target == "test" || target == "bench" {
                cmd.arg("--workspace");
            }

            if workspace == Workspace::Untraced {
                let tracing_kind = find_tracing_kind(&rust_flags);
                rust_flags = make_untraced_rustflags(&rust_flags);

                // Set the tracermode cfg macro, but without changing anything relating to code
                // generation. We can't use `-C tracer=hw` as this would turn off optimisations
                // and emit SIR for stuff we will never trace.
                cmd.arg("--features");
                let mut fs = Vec::from(features);
                fs.push(format!("yktrace/trace_{}", tracing_kind));
                cmd.arg(fs.join(","));

                // `cargo test` in the untraced workspace won't build libuntraced_api.so, so we have
                // to force-build it to avoid linkage problems for the traced workspace.
                if target == "test" || target == "bench" {
                    // Since we are running a different target to what the user requested, the
                    // extra_args they supplied may not be compatible with the "build" target.
                    // But we must pass --release if they supplied it and the "bench" target is
                    // implicitly a release mode build.
                    let mut build_args = Vec::new();
                    if extra_args.contains(&"--release".to_owned()) || target == "bench" {
                        build_args.push("--release".to_owned());
                    };
                    run_action(
                        Workspace::Untraced,
                        "build",
                        &build_args,
                        &["testing".to_owned()],
                    );
                }
            } else if workspace == Workspace::Traced && target == "clippy" {
                let tracing_kind = find_tracing_kind(&rust_flags);
                rust_flags = format!("--cfg tracermode=\"{}\"", tracing_kind);
            }
        }
        _ => fatal(format!(
            "the build system does not support the {} target",
            target
        )),
    }

    cmd.args(extra_args).env("RUSTFLAGS", rust_flags);
    let status = cmd.spawn().unwrap().wait().unwrap();

    // The clippy exception ensures that both workspaces are linted if one workspace fails. The
    // exit status may be inaccurate, but we can live with this.
    if !status.success() && (target != "clippy") {
        fatal(format!("{:?} failed with {}", cmd, status));
    }

    // Ensure that we can clean, fmt and clippy the build system. Other operations are automatic
    // due to the cargo alias in .cargo/config.
    if workspace == Workspace::Traced && ["clean", "fmt", "clippy"].contains(&target) {
        run_action(Workspace::Xtask, target, extra_args, &[]);
    }
}

fn fatal(err_str: String) -> ! {
    eprintln!("xtask: {}", err_str);
    exit(1);
}

fn main() {
    env::set_var("YK_XTASK", "YES");
    let mut args = env::args().skip(1);
    let target = args.next().unwrap();
    let extra_args = args.collect::<Vec<_>>();

    run_action(Workspace::Traced, &target, &extra_args, &[]);
}
