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
    Internal,
    External,
}

fn run_action(workspace: Workspace, target: &str, extra_args: &[String]) {
    // The external workspace depends on libykshim.so produced by the internal workspace
    if workspace == Workspace::External {
        run_action(Workspace::Internal, target, extra_args);
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
        cmd.args(&["run", "nightly", "cargo", target]);

        let this_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let ws_dir = match workspace {
            Workspace::Internal => [&this_dir, "..", "internal_ws"].iter().collect::<PathBuf>(),
            Workspace::External => [&this_dir, ".."].iter().collect::<PathBuf>(),
        };
        cmd.current_dir(ws_dir);

        cmd
    } else {
        let mut cmd = Command::new(env::var("CARGO").unwrap());
        cmd.arg(target);

        let this_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let ws_dir = match workspace {
            Workspace::Internal => [&this_dir, "..", "internal_ws", "ykshim"]
                .iter()
                .collect::<PathBuf>(),
            Workspace::External => [&this_dir, ".."].iter().collect::<PathBuf>(),
        };
        cmd.current_dir(ws_dir);

        cmd
    };

    let mut rust_flags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());

    match target {
        "audit" => {}
        "fmt" => {
            rust_flags.clear();
        }
        "clean" => {
            if workspace == Workspace::Internal {
                // The internal workspace is optimized.
                cmd.arg("--release".to_string());
            }
        }
        "bench" | "build" | "check" | "clippy" | "test" => {
            // Ensure that the whole workspace is tested and not just the base crate in the
            // workspace.
            if target == "test" {
                cmd.arg("--workspace");
            }

            if workspace == Workspace::Internal {
                let tracing_kind = find_tracing_kind(&rust_flags);
                rust_flags = make_internal_rustflags(&rust_flags);

                // Optimise the internal workspace. `--release` is not valid for `cargo bench`.
                if target != "bench" {
                    cmd.arg("--release");
                }
                // Set the tracermode cfg macro, but without changing anything relating to code
                // generation. We can't use `-C tracer=hw` as this would turn off optimisations
                // and emit SIR for stuff we will never trace.
                cmd.arg("--features");
                cmd.arg(format!("yktrace/trace_{}", tracing_kind));

                // `cargo test` in the internal workspace won't build libykshim.so, so we have
                // to force-build it to avoid linkage problems for the external workspace.
                if target == "test" {
                    run_action(Workspace::Internal, "build", &[]);
                }
            } else if workspace == Workspace::External && target == "clippy" {
                let tracing_kind = find_tracing_kind(&rust_flags);
                rust_flags = format!("--cfg tracermode=\"{}\"", tracing_kind);
            } else if workspace == Workspace::External && target == "bench" {
                // Setting this in `[profile.bench]` in `Cargo.toml` doesn't work.
                rust_flags.push_str(" -C opt-level=0");
            }
        }
        _ => bail(format!(
            "the build system does not support the {} target",
            target
        )),
    }

    let status = cmd
        .args(extra_args)
        .env("RUSTFLAGS", rust_flags)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();

    // The clippy exception ensures that both workspaces are linted if one workspace fails. The
    // exit status may be inaccurate, but we can live with this.
    if !status.success() && (target != "clippy") {
        bail(format!("{:?} failed with {}", cmd, status));
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
