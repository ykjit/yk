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
            Self::Internal => [&this_dir, "..", "internal_ws"].iter().collect::<PathBuf>(),
            Self::External => [&this_dir, ".."].iter().collect::<PathBuf>(),
        }
    }
}

/// A build step for one of the workspaces.
#[derive(Debug)]
struct WorkspaceAction<'a> {
    /// The tool we will invoke. Usually cargo.
    tool: String,
    /// Arguments to the above tool.
    tool_args: Vec<&'a str>,
    /// The path the to the workspace we will work in.
    workspace_dir: PathBuf,
    /// Arguments appended after `tool_args`.
    target_args: Vec<&'a str>,
    /// The RUSTFLAGS environment to use.
    rust_flags: String,
    /// Workspace actions to run first.
    forced_deps: Vec<Self>,
}

impl<'a> WorkspaceAction<'a> {
    fn new(workspace: Workspace, target: &'a str) -> Result<Self, String> {
        let mut tool = env::var("CARGO").unwrap();
        let mut forced_deps = Vec::new();
        let mut target_args = vec![target];
        let mut tool_args = Vec::new();
        let mut rust_flags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());

        match target {
            "audit" => (),
            "build" | "check" | "clean" | "test" => {
                let tracing_kind = find_tracing_kind(&rust_flags);
                if workspace == Workspace::Internal {
                    rust_flags.clear();
                    // Optimise the internal workspace.
                    target_args.push("--release");
                    // Set the tracermode cfg macro, but without changing anything relating to code
                    // generation. We can't use `-C tracer=hw` as this would turn off optimisations
                    // and emit SIR for stuff we will never trace.
                    rust_flags.push_str(&format!(" --cfg tracermode=\"{}\"", tracing_kind));

                    // `cargo test` in the internal workspace won't build libykshim.so, so we have
                    // to force-build it to avoid linkage problems for the external workspace.
                    if target == "test" {
                        forced_deps.push(WorkspaceAction::new(Workspace::Internal, "build")?);
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
            _ => {
                return Err(format!(
                    "the build system does not support the {} target",
                    target
                ))
            }
        }

        Ok(Self {
            tool,
            workspace_dir: workspace.dir(),
            tool_args,
            target_args,
            rust_flags,
            forced_deps,
        })
    }

    fn run(self, extra_args: &[String]) -> Result<(), String> {
        // Run any dependencies first.
        for dep in self.forced_deps {
            dep.run(&[])?;
        }

        let mut cmd = Command::new(&self.tool);
        let status = cmd
            .current_dir(self.workspace_dir)
            .args(self.tool_args)
            .args(self.target_args)
            .args(extra_args)
            .env("RUSTFLAGS", self.rust_flags)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        if !status.success() {
            let pb = PathBuf::from(self.tool);
            let base = pb.iter().last().unwrap().to_str().unwrap();
            return Err(format!("{} failed with exit code {}", base, status));
        }
        Ok(())
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

    let run_action = |ws: Workspace, target: &str, args: &[String]| {
        let act = WorkspaceAction::new(ws, target).unwrap_or_else(|e| bail(e));
        act.run(args).unwrap_or_else(|e| bail(e));
    };

    run_action(Workspace::Internal, &target, &extra_args);
    run_action(Workspace::External, &target, &extra_args);
}
