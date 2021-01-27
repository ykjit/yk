//! Custom build system for the Yorick meta-tracer.
//!
//! This is required because we need to separately compile parts of the codebase with different
//! configurations. To that end we have two workspaces.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

use std::{env, path::PathBuf, process::Command};

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
    /// The LD_LIBRARY_PATH environment to use.
    ld_library_path: String,
    /// Workspace actions to run first.
    forced_deps: Vec<Self>,
}

impl<'a> WorkspaceAction<'a> {
    fn new(workspace: Workspace, target: &'a str) -> Self {
        let mut tool = env::var("CARGO").unwrap();
        let mut forced_deps = Vec::new();
        let mut target_args = vec![target];
        let mut tool_args = Vec::new();
        let mut rust_flags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());
        let mut ld_library_path = env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| String::new());

        match target {
            "build" | "check" | "clean" | "test" => {
                if workspace == Workspace::Internal {
                    // Optimise the internal workspace.
                    target_args.push("--release");
                    // Set the tracermode cfg macro, but without changing anything relating to code
                    // generation. We can't use `-C tracer=hw` as this would turn off optimisations.
                    rust_flags.push_str(" --cfg tracermode=\"hw\"");

                    // `cargo test` in the internal workspace won't build libykshim.so, so we have
                    // to force-build it to avoid linkage problems for the external workspace.
                    if target == "test" {
                        forced_deps.push(WorkspaceAction::new(Workspace::Internal, "build"));
                    }
                } else {
                    // Emit code suitable for hardware tracing.
                    rust_flags.push_str(" -C tracer=hw");

                    if target == "test" {
                        let append = [
                            Workspace::Internal.dir().to_str().unwrap(),
                            "target",
                            "release",
                        ]
                        .iter()
                        .collect::<PathBuf>();
                        ld_library_path.push_str(":");
                        ld_library_path.push_str(append.to_str().unwrap());
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
                tool = "rustup".to_owned();
                tool_args.extend(&["run", "nightly", "cargo"]);
            }
            _ => panic!("the yk build system doesn't support this target"),
        }

        Self {
            tool,
            workspace_dir: workspace.dir(),
            tool_args,
            target_args,
            rust_flags,
            forced_deps,
            ld_library_path,
        }
    }

    fn run(self, extra_args: &[String]) {
        // Run any dependencies first.
        for dep in self.forced_deps {
            dep.run(&[]);
        }

        env::set_current_dir(self.workspace_dir).unwrap();
        let mut cmd = Command::new(self.tool);
        let status = cmd
            .args(self.tool_args)
            .args(self.target_args)
            .args(extra_args)
            .env_remove("RUSTFLAGS")
            .env("RUSTFLAGS", self.rust_flags)
            .env("LD_LIBRARY_PATH", self.ld_library_path)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        if !status.success() {
            panic!("cargo failed");
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let target = args.next().unwrap();
    let extra_args = args.collect::<Vec<_>>();

    WorkspaceAction::new(Workspace::Internal, &target).run(&extra_args);
    WorkspaceAction::new(Workspace::External, &target).run(&extra_args);
}
