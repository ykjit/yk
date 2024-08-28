//! A tool to run a C test under gdb.

use clap::Parser;
use std::{env, path::PathBuf, process::Command};
use tempfile::TempDir;
use tests::{mk_compiler, EXTRA_LINK};
use ykbuild::ykllvm_bin;

/// Run a C test under gdb.
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// The test to attach gdb to.
    test_file: PathBuf,

    /// Run the test with `YKD_LOG_IR` set to the specified value.
    #[arg(short, long)]
    log_ir: Option<String>,

    /// Run the test with `YKD_LOG_JITSTATE=-`
    #[arg(short = 'j', long)]
    log_jitstate: bool,

    /// Run the test with `YKD_SERIALISE_COMPILATION=1`
    #[arg(short, long)]
    serialise_compilation: bool,

    /// Set breakpoints at the first `N` compiled traces.
    #[arg(short = 'b', long)]
    num_breaks: Option<usize>,

    /// Don't immediately run the program.
    #[arg(short = 'n', long)]
    wait_at_prompt: bool,
}

fn main() {
    let args = Args::parse();

    let md = env::var("CARGO_MANIFEST_DIR").unwrap();
    let test_path = [&md, "c", (args.test_file.to_str().unwrap())]
        .iter()
        .collect::<PathBuf>();
    let tempdir = TempDir::new().unwrap();

    // Compile the test.
    //
    // Some tests expect to have extra objects linked.
    let extra_objs = EXTRA_LINK
        .get(&test_path.to_str().unwrap())
        .unwrap_or(&Vec::new())
        .iter()
        .map(|e| e.generate_obj(tempdir.path()))
        .collect::<Vec<PathBuf>>();

    let binstem = PathBuf::from(args.test_file.file_stem().unwrap());
    let binpath = [tempdir.path(), &binstem].iter().collect::<PathBuf>();
    let mut cmd = mk_compiler(
        ykllvm_bin("clang").as_path(),
        &binpath,
        &test_path,
        &extra_objs,
        true,
        None,
    );
    if !cmd.spawn().unwrap().wait().unwrap().success() {
        panic!("compilation failed");
    }

    // Now we have a test binary in a temporary directory, prepare an invocation of gdb, setting
    // environment variables as necessary.
    let mut gdb = Command::new("gdb");
    gdb.arg(&binpath);

    if args.serialise_compilation {
        gdb.env("YKD_SERIALISE_COMPILATION", "1");
    }

    if args.log_jitstate {
        gdb.env("YKD_LOG_JITSTATE", "1");
    }

    if let Some(irs) = args.log_ir {
        gdb.env("YKD_LOG_IR", irs);
    }

    if let Some(num_breaks) = args.num_breaks {
        gdb.args(["-ex", "set breakpoint pending on"]); // don't prompt for pending breaks.
        for i in 0..num_breaks {
            gdb.args(["-ex", &format!("b __yk_compiled_trace_{i}")]);
        }
    }

    if !args.wait_at_prompt {
        gdb.args(["-ex", "run"]);
    }

    // Run gdb!
    gdb.spawn().expect("failed to spawn gdb").wait().unwrap();
}
