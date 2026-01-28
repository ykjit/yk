//! A tool to run a C test under rr.

use clap::Parser;
use std::{env, path::PathBuf, process::Command};
use tempfile::TempDir;
use tests::{EXTRA_LINK, mk_compiler};
use ykbuild::ykllvm_bin;

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// The test to run under rr.
    test_file: PathBuf,

    /// Pass all arguments after `--` directly to rr.
    #[arg(last = true, required = false)]
    rr_args: Vec<String>,
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

    let mut rr = Command::new("rr");
    rr.arg(&binpath);

    // Pass all rr-specific arguments after '--'
    if !args.rr_args.is_empty() {
        for arg in &args.rr_args {
            rr.arg(arg);
        }
    }
    rr.spawn().expect("failed to spawn rr").wait().unwrap();
}
