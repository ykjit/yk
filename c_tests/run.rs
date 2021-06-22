use lang_tester::LangTester;
use std::{
    env,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;

const COMMENT: &str = "//";

/// Make a compiler command that compiles `src` to `exe` using the optimisation flag `opt`.
fn mk_compiler(exe: &Path, src: &Path, opt: &str) -> Command {
    let mut compiler = Command::new("clang");
    compiler.env("YK_PRINT_IR", "1");

    let mut lib_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    lib_dir.push("..");
    lib_dir.push("target");
    #[cfg(cargo_profile = "debug")]
    lib_dir.push("debug");
    #[cfg(cargo_profile = "release")]
    lib_dir.push("release");
    lib_dir.push("deps");

    let mut ykcapi_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    ykcapi_dir.push("..");
    ykcapi_dir.push("ykcapi");

    compiler.args(&[
        opt,
        #[cfg(debug_assertions)]
        "-g",
        "-Werror",
        "-Wall",
        "-fuse-ld=lld",
        "-flto",
        "-Wl,--plugin-opt=-lto-embed-bitcode=optimized",
        "-Wl,--lto-basic-block-sections=labels",
        "-I",
        ykcapi_dir.to_str().unwrap(),
        "-L",
        lib_dir.to_str().unwrap(),
        "-lykcapi",
        "-pthread",
        "-o",
        exe.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    compiler
}

fn run_suite(opt: &'static str) {
    println!("Running C tests with {}...", opt);

    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("tests")
        .test_file_filter(|p| p.extension().unwrap().to_str().unwrap() == "c")
        .test_extract(move |p| {
            let altp = p.with_extension(format!("c.{}", opt.strip_prefix("-").unwrap()));
            let p = if altp.exists() { altp.as_path() } else { p };
            read_to_string(p)
                .unwrap()
                .lines()
                .skip_while(|l| !l.starts_with(COMMENT))
                .take_while(|l| l.starts_with(COMMENT))
                .map(|l| &l[COMMENT.len()..])
                .collect::<Vec<_>>()
                .join("\n")
        })
        .test_cmds(move |p| {
            let mut exe = PathBuf::new();
            exe.push(&tempdir);
            exe.push(p.file_stem().unwrap());
            let compiler = mk_compiler(&exe, p, opt);
            let runtime = Command::new(exe.clone());
            vec![("Compiler", compiler), ("Run-time", runtime)]
        })
        .run();
}

fn main() {
    // Run the suite with the various different clang optimisation levels. We do this to maximise
    // the possibility of shaking out bugs (in both the JIT and the tests themselves).
    run_suite("-O0");
    run_suite("-O1");
    run_suite("-O2");
    run_suite("-O3");
    run_suite("-Ofast");
    run_suite("-Os");
    run_suite("-Oz");
}
