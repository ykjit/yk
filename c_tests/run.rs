use lang_tester::LangTester;
use std::{
    env,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;

const COMMENT: &str = "//";

/// Make a compiler command that compiles `src` to `exe`.
fn mk_compiler(exe: &Path, src: &Path) -> Command {
    let mut compiler = Command::new("clang");
    let mut lib_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    lib_dir.push("..");
    lib_dir.push("target");

    #[cfg(cargo_profile = "debug")]
    lib_dir.push("debug");
    #[cfg(cargo_profile = "release")]
    lib_dir.push("release");

    compiler.args(&[
        "-fuse-ld=lld",
        "-flto",
        "-Wl,--plugin-opt=-lto-embed-bitcode=optimized",
        "-L",
        lib_dir.to_str().unwrap(),
        "-lyktrace",
        "-o",
        exe.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    compiler
}

fn main() {
    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("tests")
        .test_file_filter(|p| p.extension().unwrap().to_str().unwrap() == "c")
        .test_extract(|p| {
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
            let compiler = mk_compiler(&exe, p);
            let runtime = Command::new(exe.clone());
            vec![("Compiler", compiler), ("Run-time", runtime)]
        })
        .run();
}
