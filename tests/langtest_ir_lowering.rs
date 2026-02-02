//! A suite for testing lowerings of LLVM IR to Yk IR.

use lang_tester::LangTester;
use regex::Regex;
use std::{env, fs::read_to_string, path::PathBuf, process::Command};
use tempfile::TempDir;
use tests::full_cargo_profile;
use ykbuild::ykllvm_bin;

const COMMENT: &str = ";";

fn main() {
    println!("Running IR lowering tests...");

    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("ir_lowering")
        .test_path_filter(|p| p.extension().and_then(|p| p.to_str()) == Some("ll"))
        .test_extract(move |p| {
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

            // We don't use yk-config here, as we are testing one very specific functionality that
            // requires only one special flag.
            let mut compiler = Command::new(ykllvm_bin("clang"));
            let md = env::var("CARGO_MANIFEST_DIR").unwrap();
            let profile = full_cargo_profile();
            let ykcapi_path = [&md, "..", "target", &profile, "deps"]
                .iter()
                .collect::<PathBuf>();
            let ykcapi_linkdir = format!("-L{}", ykcapi_path.to_str().unwrap());
            compiler.args([
                "-flto",
                "-fuse-ld=lld",
                "-O0",
                "-o",
                exe.to_str().unwrap(),
                "-Wl,-mllvm=--yk-embed-ir",
                // The serialiser now assumes that we are doing software tracing.
                "-Wl,--mllvm=--yk-basicblock-tracer",
                // Enable conditional promote calls.
                "-Wl,--mllvm=--yk-conditional-promote-calls",
                // Use direct calls (bypassing PLT) for yk patchpoint functions.
                "-Wl,--mllvm=--yk-patchpoint-direct-functions-call",
                // Link libykcapi so that the tests inherit the necessary software tracing symbols.
                &ykcapi_linkdir,
                "-lykcapi",
                p.to_str().unwrap(),
            ]);
            let dumper_path = [&md, "..", "target", &profile, "dump_ir"]
                .iter()
                .collect::<PathBuf>();
            let mut dumper = Command::new(dumper_path);
            dumper.arg(exe);

            vec![("Compiler", compiler), ("Dump", dumper)]
        })
        .fm_options(|_, _, fmb| {
            // Use `{{}}` to match non-literal strings in tests.
            // E.g. use `%{{var}}` to capture the name of a variable.
            let ptn_re_ignore = Regex::new(r"\{\{_}\}").unwrap();
            let ptn_re = Regex::new(r"\{\{.+?\}\}").unwrap();
            let text_re = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
            fmb.name_matcher_ignore(ptn_re_ignore, text_re.clone())
                .name_matcher(ptn_re, text_re)
        })
        .run();
}
