//! Lang tester harness for trace compiler tests.

#![feature(once_cell)]

use lang_tester::LangTester;
use regex::Regex;
use std::{env, fs::read_to_string, path::PathBuf, process::Command};

const COMMENT: &str = ";";

fn main() {
    println!("Running trace compiler tests...");

    // Find the `run_trace_compiler_test` binary in the target dir.
    let md = env::var("CARGO_MANIFEST_DIR").unwrap();
    let mut run_tc_test = PathBuf::from(md);
    run_tc_test.push("..");
    run_tc_test.push("target");
    #[cfg(cargo_profile = "release")]
    run_tc_test.push("release");
    #[cfg(cargo_profile = "debug")]
    run_tc_test.push("debug");
    run_tc_test.push("run_trace_compiler_test");

    LangTester::new()
        .test_dir("trace_compiler")
        .test_file_filter(|p| {
            if let Some(ext) = p.extension() {
                ext == "ll"
            } else {
                false
            }
        })
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
            let mut run_tc_test = Command::new(run_tc_test.clone());
            run_tc_test.arg(p);
            vec![("Run-time", run_tc_test)]
        })
        .fm_options(|_, _, fmb| {
            // Use `{{}}` to match non-literal strings in tests.
            // E.g. use `%{{var}}` to capture the name of a variable.
            let ptn_re = Regex::new(r"\{\{.+?\}\}").unwrap();
            let text_re = Regex::new(r".+?\b").unwrap();
            fmb.name_matcher(ptn_re, text_re)
        })
        .run();
}
