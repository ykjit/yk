#![feature(lazy_cell)]

use lang_tester::LangTester;
use regex::Regex;
use std::{
    env,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;
use tests::{mk_compiler, EXTRA_LINK};
use ykbuild::{completion_wrapper::CompletionWrapper, ykllvm_bin};

const COMMENT: &str = "//";

fn run_suite(opt: &'static str) {
    println!("Running C tests with opt level {}...", opt);

    let tempdir = TempDir::new().unwrap();

    // Generate a `compile_commands.json` database for clangd.
    let ccg = CompletionWrapper::new(ykllvm_bin("clang"), "c_tests");
    for (k, v) in ccg.build_env() {
        env::set_var(k, v);
    }
    let wrapper_path = ccg.wrapper_path();

    // Set variables for tests `ignore-if`s.
    #[cfg(cargo_profile = "debug")]
    env::set_var("YK_CARGO_PROFILE", "debug");
    #[cfg(cargo_profile = "release")]
    env::set_var("YK_CARGO_PROFILE", "release");

    #[cfg(target_arch = "x86_64")]
    env::set_var("YK_ARCH", "x86_64");
    #[cfg(not(target_arch = "x86_64"))]
    panic!("Unknown target_arch");

    let filter = match env::var("YKD_NEW_CODEGEN") {
        Ok(x) if x == "1" => {
            env::set_var("YK_JIT_COMPILER", "yk");
            |p: &Path| {
                // A temporary hack because at the moment virtually no tests run on the new JIT
                // compiler.
                p.extension().as_ref().and_then(|p| p.to_str()) == Some("c")
                    && p.file_name().unwrap().to_str().unwrap().contains(".newcg")
            }
        }
        _ => {
            env::set_var("YK_JIT_COMPILER", "llvm");
            |p: &Path| p.extension().as_ref().and_then(|p| p.to_str()) == Some("c")
        }
    };

    LangTester::new()
        .comment_prefix("#")
        .test_dir("c")
        .test_path_filter(filter)
        .test_extract(move |p| {
            let altp = p.with_extension(format!("c.{}", opt.strip_prefix('-').unwrap()));
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

            // Decide if we have extra objects to link to the test.
            let key = p.file_name().unwrap().to_str().unwrap();
            let extra_objs = EXTRA_LINK
                .get(key)
                .unwrap_or(&Vec::new())
                .iter()
                .map(|l| l.generate_obj(tempdir.path()))
                .collect::<Vec<PathBuf>>();

            let mut compiler = mk_compiler(wrapper_path.as_path(), &exe, p, opt, &extra_objs, true);
            compiler.env("YK_COMPILER_PATH", ykllvm_bin("clang"));
            let runtime = Command::new(exe.clone());
            vec![("Compiler", compiler), ("Run-time", runtime)]
        })
        .fm_options(|_, _, fmb| {
            // Use `{{}}` to match non-literal strings in tests.
            // E.g. use `%{{var}}` to capture the name of a variable.
            let ptn_re = Regex::new(r"\{\{.+?\}\}").unwrap();
            let text_re = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
            fmb.name_matcher(ptn_re, text_re)
        })
        .run();
    ccg.generate();
}

fn main() {
    // For now we can only compile with -O0 since higher optimisation levels introduce machine code
    // we currently don't know how to deal with, e.g. temporaries which break stackmap
    // reconstruction. This isn't a huge problem as in the future we will keep two versions of the
    // interpreter around and only swap to -O0 when tracing and run on higher optimisation levels
    // otherwise.
    run_suite("-O0");
}
