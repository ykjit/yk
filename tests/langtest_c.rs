#![feature(lazy_cell)]

use lang_tester::LangTester;
use regex::Regex;
use std::{
    env,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
    sync::LazyLock,
};
use tempfile::TempDir;
use tests::{mk_compiler, EXTRA_LINK};
use ykbuild::{completion_wrapper::CompletionWrapper, ykllvm_bin};

const COMMENT: &str = "//";

static NEW_CODEGEN: LazyLock<bool> = LazyLock::new(|| {
    if let Ok(val) = env::var("YKD_NEW_CODEGEN") {
        if val == "1" {
            return true;
        }
    }
    false
});

static SUPPORTED_CPU_ARCHES: [&str; 1] = ["x86_64"];

/// Transitionary hack to allow the LLVM and new codegen to co-exist.
///
/// Once the new codegen is complete we can kill this.
fn correct_codegen_for_test(p: &Path) -> bool {
    let fname = p.file_name().unwrap().to_str().unwrap();
    if *NEW_CODEGEN {
        fname.contains(".newcg.")
    } else {
        !fname.contains(".newcg.")
    }
}

/// Get string name for the current CPU architecture.
fn arch() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        panic!("unknown CPU architecture");
    }
}

/// Check if we are on the right CPU for a test.
fn correct_cpu_arch(p: &Path) -> bool {
    let my_arch = arch();
    let other_arches = SUPPORTED_CPU_ARCHES.iter().filter(|a| *a != &my_arch);
    let p = p.to_str().unwrap();
    for oa in other_arches {
        if p.contains(oa) {
            return false;
        }
    }
    return true;
}

fn run_suite(opt: &'static str) {
    println!("Running C tests with opt level {}...", opt);

    fn is_c(p: &Path) -> bool {
        p.extension().as_ref().and_then(|p| p.to_str()) == Some("c")
    }

    // Tests with the filename prefix `debug_` are only run in debug builds.
    #[cfg(cargo_profile = "release")]
    let filter = |p: &Path| {
        is_c(p)
            && correct_cpu_arch(p)
            && correct_codegen_for_test(p)
            && !p
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("debug_")
    };
    #[cfg(cargo_profile = "debug")]
    let filter = |p: &Path| is_c(p) && correct_cpu_arch(p) && correct_codegen_for_test(p);

    let tempdir = TempDir::new().unwrap();

    // Generate a `compile_commands.json` database for clangd.
    let ccg = CompletionWrapper::new(ykllvm_bin("clang"), "c_tests");
    for (k, v) in ccg.build_env() {
        env::set_var(k, v);
    }
    let wrapper_path = ccg.wrapper_path();

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
