use lang_tester::LangTester;
use regex::Regex;
use std::{
    env,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;
use tests::{EXTRA_LINK, mk_compiler};
use ykbuild::{completion_wrapper::CompletionWrapper, ykllvm_bin};

const COMMENT: &str = "//";
const COMMENT_PREFIX: &str = "##";

fn main() {
    println!("Running C tests...");

    let tempdir = TempDir::new().unwrap();

    // Generate a `compile_commands.json` database for clangd.
    let ccg = CompletionWrapper::new(ykllvm_bin("clang"), "c_tests");
    for (k, v) in ccg.build_env() {
        // While this is unsafe, this is only for clangd and doesn't affect the tests.
        unsafe { env::set_var(k, v) };
    }
    let wrapper_path = ccg.wrapper_path();

    // Set variables for tests `ignore-if`s.
    // These env vars stay the same for the entirety of the `cargo test` run, so we don't need to
    // worry about this being unsafe.
    #[cfg(cargo_profile = "debug")]
    unsafe {
        env::set_var("YK_CARGO_PROFILE", "debug")
    };
    #[cfg(cargo_profile = "release")]
    unsafe {
        env::set_var("YK_CARGO_PROFILE", "release")
    };

    // As with the above, this env var remains the same for the entire `cargo test` run.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        env::set_var("YK_ARCH", "x86_64")
    };
    #[cfg(not(target_arch = "x86_64"))]
    panic!("Unknown target_arch");

    // Ensure YKB_TRACER is set, so that tests don't have to consider what the default tracer is
    // when YKB_TRACER is absent from the env.
    #[cfg(tracer_swt)]
    unsafe {
        env::set_var("YKB_TRACER", "swt")
    };

    LangTester::new()
        .comment_prefix(COMMENT_PREFIX)
        .test_dir("c")
        .test_path_filter(|p: &Path| p.extension().as_ref().and_then(|p| p.to_str()) == Some("c"))
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

            // Decide if we have extra objects to link to the test.
            let key = p.file_name().unwrap().to_str().unwrap();
            let extra_objs = EXTRA_LINK
                .get(key)
                .unwrap_or(&Vec::new())
                .iter()
                .map(|l| l.generate_obj(tempdir.path()))
                .collect::<Vec<PathBuf>>();

            let mut compiler =
                mk_compiler(wrapper_path.as_path(), &exe, p, &extra_objs, true, None);
            compiler.env("YK_COMPILER_PATH", ykllvm_bin("clang"));
            let runtime = Command::new(exe.clone());
            vec![("Compiler", compiler), ("Run-time", runtime)]
        })
        .fm_options(|_, _, fmb| {
            // Use `{{}}` to match non-literal strings in tests.
            // E.g. use `%{{var}}` to capture the name of a variable.
            let ptn_re = Regex::new(r"\{\{.+?\}\}").unwrap();
            let ptn_re_ignore = Regex::new(r"\{\{_}\}").unwrap();
            let text_re = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
            fmb.name_matcher_ignore(ptn_re_ignore, text_re.clone())
                .name_matcher(ptn_re, text_re)
        })
        .run();
    ccg.generate();
}
