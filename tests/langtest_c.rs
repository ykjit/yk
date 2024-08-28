use lang_tester::LangTester;
use regex::Regex;
use std::{
    collections::HashMap,
    env,
    error::Error,
    fs::{read_to_string, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;
use tests::{mk_compiler, EXTRA_LINK};
use ykbuild::{completion_wrapper::CompletionWrapper, ykllvm_bin};

const COMMENT: &str = "//";
const COMMENT_PREFIX: &str = "##";

/// Parse any "extra environment" to pass to yk-config out of the test file.
fn parse_yk_config_env(p: &Path) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let f = File::open(p)?;
    let rdr = BufReader::new(&f);
    let mut extra = HashMap::new();
    let env_prefix = format!("{} {} yk-config-env:", COMMENT, COMMENT_PREFIX);
    for line in rdr.lines() {
        let line = line?;
        if !line.starts_with(COMMENT) {
            break; // won't find any lower down the file.
        }
        if line.starts_with(&env_prefix) {
            let sfx = &line[(&env_prefix).len()..];
            let mut elems = sfx.split("=");
            extra.insert(
                elems.next().ok_or("bad env spec")?.trim().into(),
                elems.next().ok_or("bad env spec")?.trim().into(),
            );
        }
    }
    Ok(extra)
}

fn main() {
    println!("Running C tests...");

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

    LangTester::new()
        .comment_prefix(COMMENT_PREFIX)
        .test_dir("c")
        .test_path_filter(|p: &Path| {
            p.extension().as_ref().and_then(|p| p.to_str()) == Some("c")
                && !p.file_name().unwrap().to_str().unwrap().contains(".old")
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

            let yk_config_env = parse_yk_config_env(p).unwrap();
            let mut compiler = mk_compiler(
                wrapper_path.as_path(),
                &exe,
                p,
                &extra_objs,
                true,
                Some(&yk_config_env),
            );
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
