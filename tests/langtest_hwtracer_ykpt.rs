#![feature(lazy_cell)]

use lang_tester::LangTester;
use regex::Regex;
use std::{
    collections::HashMap,
    fs::read_to_string,
    path::{Path, PathBuf},
    process::Command,
    sync::LazyLock,
};
use tempfile::TempDir;
use tests::mk_compiler;
use tests::ExtraLinkage;
use ykbuild::ccgen::CCLang;

const COMMENT: &str = "//";

pub static EXTRA_LINK_HWTRACER_YKPT: LazyLock<HashMap<&'static str, Vec<ExtraLinkage>>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();

        // These tests get an extra, separately compiled (thus opaque to LTO), object file linked in.
        map.insert(
            "foreign.c",
            vec![ExtraLinkage::new(
                "%%TEMPDIR%%/call_me.o",
                &[
                    "clang",
                    "-c",
                    "-O0",
                    "extra_linkage/call_me.c",
                    "-o",
                    "%%TEMPDIR%%/call_me.o",
                ],
            )],
        );
        map.insert(
            "stifle_compressed_ret.c",
            vec![ExtraLinkage::new(
                "%%TEMPDIR%%/fudge.o",
                &[
                    "clang",
                    "-c",
                    "-O0",
                    "extra_linkage/fudge.s",
                    "-o",
                    "%%TEMPDIR%%/fudge.o",
                ],
            )],
        );

        map
    });

fn run_suite(opt: &'static str) {
    println!("Running hwtracer_ykpt tests with {}...", opt);

    let filter: fn(&Path) -> bool = |p| {
        if let Some(ext) = p.extension() {
            ext == "c"
        } else {
            false
        }
    };

    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("hwtracer_ykpt")
        .test_file_filter(filter)
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
            let extra_objs = EXTRA_LINK_HWTRACER_YKPT
                .get(key)
                .unwrap_or(&Vec::new())
                .iter()
                .map(|l| l.generate_obj(tempdir.path()))
                .collect::<Vec<PathBuf>>();

            let mut compiler = mk_compiler(
                CCLang::C.compiler_wrapper().to_str().unwrap(),
                &exe, p, opt, &extra_objs, false);
            compiler.arg("-ltests");
            let runtime = Command::new(exe.clone());
            vec![("Compiler", compiler), ("Run-time", runtime)]
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

fn main() {
    // See `main()` in `langtest_c.rs` for why we test only `-O0`.
    run_suite("-O0");
}
