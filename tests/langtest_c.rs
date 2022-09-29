#![feature(once_cell)]

use lang_tester::LangTester;
use regex::Regex;
use std::sync::LazyLock;
use std::{
    collections::HashMap,
    env,
    fs::read_to_string,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;

const COMMENT: &str = "//";

const TEMPDIR_SUBST: &'static str = "%%TEMPDIR%%";
static EXTRA_LINK: LazyLock<HashMap<&'static str, Vec<ExtraLinkage>>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    // These tests get an extra, separately compiled (thus opaque to LTO), object file linked in.
    for test_file in &[
        "call_ext_in_obj.c",
        "loopy_funcs_not_inlined_by_default.c",
        "not_loopy_funcs_inlined_by_default.c",
        "unroll_safe_implies_noinline.c",
        "unroll_safe_inlines.c",
        "yk_unroll_safe_vs_yk_outline.c",
    ] {
        map.insert(
            *test_file,
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
    }
    map
});

/// Describes an extra object file to link to a C test.
struct ExtraLinkage<'a> {
    /// The name of the object file to be generated.
    output_file: &'a str,
    /// The command that generates the object file.
    gen_cmd: &'a [&'a str],
}

impl<'a> ExtraLinkage<'a> {
    fn new(output_file: &'a str, gen_cmd: &'a [&'a str]) -> Self {
        Self {
            output_file,
            gen_cmd,
        }
    }

    /// Run the command to generate the object in `tempdir` and return the absolute path to the
    /// generated object.
    fn generate_obj(&self, tempdir: &Path) -> PathBuf {
        let mut cmd = Command::new(self.gen_cmd[0]);
        let tempdir_s = tempdir.to_str().unwrap();
        for arg in self.gen_cmd[1..].iter() {
            cmd.arg(arg.replace(TEMPDIR_SUBST, tempdir_s));
        }
        let out = cmd.output().unwrap();
        assert!(tempdir.exists());
        if !out.status.success() {
            io::stdout().write_all(&out.stdout).unwrap();
            io::stderr().write_all(&out.stderr).unwrap();
            panic!();
        }
        let mut ret = PathBuf::from(tempdir);
        ret.push(&self.output_file.replace(TEMPDIR_SUBST, tempdir_s));
        ret
    }
}

/// Make a compiler command that compiles `src` to `exe` using the optimisation flag `opt`.
/// `extra_objs` is a collection of other object files to link.
fn mk_compiler(exe: &Path, src: &Path, opt: &str, extra_objs: &[PathBuf]) -> Command {
    let mut compiler = Command::new("clang");
    compiler.env("YKD_PRINT_IR", "1");

    let mut lib_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    lib_dir.push("..");
    lib_dir.push("target");
    #[cfg(cargo_profile = "debug")]
    lib_dir.push("debug");
    #[cfg(cargo_profile = "release")]
    lib_dir.push("release");
    lib_dir.push("deps");
    let lib_dir_str = lib_dir.to_str().unwrap();

    let mut ykcapi_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    ykcapi_dir.push("..");
    ykcapi_dir.push("ykcapi");

    // FIXME: https://github.com/ykjit/yk/issues/427
    // yk requires a lot of ykllvm compiler flags. We should consider adding "one main yk flag"
    // which turns on all of the other flags.
    compiler.args(&[
        opt,
        // If this is a debug build, include debug info in the test binary.
        #[cfg(debug_assertions)]
        "-g",
        // Be strict.
        "-Werror",
        "-Wall",
        // Enable LTO with lld.
        "-fuse-ld=lld",
        "-flto",
        // Outline functions containing loops during AOT compilation. Needed for `yk_unroll_safe`.
        "-fyk-noinline-funcs-with-loops",
        // Embed LLVM bitcode as late as possible.
        "-Wl,--mllvm=--embed-bitcode-final",
        // Disable machine passes that would interfere with block mapping.
        //
        // If you are trying to figure out which pass is breaking the mapping, you can add
        // "-Wl,--mllvm=--print-before-all"/"-Wl,--mllvm=--print-after-all" to see the MIR
        // before/after each pass. You can make the output smaller by filtering the output by
        // function name with "-Wl,--mllvm=--filter-print-funcs=<func>". When you have found the
        // candidate, look in `TargetPassConfig.cpp` (in ykllvm) to find the CLI switch required to
        // disable the pass. If you can't (or don't want to) eliminate a whole pass, then you can
        // add (or re-use) a yk-specific flag to disable only aspects of passes.
        "-Wl,--mllvm=--disable-branch-fold",
        "-Wl,--mllvm=--disable-block-placement",
        "-Wl,--mllvm=--disable-early-taildup", // Interferes with the BlockDisambiguate pass.
        "-Wl,--mllvm=--disable-tail-duplicate", // ^^^
        "-Wl,--mllvm=--yk-disable-tail-call-codegen", // Interferes with the JIT's inlining stack.
        "-Wl,--mllvm=--yk-no-fallthrough",     // Fallthrough optimisations distort block mapping.
        // Ensure control point is patched.
        "-Wl,--mllvm=--yk-patch-control-point",
        // Emit stackmaps used for JIT deoptimisation.
        "-Wl,--mllvm=--yk-insert-stackmaps",
        // Ensure we can unambiguously map back to LLVM IR blocks.
        "-Wl,--mllvm=--yk-block-disambiguate",
        // Have the `.llvmbc` section loaded into memory by the loader.
        "-Wl,--mllvm=--yk-alloc-llvmbc-section",
        // Emit a basic block map section. Used for block mapping.
        "-Wl,--lto-basic-block-sections=labels",
        // FIXME: https://github.com/ykjit/yk/issues/381
        // Find a better way of handling unexported globals inside a trace.
        "-Wl,--export-dynamic",
        // Ensure the tests can find and use libykcapi.
        "-I",
        ykcapi_dir.to_str().unwrap(),
        "-L",
        lib_dir_str,
        "-lykcapi",
        // Encode an rpath so that we don't have to set LD_LIBRARY_PATH.
        &format!("-Wl,-rpath={}", lib_dir_str),
        // Some tests are multi-threaded via the pthread API.
        "-pthread",
        // The input and output files.
        "-o",
        exe.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    compiler.args(extra_objs);
    compiler
}

fn run_suite(opt: &'static str) {
    println!("Running C tests with {}...", opt);

    // Tests with the filename prefix `debug_` are only run in debug builds.
    #[cfg(cargo_profile = "release")]
    let filter: fn(&Path) -> bool = |p| {
        if let Some(ext) = p.extension() {
            ext == "c"
                && !p
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("debug_")
        } else {
            false
        }
    };
    #[cfg(cargo_profile = "debug")]
    let filter: fn(&Path) -> bool = |p| {
        if let Some(ext) = p.extension() {
            ext == "c"
        } else {
            false
        }
    };

    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("c")
        .test_file_filter(filter)
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

            // Decide if we have extra objects to link to the test.
            let key = p.file_name().unwrap().to_str().unwrap();
            let extra_objs = EXTRA_LINK
                .get(key)
                .unwrap_or(&Vec::new())
                .iter()
                .map(|l| l.generate_obj(tempdir.path()))
                .collect::<Vec<PathBuf>>();

            let compiler = mk_compiler(&exe, p, opt, &extra_objs);
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
    // For now we can only compile with -O0 since higher optimisation levels introduce machine code
    // we currently don't know how to deal with, e.g. temporaries which break stackmap
    // reconstruction. This isn't a huge problem as in the future we will keep two versions of the
    // interpreter around and only swap to -O0 when tracing and run on higher optimisation levels
    // otherwise.
    run_suite("-O0");
}
