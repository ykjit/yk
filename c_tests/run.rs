use lang_tester::LangTester;
use once_cell::sync::Lazy;
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
static EXTRA_LINK: Lazy<HashMap<&'static str, Vec<ExtraLinkage>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "call_ext_in_obj.c",
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

    compiler.args(&[
        opt,
        #[cfg(debug_assertions)]
        "-g",
        "-Werror",
        "-Wall",
        "-fuse-ld=lld",
        "-flto",
        "-Wl,--mllvm=--embed-bitcode-final",
        "-Wl,--lto-basic-block-sections=labels",
        // FIXME: https://github.com/ykjit/yk/issues/381
        // Find a better way of handling unexported globals inside a trace.
        "-Wl,--export-dynamic",
        "-I",
        ykcapi_dir.to_str().unwrap(),
        "-L",
        lib_dir_str,
        &format!("-Wl,-rpath={}", lib_dir_str),
        "-lykcapi",
        "-pthread",
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
        p.extension().unwrap().to_str().unwrap() == "c"
            && !p
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("debug_")
    };
    #[cfg(cargo_profile = "debug")]
    let filter: fn(&Path) -> bool = |p| p.extension().unwrap().to_str().unwrap() == "c";

    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("tests")
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
