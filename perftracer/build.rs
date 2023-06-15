use rerun_except::rerun_except;
use std::env;
use std::path::PathBuf;
use ykbuild::{completion_wrapper::CompletionWrapper, ykllvm_bin};

const FEATURE_CHECKS_PATH: &str = "feature_checks";

/// Simple feature check, returning `true` if we have the feature.
///
/// The checks themselves are in files under `FEATURE_CHECKS_PATH`.
fn feature_check(filename: &str, output_file: &str) -> bool {
    let mut path = PathBuf::new();
    path.push(FEATURE_CHECKS_PATH);
    path.push(filename);

    let mut check_build = cc::Build::new();
    check_build.file(path).try_compile(output_file).is_ok()
}

fn main() {
    let mut c_build = cc::Build::new();

    // Generate a `compile_commands.json` database for clangd.
    let ccg = CompletionWrapper::new(ykllvm_bin("clang"), "perftracer");
    for (k, v) in ccg.build_env() {
        env::set_var(k, v);
    }
    c_build.compiler(ccg.wrapper_path());

    // Check if perftracer can plausibly be built.
    if !cfg!(all(target_os = "linux", target_arch = "x86_64"))
        || !feature_check("check_perf.c", "check_perf")
    {
        panic!("perftracer is not supported on this OS and / or architecture");
    }

    #[cfg(target_arch = "x86_64")]
    println!("cargo:rustc-cfg=decoder_ykpt");

    c_build.file("src/collect/perf/collect.c");
    c_build.compile("perftracer_c");

    // Additional circumstances under which to re-run this build.rs.
    rerun_except(&[
        "README.md",
        "deny.toml",
        "LICENSE-*",
        "COPYRIGHT",
        "bors.toml",
        ".buildbot.sh",
    ])
    .unwrap();

    ccg.generate();
}
