use std::env;
use which::which;

const YKLLVM: &str = "../ykllvm/llvm";
const PROFILE: &str = "Release";

fn main() {
    if env::var("YKB_YKLLVM_INSTALL_DIR").is_ok() {
        println!(
            "cargo:ykllvm={}",
            env::var("YKB_YKLLVM_INSTALL_DIR").unwrap()
        );
        return;
    }

    // Ninja builds are faster, so use this if it's available.
    let use_ninja = which("ninja").is_ok();
    let is_debug = env::var("PROFILE").unwrap() == "debug";
    let nprocs = format!("-j {}", num_cpus::get());

    let mut ykllvm = cmake::Config::new(YKLLVM);
    ykllvm
        .profile(PROFILE)
        .generator(if use_ninja { "Ninja" } else { "Unix Makefiles" })
        .define("LLVM_INSTALL_UTILS", "ON")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("LLVM_ENABLE_PROJECTS", "lld;clang")
        .define(
            "LLVM_ENABLE_ASSERTIONS",
            if is_debug { "ON" } else { "OFF" },
        )
        // Due to an LLVM bug, PIE breaks our mapper, and it's not enough to pass
        // `-fno-pie` to clang for some reason:
        // https://github.com/llvm/llvm-project/issues/57085
        .define("CLANG_DEFAULT_PIE_ON_LINUX", "OFF")
        .build_arg(nprocs);

    if let Ok(args) = env::var("YKB_YKLLVM_BUILD_ARGS") {
        for arg in args.split(",") {
            if !arg.starts_with("-D") {
                panic!("YKB_YKLLVM_BUILD_ARGS must only contain -D arguments");
            }
            let (k, v) = arg.strip_prefix("-D").unwrap().split_once("=").unwrap();
            ykllvm.define(k, v);
        }
    }

    let dsp = ykllvm.build();

    // We need to be able to locate the ykllvm install llvm bins from other
    // crates, so this sets a `DEP_YKBUILD_YKLLVM` env var which can be accessed
    // from any other crate in the yk workspace.
    println!("cargo:ykllvm={}/bin/", dsp.display())
}