#![feature(fn_traits)]

use rerun_except::rerun_except;
use std::{env, process::Command};
use ykbuild::{apply_llvm_ld_library_path, completion_wrapper::CompletionWrapper, ykllvm_bin};

fn main() {
    // Ensure changing C++ source files or headers retriggers a build.
    rerun_except(&[]).unwrap();

    // Compile our wrappers with the right LLVM C++ flags.
    let cxxflags_out = Command::new(&ykllvm_bin("llvm-config"))
        .arg("--link-shared")
        .arg("--cxxflags")
        .output()
        .unwrap()
        .stdout;
    let cxxflags_str = std::str::from_utf8(&cxxflags_out).unwrap();
    let cxxflags = cxxflags_str.split_whitespace().collect::<Vec<_>>();

    let mut comp = cc::Build::new();
    for cf in cxxflags {
        if cf.contains("-std=") {
            // llvm-config contains a -std=c++XY flag, but we want to force a particular
            // version below, so we override it.
            continue;
        }
        comp.flag(cf);
    }
    comp.flag("-Wall");
    comp.flag("-Werror");
    comp.flag("-std=c++20");
    comp.file("src/ykllvmwrap.cc")
        .file("src/jitmodbuilder.cc")
        .file("src/memman.cc")
        // Lots of unused parameters in the LLVM headers.
        .flag("-Wno-unused-parameter")
        .cpp(true);

    // If building with testing support, define a macro so we can conditionally compile stuff.
    #[cfg(feature = "yk_testing")]
    comp.flag("-DYK_TESTING");

    // Set the C NDEBUG macro if Cargo is building in release mode. This ensures that assertions
    // (and other things we guard with NDEBUG) only happen in debug builds.
    if env::var("PROFILE").unwrap() == "release" {
        comp.flag("-DNDEBUG");
    }

    // Generate a `compile_commands.json` database for clangd.
    let ccg = CompletionWrapper::new("ykllvmwrap");
    for (k, v) in ccg.build_env() {
        env::set_var(k, v);
    }
    comp.compiler(ccg.wrapper_path());

    // Actually do the compilation.
    comp.compile("ykllvmwrap");

    // Generate `compile_commands.json`.
    ccg.generate();

    // Ensure that downstream crates performing linkage use the right -L and -l flags.
    let lib_dir = Command::new(&ykllvm_bin("llvm-config"))
        .arg("--link-shared")
        .arg("--libdir")
        .output()
        .unwrap()
        .stdout;
    let lib_dir = std::str::from_utf8(&lib_dir).unwrap();
    println!("cargo:rustc-link-search={}", lib_dir);
    apply_llvm_ld_library_path();

    let libs = Command::new(&ykllvm_bin("llvm-config"))
        .arg("--link-shared")
        .arg("--libs")
        .output()
        .unwrap()
        .stdout;
    let libs = std::str::from_utf8(&libs).unwrap();
    for lib in libs.split_whitespace() {
        assert!(lib.starts_with("-l"));
        println!("cargo:rustc-link-lib={}", &lib[2..]);
    }
    println!("cargo:rustc-link-lib=tinfo");
    println!("cargo:rustc-link-lib=z");
}
