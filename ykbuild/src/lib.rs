//! Utilities for the yk build system.

use std::{env, process::Command};

pub mod ccgen;

fn manifest_dir() -> String {
    env::var("CARGO_MANIFEST_DIR").unwrap()
}

pub fn llvm_config() -> Command {
    let mut c = Command::new("llvm-config");
    if let (Ok(bin_dir), Ok(path)) = (env::var("YKB_YKLLVM_INSTALL_DIR"), env::var("PATH")) {
        c.env("PATH", format!("{bin_dir}:{path}"));
    }
    c.arg("--link-shared");
    c
}

/// Call from a build script to ensure that the LLVM libraries are in the loader path.
///
/// This is preferred to adding an rpath, as we wouldn't want to distribute binaries with
/// system-local rpaths inside.
pub fn apply_llvm_ld_library_path() {
    let lib_dir = llvm_config().arg("--libdir").output().unwrap().stdout;
    let lib_dir = std::str::from_utf8(&lib_dir).unwrap();
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", lib_dir);
}
