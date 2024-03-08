//! Utilities for the yk build system.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

pub mod completion_wrapper;

/// Return the subdirectory of Cargo's `target` directory where we should be building things.
///
/// There are no guarantees about where this directory will be or what its name is.
fn target_dir() -> PathBuf {
    Path::new(env!("OUT_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}

/// Return a [Path] to the directory containing a ykllvm installation.
pub fn ykllvm_bin_dir() -> PathBuf {
    match env::var("YKB_YKLLVM_BIN_DIR") {
        Ok(x) => Path::new(&x).to_owned(),
        Err(_) => {
            // The directory returned here *must* be exactly the same as that produced by
            // `ykbuild/build.rs`.
            let mut ykllvm_dir = target_dir();
            ykllvm_dir.push("ykllvm");
            ykllvm_dir.push("bin");
            ykllvm_dir
        }
    }
}

/// Return the location of the ykllvm binary `bin_name`.
///
/// # Panics
///
/// If `bin_name` is not found.
pub fn ykllvm_bin(bin_name: &str) -> PathBuf {
    let mut p = ykllvm_bin_dir();
    p.push(bin_name);
    if p.exists() {
        return p;
    }
    panic!("ykllvm binary {} not found", p.to_str().unwrap_or(bin_name))
}

/// Call from a build script to ensure that the LLVM libraries are in the loader path.
///
/// This is preferred to adding an rpath, as we wouldn't want to distribute binaries with
/// system-local rpaths inside.
pub fn apply_llvm_ld_library_path() {
    let lib_dir = Command::new(ykllvm_bin("llvm-config"))
        .arg("--link-shared")
        .arg("--libdir")
        .output()
        .unwrap()
        .stdout;
    let lib_dir = std::str::from_utf8(&lib_dir).unwrap();
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", lib_dir);
}
