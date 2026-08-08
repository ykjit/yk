//! Utilities for the yk build system.

use std::{
    env,
    path::{Path, PathBuf},
};

pub mod completion_wrapper;

/// Return the subdirectory of Cargo's `target` directory where we should be building things.
///
/// There are no guarantees about where this directory will be or what its name is.
pub fn target_dir() -> PathBuf {
    let target_dir = cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .expect("failed to run `cargo metadata`")
        .target_directory
        .into_std_path_buf();
    let profile = Path::new(env!("OUT_DIR"))
        .strip_prefix(&target_dir)
        .expect("OUT_DIR is not inside cargo's target directory")
        .components()
        .next()
        .unwrap()
        .as_os_str()
        .to_owned();
    let dir = target_dir.join(profile);
    assert!(dir.is_dir(), "{dir:?} is not a directory");
    dir
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
