//! Utilities for the yk build system.

use std::{
    env,
    path::{Path, PathBuf},
};

pub mod completion_wrapper;

/// Cargo's top-level `target` directory (`target_directory` from `cargo metadata`), before any
/// `<triple>/<profile>` components are appended.
pub fn cargo_target_dir() -> PathBuf {
    cargo_metadata::MetadataCommand::new()
        .no_deps()
        .exec()
        .expect("failed to run `cargo metadata`")
        .target_directory
        .into_std_path_buf()
}

/// Return the subdirectory of Cargo's `target` directory where we should be building things.
///
/// There are no guarantees about where this directory will be or what its name is.
pub fn target_dir() -> PathBuf {
    let target_dir = cargo_target_dir();
    // OUT_DIR is `<target_dir>/[<triple>/]<profile>/build/<pkg>-<hash>/out`. The `<triple>/`
    // component is only present when cargo was invoked with an explicit `--target`, so we can't
    // assume a fixed depth for `<profile>`.
    let out_dir = Path::new(env!("OUT_DIR"))
        .strip_prefix(&target_dir)
        .expect("OUT_DIR is not inside cargo's target directory");
    let build_dir_idx = out_dir
        .components()
        .position(|c| c.as_os_str() == "build")
        .expect("couldn't find `build` component in OUT_DIR");
    let profile = out_dir
        .components()
        .take(build_dir_idx)
        .collect::<PathBuf>();
    let target_dir = target_dir.join(profile);
    assert!(target_dir.is_dir());
    target_dir
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
