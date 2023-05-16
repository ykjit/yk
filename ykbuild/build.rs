#![feature(exit_status_error)]

use fs4::FileExt;
use std::{
    env,
    fs::{canonicalize, create_dir_all, read_to_string, write, File},
    path::Path,
    process::Command,
};
use which::which;

/// Where is ykllvm's source code relative to this crate?
const YKLLVM_SUBMODULE_PATH: &str = "../ykllvm/llvm";

fn main() {
    // If the user defines YKB_YKLLVM_BIN_DIR then we don't try to build ykllvm ourselves.
    if env::var("YKB_YKLLVM_BIN_DIR").is_ok() {
        return;
    }

    // To avoid running cmake config/build/install, we only do anything with cmake if either a) no
    // build of ykllvm exists b) the ykllvm submodule's hash has changed since we last built it.
    //
    // First we get the ykllvms submodule's hash.
    let out = Command::new("git")
        .args(["rev-parse", ":ykllvm"])
        .output()
        .unwrap();
    assert!(out.status.exit_ok().is_ok());
    let ykllvm_hash = String::from_utf8(out.stdout).unwrap();

    // Build ykllvm in "target/[debug|release]". Note that the directory used here *must*
    // be exactly the same as that produced by `ykbuild/src/lib.rs:llvm_bin_dir` and
    // yk-config.
    let mut ykllvm_dir = Path::new(&env::var("OUT_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    {
        let leaf = ykllvm_dir.file_name().unwrap().to_str().unwrap();
        assert!(leaf == "debug" || leaf == "release");
    }
    ykllvm_dir.push("ykllvm");
    create_dir_all(&ykllvm_dir).unwrap();

    let mut cached_hash_path = ykllvm_dir.clone();
    cached_hash_path.push("hash");
    if cached_hash_path.exists() {
        let cached_hash = read_to_string(&cached_hash_path).unwrap();
        // We use trim() here because git's output -- and possibly anyone manually edits the `hash`
        // file -- is likely to leave a trailing newline.
        if ykllvm_hash.trim() == cached_hash.trim() {
            // The submodule hash hasn't changed since the last build, so there's nothing to do.
            return;
        }
    }

    // We now know we want to build ykllvm. However, cargo can -- and in release mode does! -- run
    // more than 1 copy of this build script in parallel. We thus need to make sure that we don't
    // try and build multiple copies of ykllvm in the same directory at the same time, because that
    // leads to the sort of hilarity where no-one laughs.
    //
    // To avoid parallel builds stomping on each other, we use a lock file to which we gain
    // exclusive access: the first build script to do this wins, and any other build scripts that
    // try to do so then sleep until the first has completed. File locking is a cross-platform
    // nightmare, especially if you try to do something clever. Fortunately, we're not trying to
    // make something clever like a read-write lock: we just gain exclusive (which probably means
    // "write" on most platforms) access. Unix and Windows both remove the lock if the process dies
    // unexpectedly (though Windows says, in essence, that it doesn't guarantee to do the unlocking
    // very quickly), so if we fail while building, this should not cause parallel runs of this
    // build script to deadlock.
    let mut lock_path = ykllvm_dir.clone();
    lock_path.push("build_lock");
    let lock_file = File::create(lock_path).unwrap();
    lock_file.lock_exclusive().unwrap();

    // We execute three commands, roughly:
    //   cmake <configure>
    //   cmake --build .
    //   cmake --install .
    // We have to build up the precise command in steps.

    let mut build_dir = ykllvm_dir.clone();
    build_dir.push("build");
    create_dir_all(&build_dir).unwrap();

    // If ninja is available use that, otherwise use standard "make".
    let mut generator = which("ninja")
        .map(|_| "Ninja")
        .unwrap_or("Unix Makefiles")
        .to_owned();
    let mut cfg_cmd = Command::new("cmake");
    cfg_cmd
        .args([
            &format!(
                "-DCMAKE_INSTALL_PREFIX={}",
                ykllvm_dir.as_os_str().to_str().unwrap()
            ),
            "-DLLVM_INSTALL_UTILS=On",
            "-DCMAKE_BUILD_TYPE=release",
            #[cfg(debug_assertions)]
            "-DLLVM_ENABLE_ASSERTIONS=On",
            #[cfg(not(debug_assertions))]
            "-DLLVM_ENABLE_ASSERTIONS=Off",
            "-DLLVM_ENABLE_PROJECTS=lld;clang",
            // We have to turn off PIE due to: https://github.com/llvm/llvm-project/issues/57085
            "-DCLANG_DEFAULT_PIE_ON_LINUX=OFF",
            "-DBUILD_SHARED_LIBS=ON",
        ])
        .current_dir(build_dir.as_os_str().to_str().unwrap());

    let mut build_cmd = Command::new("cmake");
    build_cmd
        .args(["--build", "."])
        .current_dir(build_dir.as_os_str().to_str().unwrap());

    let mut inst_cmd = Command::new("cmake");
    inst_cmd
        .args(["--install", "."])
        .current_dir(build_dir.as_os_str().to_str().unwrap());

    if let Ok(args) = env::var("YKB_YKLLVM_BUILD_ARGS") {
        // Caveat: this assumes no cmake argument contains a ',' or a ':'.
        for arg in args.split(',') {
            match arg.split(':').collect::<Vec<_>>()[..] {
                ["define", x] => {
                    cfg_cmd.arg(x);
                }
                ["build_arg", x] => {
                    build_cmd.arg(x);
                }
                ["generator", x] => {
                    generator = x.to_owned();
                }
                [k, _] => panic!("Unknown kind {k}"),
                _ => panic!("Incorrectly formatted option {arg}"),
            }
        }
    }

    cfg_cmd.arg(&format!("-G{generator}"));
    cfg_cmd.arg(
        canonicalize(YKLLVM_SUBMODULE_PATH)
            .unwrap()
            .as_os_str()
            .to_str()
            .unwrap(),
    );

    if generator == "Unix Makefiles" {
        build_cmd.args(["-j", num_cpus::get().to_string().as_str()]);
    }

    cfg_cmd.status().unwrap().exit_ok().unwrap();
    build_cmd.status().unwrap().exit_ok().unwrap();
    inst_cmd.status().unwrap().exit_ok().unwrap();

    // We only update the hash if we're successful: if we're not successful, we expect that the
    // user will change something in the wider environment and expect us to rerun things.
    write(cached_hash_path, ykllvm_hash).unwrap();

    // We don't particularly need to unlock manually, but this might help the OS clean the lock up
    // sooner (Windows suggests that if a process leaves it to the OS to do the unlocking
    // automatically, it might not be particularly speedy) and allow parallel runs to advance
    // quicker.
    lock_file.unlock().unwrap();
}
