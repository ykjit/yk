#![feature(exit_status_error)]

use fs4::FileExt;
use rerun_except::rerun_except;
use std::{
    collections::HashMap,
    env,
    fs::{canonicalize, create_dir_all, read_to_string, write, File},
    path::Path,
    process::Command,
};
use which::which;

/// The path where we store "KEY=VALUE" pairs for environment variables relevant to ykbuild.
const ENV_VARS_LEAF: &str = "yk_env_vars";
/// Which environment variables need to be cached, as changes in their value require us to
/// rerun the full cmake configure process?
const ENV_VARS_RERUN: &[&str] = &[
    "CC",
    "CFLAGS",
    "CXX",
    "CPPFLAGS",
    "LD",
    "LDFLAGS",
    "YKB_YKLLVM_BIN_DIR",
    "YKB_YKLLVM_BUILD_ARGS",
];
/// The path we use to determine which ykllvm files will cause this build.rs file to be rerun.
const YKLLVM_SRC_DEPEND_PATH: &str = "../ykllvm";
/// The path we use to determine that the ykllvm submodule has been cloned.
const YKLLVM_SUBMODULE_PATH: &str = "../ykllvm/llvm";

fn main() {
    for k in ENV_VARS_RERUN {
        println!("cargo::rerun-if-env-changed={}", k);
    }

    // If the user defines YKB_YKLLVM_BIN_DIR then we don't try to build ykllvm ourselves.
    if env::var("YKB_YKLLVM_BIN_DIR").is_ok() {
        return;
    }

    if !Path::new(YKLLVM_SUBMODULE_PATH).is_dir() {
        panic!("YKLLVM Submodule ({}) was not found! To check submodules, run:\n $ git submodule update --init --recursive\n", YKLLVM_SUBMODULE_PATH);
    }

    println!("cargo::rerun-if-changed={YKLLVM_SRC_DEPEND_PATH}");
    rerun_except(&[]).unwrap();

    // Build ykllvm in "target/<cargo-profile>". Note that the directory used here *must* be
    // exactly the same as that produced by `ykbuild/src/lib.rs:llvm_bin_dir` and yk-config.
    let mut ykllvm_dir = Path::new(&env::var("OUT_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    ykllvm_dir.push("ykllvm");
    create_dir_all(&ykllvm_dir).unwrap();

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
    //   cmake <configure> # Can be skipped in some cases
    //   cmake --build .
    //   cmake --install .
    // We have to build up the precise command in steps.

    let mut build_dir = ykllvm_dir.clone();
    build_dir.push("build");

    let mut cached_env_vars = ykllvm_dir.clone();
    cached_env_vars.push(ENV_VARS_LEAF);
    let run_cfg_cmd = if cached_env_vars.exists() {
        let mut new_vars = HashMap::new();
        for k in ENV_VARS_RERUN {
            new_vars.insert((*k).to_owned(), env::var(k).unwrap_or("".to_owned()));
        }
        let mut cached_vars = HashMap::new();
        for l in read_to_string(&cached_env_vars).unwrap().lines() {
            let l = l.trim();
            if l.is_empty() {
                continue;
            }
            let v = l.splitn(2, '=').collect::<Vec<_>>();
            assert_eq!(v.len(), 2);
            cached_vars.insert(v[0].to_owned(), v[1].to_owned());
        }
        new_vars != cached_vars
    } else {
        true
    };

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
            "-DLLVM_ENABLE_PROJECTS=lld;clang;clang-tools-extra",
            // We have to turn off PIE due to: https://github.com/llvm/llvm-project/issues/57085
            "-DCLANG_DEFAULT_PIE_ON_LINUX=OFF",
            "-DBUILD_SHARED_LIBS=ON",
        ])
        .current_dir(build_dir.as_os_str().to_str().unwrap());

    let mut build_cmd = Command::new("cmake");
    let mut build_args = vec!["--build".into(), ".".into()];
    if let Ok(jobs) = env::var("NUM_JOBS") {
        build_args.push("-j".into());
        build_args.push(jobs);
    }
    build_cmd
        .args(build_args)
        .current_dir(build_dir.as_os_str().to_str().unwrap());

    let mut inst_cmd = Command::new("cmake");
    inst_cmd
        .args(["--install", "."])
        .current_dir(build_dir.as_os_str().to_str().unwrap());

    // If ninja is available use that, otherwise use standard "make".
    let mut generator = which("ninja")
        .map(|_| "Ninja")
        .unwrap_or("Unix Makefiles")
        .to_owned();
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
                    x.clone_into(&mut generator);
                }
                [k, _] => panic!("Unknown kind {k}"),
                _ => panic!("Incorrectly formatted option {arg}"),
            }
        }
    }

    if run_cfg_cmd {
        create_dir_all(&build_dir).unwrap();

        cfg_cmd.arg(format!("-G{generator}"));
        cfg_cmd.arg(
            canonicalize(YKLLVM_SUBMODULE_PATH)
                .unwrap()
                .as_os_str()
                .to_str()
                .unwrap(),
        );

        cfg_cmd.status().unwrap().exit_ok().unwrap();
    }

    build_cmd.status().unwrap().exit_ok().unwrap();
    inst_cmd.status().unwrap().exit_ok().unwrap();

    // We only update the env_vars if we're successful: if we're not successful, we expect that the
    // user will change something in the wider environment and expect us to rerun things.
    let env_vars = ENV_VARS_RERUN
        .iter()
        .map(|k| format!("{k}={}", env::var(k).unwrap_or_else(|_| "".to_owned())))
        .collect::<Vec<String>>();
    write(cached_env_vars, env_vars.join("\n")).unwrap();

    // We don't particularly need to unlock manually, but this might help the OS clean the lock up
    // sooner (Windows suggests that if a process leaves it to the OS to do the unlocking
    // automatically, it might not be particularly speedy) and allow parallel runs to advance
    // quicker.
    lock_file.unlock().unwrap();
}
