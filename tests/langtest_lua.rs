#![feature(exit_status_error)]

use inner::main;

#[cfg(cargo_profile = "debug")]
mod inner {
    pub(super) fn main() {
        // Right now, these tests tend to cause "trace too long" too often to be usable in debug
        // mode.
        println!("<Lua tests are not run in debug mode>");
    }
}

#[cfg(cargo_profile = "release")]
mod inner {
    use fs4::fs_std::FileExt;
    use lang_tester::LangTester;
    use regex::Regex;
    use std::{
        env,
        fs::{File, canonicalize, create_dir_all, read_to_string, write},
        path::Path,
        process::{Command, exit},
    };
    use tests::full_cargo_profile;
    use ykbuild::{target_dir, ykllvm_bin_dir};

    const YKLUA_SUBMODULE_PATH: &str = "yklua";
    const YKLLLVM_SUBMODULE_PATH: &str = "../ykllvm";

    const COMMENT: &str = "--";
    const COMMENT_PREFIX: &str = "##";

    fn build() {
        // Build yklua for testing purposes.

        if !Path::new(&format!("{}/Makefile", YKLUA_SUBMODULE_PATH)).is_file() {
            panic!(
                "yklua submodule not found. To checkout:\n  git submodule update --init --recursive"
            );
        }

        let mut yklua_tgt_dir = target_dir();
        yklua_tgt_dir.push("yklua");
        create_dir_all(&yklua_tgt_dir).unwrap();

        // We use `rsync` to copy from the submodule to the target dir. This means that we can
        // transparently build multiple copies of yklua without the user having to know that there is
        // more than one copy.
        let mut rsync_cmd = Command::new("rsync");
        rsync_cmd.args(["-a", YKLUA_SUBMODULE_PATH, yklua_tgt_dir.to_str().unwrap()]);
        rsync_cmd.status().unwrap().exit_ok().unwrap();

        // cargo can sometimes run multiple builds in parallel. At the moment that's probably only
        // build scripts, though that doesn't seem to be precisely specified and might change in the
        // future. To avoid tripping us up, we use the same trick as `ykbuild/build.rs` and use a
        // `build_lock` so that we don't have two builds trampling on each other.
        let mut lock_path = yklua_tgt_dir.clone();
        lock_path.push("build_lock");
        let lock_file = File::create(lock_path).unwrap();
        lock_file.lock_exclusive().unwrap();

        let mut build_dir = yklua_tgt_dir.clone();
        build_dir.push("yklua");
        build_dir.push("src");

        // If ykllvm changes, we need to rebuild yklua from scratch. There isn't a perfect way to do
        // this, because we can't easily distinguish dirty changes so:
        //   1. If the commit hash changes, we `make clean`.
        //   2. If the commit hash is suffixed with `-dirty`, we `make clean`.
        // The second clause means that if you have local changes in the yklua submodule, you'll pay a
        // rebuild penalty every time you run `cargo`.
        let mut ykllvm_git_hash_path = yklua_tgt_dir.clone();
        ykllvm_git_hash_path.push("ykllvm_hash");
        let ykllvm_hash = String::from_utf8(
            Command::new("git")
                .current_dir(YKLLLVM_SUBMODULE_PATH)
                .args(["describe", "--always", "--dirty", "--no-abbrev"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap();
        let do_make_clean = if ykllvm_hash.trim().ends_with("-dirty") {
            true
        } else if let Ok(x) = read_to_string(&ykllvm_git_hash_path) {
            x.trim() != ykllvm_hash.trim()
        } else {
            true
        };
        write(ykllvm_git_hash_path, ykllvm_hash).unwrap();
        if do_make_clean {
            let mut cmd = Command::new("make");
            cmd.arg("clean")
                .current_dir(build_dir.as_os_str().to_str().unwrap());
            supuner(cmd);
        }

        // Because `make` is so quick, we can afford to run it every time this file is run -- if
        // there's nothing to do, it'll finish quicker than we can notice!
        let mut make_cmd = Command::new("make");
        let profile = full_cargo_profile();
        make_cmd
            .args([
                "-j".into(),
                num_cpus::get().to_string(),
                "MYCFLAGS=-DYKLUA_DEBUG_STRS=2".into(),
            ])
            .env(
                "PATH",
                format!(
                    "{}:{}:{}",
                    canonicalize("../bin/").unwrap().to_str().unwrap(),
                    ykllvm_bin_dir().to_str().unwrap(),
                    env::var("PATH").unwrap()
                ),
            )
            .env("YK_BUILD_TYPE", profile)
            .current_dir(build_dir.as_os_str().to_str().unwrap());
        supuner(make_cmd);

        FileExt::unlock(&lock_file).unwrap();
    }

    /// Run `cmd` without outputting anything to the terminal unless it fails. Failure will also cause
    /// this process to terminate.
    fn supuner(mut cmd: Command) {
        let output = cmd.output().unwrap();
        if !output.status.success() {
            eprint!(
                "{cmd:?} failed with error code {}.\n\n",
                output
                    .status
                    .code()
                    .map_or_else(|| "<unknown>".to_string(), |x| x.to_string())
            );
            eprint!(
                "--- begin stdout ---\n{}\n--- end stdout ---\n\n",
                String::from_utf8_lossy(&output.stdout)
            );
            eprint!(
                "--- begin stderr ---\n{}\n--- end stderr ---\n\n",
                String::from_utf8_lossy(&output.stderr)
            );
            exit(1);
        }
    }

    pub(super) fn main() {
        build();

        println!("Running Lua tests...");

        // Set variables for tests `ignore-if`s.
        #[cfg(cargo_profile = "debug")]
        unsafe {
            env::set_var("YK_CARGO_PROFILE", "debug")
        };
        #[cfg(cargo_profile = "release")]
        unsafe {
            env::set_var("YK_CARGO_PROFILE", "release")
        };

        #[cfg(target_arch = "x86_64")]
        unsafe {
            env::set_var("YK_ARCH", "x86_64")
        };
        #[cfg(not(target_arch = "x86_64"))]
        panic!("Unknown target_arch");

        LangTester::new()
            .comment_prefix(COMMENT_PREFIX)
            .test_dir("lua")
            .test_path_filter(|p: &Path| {
                p.extension().as_ref().and_then(|p| p.to_str()) == Some("lua")
            })
            .test_extract(move |p| {
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
                let mut yklua_exe = target_dir();
                yklua_exe.push("yklua");
                yklua_exe.push("yklua");
                yklua_exe.push("src");
                yklua_exe.push("lua");
                let mut cmd = Command::new(yklua_exe);
                cmd.arg(p);
                vec![("Run-time", cmd)]
            })
            .fm_options(|_, _, fmb| {
                // Use `${{...}}` to match non-literal strings in tests.
                // E.g. use `${{var}}` to capture the name of a variable.
                let ptn_re = Regex::new(r"\$\{\{.+?\}\}").unwrap();
                let ptn_re_ignore = Regex::new(r"\$\{\{_}\}").unwrap();
                let text_re = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
                fmb.name_matcher_ignore(ptn_re_ignore, text_re.clone())
                    .name_matcher(ptn_re, text_re)
            })
            .run();
    }
}
