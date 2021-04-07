use std::env;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::process::Command;

include!("../../build_aux.rs");

const UNTRACED_SO: &str = "libuntraced_api.so";

fn main() {
    let mut untraced_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    untraced_dir.push("..");
    untraced_dir.push("..");
    untraced_dir.push("untraced");
    let cargo = env::var("CARGO").unwrap();
    let profile = env::var("PROFILE").unwrap();

    // Only build the untraced workspace now if we are not using xtask, i.e. if we are being
    // consumed as a dependency. This means we see the compiler output for the untraced workspace
    // correctly when working directly with the yk repo (via xtask).
    if env::var("YK_XTASK").is_err() {
        let mut rustflags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());
        let tracing_kind = find_tracing_kind(&rustflags);
        rustflags = make_untraced_rustflags(&rustflags);
        let mut cmd = Command::new(cargo);
        cmd.current_dir(untraced_dir.join("untraced_api"))
            .arg("build")
            .arg("--features")
            .arg(format!("yktrace/trace_{}", tracing_kind))
            .env("RUSTFLAGS", &rustflags);

        if profile == "release" {
            cmd.arg("--release");
        }

        let status = cmd.spawn().unwrap().wait().unwrap();
        if !status.success() {
            eprintln!("building untraced workspace failed");
            std::process::exit(1);
        }
    }

    // If we symlink libuntraced_api.so into the target dir, then this is already in the linker path when
    // run under cargo. In other words, the user won't have to set LD_LIBRARY_PATH.
    let mut sym_dst = PathBuf::from(env::var("OUT_DIR").unwrap());
    sym_dst.push("..");
    sym_dst.push("..");
    sym_dst.push("..");
    sym_dst.push(UNTRACED_SO);
    if !PathBuf::from(&sym_dst).exists() {
        let mut sym_src = untraced_dir;
        sym_src.push("target");
        sym_src.push(&profile);
        sym_src.push(UNTRACED_SO);
        symlink(sym_src, sym_dst).unwrap();
    }

    println!(
        "cargo:rustc-link-search={}/../../untraced/target/{}/",
        env::current_dir().unwrap().to_str().unwrap(),
        profile
    );
    println!("cargo:rustc-link-lib=dylib=untraced_api");
}
