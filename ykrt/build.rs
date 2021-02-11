use regex::Regex;
use std::env;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::process::Command;

include!("../build_aux.rs");

const YKSHIM_SO: &str = "libykshim.so";

fn main() {
    let ykrt_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut internal_dir = ykrt_dir.clone();
    internal_dir.push("..");
    internal_dir.push("internal_ws");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cargo = env::var("CARGO").unwrap();

    // Only build the internal workspace now if we are not using xtask, i.e. if we are being
    // consumed as a dependency. This means we see the compiler output for the internal workspace
    // correctly when working directly with the yk repo (via xtask).
    if env::var("YK_XTASK").is_err() {
        let mut rustflags = env::var("RUSTFLAGS").unwrap_or_else(|_| String::new());
        rustflags = make_internal_rustflags(&rustflags);
        let status = Command::new(cargo)
            .current_dir(&internal_dir)
            .arg("build")
            .arg("--release")
            .env("RUSTFLAGS", &rustflags)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
        if !status.success() {
            eprintln!("building internal workspace failed");
            std::process::exit(1);
        }
    }

    // If we symlink libykshim.so into the target dir, then this is already in the linker path when
    // run under cargo. In other words, the user won't have to set LD_LIBRARY_PATH.
    let mut sym_dest = out_dir.clone();
    sym_dest.push("..");
    sym_dest.push("..");
    sym_dest.push("..");
    sym_dest.push(YKSHIM_SO);
    if !PathBuf::from(&sym_dest).exists() {
        let mut sym_src = internal_dir.clone();
        sym_src.push("target");
        sym_src.push("release");
        sym_src.push(YKSHIM_SO);
        dbg!(&sym_src, &sym_dest);
        symlink(sym_src, sym_dest).unwrap();
    }

    println!(
        "cargo:rustc-link-search={}/../internal_ws/target/release/",
        env::current_dir().unwrap().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=dylib=ykshim");
}
