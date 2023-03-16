use glob::glob;
use rerun_except::rerun_except;
use std::{
    env,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::TempDir;

/// Collect the compilation commands found (one per-file) in `tmpdir` and generate a
/// `compile_commands.json` file for clangd to use.
fn write_clangd_json(tmpdir: &Path) {
    let mut entries = Vec::new();
    let mdir = env::var("CARGO_MANIFEST_DIR").unwrap();

    for path in glob(&format!("{}/*", tmpdir.to_str().unwrap())).unwrap() {
        let mut infile = File::open(path.unwrap()).unwrap();
        let mut buf = String::new();
        infile.read_to_string(&mut buf).unwrap();
        let buf = buf.trim();

        // The `cc` crate always puts the source file on the end.
        let ccfile = buf.split(" ").last().unwrap();
        assert!(ccfile.starts_with("src/") && ccfile.ends_with(".cc"));

        let mut entry = String::new();
        entry.push_str("  {\n");
        entry.push_str(&format!("    \"directory\": \"{mdir}\",\n"));
        entry.push_str(&format!("    \"command\": \"{buf}\",\n"));
        entry.push_str(&format!("    \"file\": \"{ccfile}\",\n"));
        entry.push_str("  }");
        entries.push(entry);
    }

    // Write JSON to Rust target dir.
    let outpath = [&mdir, "..", "target", "compile_commands.json"]
        .iter()
        .collect::<PathBuf>();
    let mut outfile = File::create(outpath).unwrap();
    write!(outfile, "[\n").unwrap();
    write!(outfile, "{}", entries.join(",\n")).unwrap();
    write!(outfile, "\n]\n").unwrap();
}

fn main() {
    // Ensure changing C++ source files or headers retriggers a build.
    rerun_except(&[]).unwrap();

    // Compile our wrappers with the right LLVM C++ flags.
    let cxxflags_out = Command::new("llvm-config")
        .arg("--cxxflags")
        .output()
        .unwrap()
        .stdout;
    let cxxflags_str = std::str::from_utf8(&cxxflags_out).unwrap();
    let cxxflags = cxxflags_str.split_whitespace().collect::<Vec<_>>();

    let mut comp = cc::Build::new();
    for cf in cxxflags {
        comp.flag(cf);
    }
    comp.flag("-Wall");
    comp.flag("-Werror");
    comp.flag("-std=c++17");
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

    // We use a compiler wrapper to capture compilation commands and generate a
    // `compile_commands.json` database for clangd.
    let td = TempDir::new().unwrap();
    env::set_var("YKLLVMWRAP_JSON_TEMP", td.path());
    comp.compiler("./wrap-clang++.sh");

    // Actually do the compilation.
    comp.compile("ykllvmwrap");

    // Generate `compile_commands.json`.
    write_clangd_json(td.path());

    // Ensure that downstream crates performing linkage use the right -L and -l flags.
    let lib_dir = Command::new("llvm-config")
        .arg("--libdir")
        .output()
        .unwrap()
        .stdout;
    let lib_dir = std::str::from_utf8(&lib_dir).unwrap();
    println!("cargo:rustc-link-search={}", lib_dir);

    let libs = Command::new("llvm-config")
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
