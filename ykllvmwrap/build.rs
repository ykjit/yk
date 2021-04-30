use std::process::Command;

fn main() {
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
    comp.file("src/ykllvmwrap.cc")
        .compiler("clang++")
        // Lots of unused parameters in the LLVM headers.
        .flag("-Wno-unused-parameter")
        .cpp(true);
    comp.compile("ykllvmwrap");

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
