use cc;
use std::env;

fn main() {
    cc::Build::new()
        .file("src/test_helpers.c")
        .compile("ykcompile_test_helpers");

    println!(
        "cargo:rustc-link-search={}/../internal_ws/target/release/",
        env::current_dir().unwrap().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=dylib=ykshim");
}
