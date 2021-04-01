use std::env;

fn main() {
    cc::Build::new()
        .file("src/test_helpers.c")
        .compile("ykcompile_test_helpers");

    let profile = env::var("PROFILE").unwrap();
    println!(
        "cargo:rustc-link-search={}/../../untraced/target/{}/",
        env::current_dir().unwrap().to_str().unwrap(),
        profile
    );
    println!("cargo:rustc-link-lib=dylib=to_traced");
}
