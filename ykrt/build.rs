use std::env;

fn main() {
    println!(
        "cargo:rustc-link-search={}/../internal_ws/target/release/",
        env::current_dir().unwrap().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=dylib=ykshim");
}
