use std::env;
use ykbuild;

pub fn main() {
    // Expose the cargo profile to run.rs so that it can set the right link flags.
    if let Ok(profile) = env::var("PROFILE") {
        println!("cargo:rustc-cfg=cargo_profile=\"{}\"", profile);
    }

    ykbuild::apply_llvm_ld_library_path();
}
