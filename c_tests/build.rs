use std::env;

pub fn main() {
    // Expose the cargo profile to run.rs so that it can set the right link flags.
    if let Ok(profile) = env::var("PROFILE") {
        println!("cargo:rustc-cfg=cargo_profile=\"{}\"", profile);
    }
}
