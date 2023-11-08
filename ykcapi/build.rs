pub fn main() {
    ykbuild::apply_llvm_ld_library_path();

    // FIXME: This is a temporary hack because LLVM has problems if the main thread exits before
    // compilation threads have finished.
    println!("cargo:rustc-cfg=yk_llvm_sync_hack");
}
