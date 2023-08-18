pub fn main() {
    ykbuild::apply_llvm_ld_library_path();

    println!("cargo:rustc-cfg=jitc_llvm");
}
