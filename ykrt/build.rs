pub fn main() {
    ykbuild::apply_llvm_ld_library_path();

    // Always compile in the LLVM JIT compiler.
    println!("cargo:rustc-cfg=jitc_llvm");
    // Always compile in the HWT tracer.
    println!("cargo:rustc-cfg=tracer_hwt");
}
