pub fn main() {
    ykbuild::apply_llvm_ld_library_path();

    // Always compile in the LLVM JIT compiler.
    println!("cargo:rustc-cfg=jitc_llvm");
    // Always compile in our bespoke JIT compiler.
    println!("cargo:rustc-cfg=jitc_yk");
    // Always compile in the HWT tracer.
    println!("cargo:rustc-cfg=tracer_hwt");
    // FIXME: This is a temporary hack because LLVM has problems if the main thread exits before
    // compilation threads have finished.
    println!("cargo:rustc-cfg=yk_llvm_sync_hack");
}
