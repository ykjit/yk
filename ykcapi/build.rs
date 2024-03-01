use std::env;

pub fn main() {
    ykbuild::apply_llvm_ld_library_path();

    // FIXME: This is a temporary hack because LLVM has problems if the main thread exits before
    // compilation threads have finished.
    println!("cargo:rustc-cfg=yk_llvm_sync_hack");
    println!("cargo:rerun-if-env-changed=YKB_TRACER");
    match env::var("YKB_TRACER") {
        Ok(ref tracer) if tracer == "swt" => println!("cargo:rustc-cfg=tracer_swt"),
        Ok(ref tracer) if tracer == "hwt" => println!("cargo:rustc-cfg=tracer_hwt"),
        Err(env::VarError::NotPresent) => println!("cargo:rustc-cfg=tracer_hwt"),
        Ok(x) => panic!("Unknown tracer {x}"),
        Err(_) => panic!("Invalid value for YKB_TRACER"),
    }
}
