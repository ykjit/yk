use std::env;

pub fn main() {
    // Declare the custom cfg flags to avoid unexpected_cfgs warnings
    println!("cargo::rustc-check-cfg=cfg(swt_modclone)");

    println!("cargo::rerun-if-env-changed=YKB_TRACER");
    println!("cargo::rerun-if-env-changed=YKB_SWT_MODCLONE");
    match env::var("YKB_TRACER") {
        Ok(ref tracer) if tracer == "swt" => println!("cargo::rustc-cfg=tracer_swt"),
        Ok(ref tracer) if tracer == "hwt" => println!("cargo::rustc-cfg=tracer_hwt"),
        Err(env::VarError::NotPresent) => println!("cargo::rustc-cfg=tracer_swt"),
        Ok(x) => panic!("Unknown tracer {x}"),
        Err(_) => panic!("Invalid value for YKB_TRACER"),
    }

    match env::var("YKB_SWT_MODCLONE") {
        Ok(ref modclone) if modclone == "1" => println!("cargo::rustc-cfg=swt_modclone"),
        _ => {}
    }
}
