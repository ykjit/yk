use std::env;

pub fn main() {
    println!("cargo::rerun-if-env-changed=YKB_TRACER");
    match env::var("YKB_TRACER") {
        Ok(ref tracer) if tracer == "swt" => println!("cargo::rustc-cfg=tracer_swt"),
        Ok(ref tracer) if tracer == "hwt" => println!("cargo::rustc-cfg=tracer_hwt"),
        Err(env::VarError::NotPresent) => println!("cargo::rustc-cfg=tracer_swt"),
        Ok(x) => panic!("Unknown tracer {x}"),
        Err(_) => panic!("Invalid value for YKB_TRACER"),
    }
}
