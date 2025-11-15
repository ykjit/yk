use cfgrammar::yacc::YaccKind;
use lrlex::{CTLexerBuilder, DefaultLexerTypes};
use which::which;

use {
    std::env,
    std::io::{self, Write},
    std::process::{Command, exit},
};

pub fn main() {
    println!("cargo::rerun-if-env-changed=YKB_TRACER");

    println!("cargo::rustc-check-cfg=cfg(tracer_hwt)");
    println!("cargo::rustc-check-cfg=cfg(tracer_swt)");
    match env::var("YKB_TRACER") {
        Ok(ref tracer) if tracer == "swt" => println!("cargo::rustc-cfg=tracer_swt"),
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        Ok(ref tracer) if tracer == "hwt" => println!("cargo::rustc-cfg=tracer_hwt"),
        Err(env::VarError::NotPresent) => println!("cargo::rustc-cfg=tracer_swt"),
        Ok(x) => panic!("Unknown tracer {x}"),
        Err(_) => panic!("Invalid value for YKB_TRACER"),
    }

    // We need to explicitly tell Cargo to track these files otherwise it won't rebuild when they
    // change.
    println!("cargo::rerun-if-changed=src/compile/jitc_yk/jit_ir/jit_ir.y");
    println!("cargo::rerun-if-changed=src/compile/jitc_yk/jit_ir/jit_ir.l");
    CTLexerBuilder::<DefaultLexerTypes<u8>>::new_with_lexemet()
        .lrpar_config(|ctp| {
            ctp.yacckind(YaccKind::Grmtools)
                .grammar_in_src_dir("compile/jitc_yk/jit_ir/jit_ir.y")
                .unwrap()
        })
        .lexer_in_src_dir("compile/jitc_yk/jit_ir/jit_ir.l")
        .unwrap()
        .build()
        .unwrap();

    // We need to explicitly tell Cargo to track these files otherwise it won't rebuild when they
    // change.
    println!("cargo::rerun-if-changed=src/compile/j2/hir.l");
    println!("cargo::rerun-if-changed=src/compile/j2/hir.y");
    CTLexerBuilder::<DefaultLexerTypes<u16>>::new_with_lexemet()
        .lrpar_config(|ctp| {
            ctp.yacckind(YaccKind::Grmtools)
                .grammar_in_src_dir("compile/j2/hir.y")
                .unwrap()
        })
        .lexer_in_src_dir("compile/j2/hir.l")
        .unwrap()
        .build()
        .unwrap();

    // Build the gdb plugin.
    env::set_current_dir("yk_gdb_plugin").unwrap();
    let make = {
        if which("gmake").is_ok() {
            "gmake"
        } else {
            "make"
        }
    };
    let out = Command::new(make).output().unwrap();
    if !out.status.success() {
        io::stderr().write_all(&out.stdout).unwrap();
        io::stderr().write_all(&out.stderr).unwrap();
        exit(1);
    }
    env::set_current_dir("..").unwrap();
    println!("cargo::rerun-if-changed=yk_gdb_plugin/yk_gdb_plugin.c");
}
