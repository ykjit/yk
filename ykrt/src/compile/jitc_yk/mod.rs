//! Yk's built-in trace compiler.

use super::CompilationError;
use crate::{
    compile::{
        jitc_yk::codegen::{
            reg_alloc::{spill_alloc::SpillAllocator, RegisterAllocator, StackDirection},
            CodeGen,
        },
        CompiledTrace, Compiler,
    },
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::AOTTraceIterator,
};

use parking_lot::Mutex;
use std::{
    collections::HashSet,
    env,
    error::Error,
    ffi::CString,
    slice,
    sync::{Arc, LazyLock},
};
use ykaddr::addr::symbol_vaddr;

pub mod aot_ir;
mod codegen;
pub mod jit_ir;
mod trace_builder;

#[derive(Eq, Hash, PartialEq)]
enum IRPhase {
    AOT,
    PreOpt,
    PostOpt,
    Asm,
}

impl IRPhase {
    fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
        match s {
            "aot" => Ok(Self::AOT),
            "jit-pre-opt" => Ok(Self::PreOpt),
            "jit-post-opt" => Ok(Self::PostOpt),
            "jit-asm" => Ok(Self::Asm),
            _ => Err(format!("Invalid YKD_PRINT_IR value: {s}").into()),
        }
    }
}

static PHASES_TO_PRINT: LazyLock<HashSet<IRPhase>> = LazyLock::new(|| {
    if let Ok(stages) = env::var("YKD_PRINT_IR") {
        stages
            .split(',')
            .map(IRPhase::from_str)
            .map(|res| res.unwrap())
            .collect::<HashSet<IRPhase>>()
    } else {
        HashSet::new()
    }
});

pub(crate) struct JITCYk;

impl Compiler for JITCYk {
    fn compile(
        &self,
        _mt: Arc<MT>,
        aottrace_iter: (Box<dyn AOTTraceIterator>, Box<[usize]>),
        sti: Option<SideTraceInfo>,
        _hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        if sti.is_some() {
            todo!();
        }
        // If either `unwrap` fails, there is no chance of the system working correctly.
        let ir_slice = yk_ir_section().unwrap();
        let aot_mod = aot_ir::deserialise_module(ir_slice).unwrap();

        if PHASES_TO_PRINT.contains(&IRPhase::AOT) {
            eprintln!("--- Begin aot ---");
            aot_mod.dump();
            eprintln!("--- End aot ---");
        }

        let jit_mod = trace_builder::build(&aot_mod, aottrace_iter.0)?;

        if PHASES_TO_PRINT.contains(&IRPhase::PreOpt) {
            eprintln!("--- Begin pre-opt ---");
            // FIXME: If the `unwrap` fails, something rather bad has happened: does recovery even
            // make sense?
            jit_mod.dump().unwrap();
            eprintln!("--- End pre-opt ---");
        }

        let mut ra = SpillAllocator::new(StackDirection::GrowsDown);
        let cg = codegen::x86_64::X64CodeGen::new(&jit_mod, &mut ra).unwrap();
        let ct = cg.codegen()?;

        #[cfg(any(debug_assertions, test))]
        if PHASES_TO_PRINT.contains(&IRPhase::Asm) {
            eprintln!("--- Begin jit-asm ---");
            // If this unwrap fails, something went wrong in codegen.
            eprintln!("{}", ct.disassemble().unwrap());
            eprintln!("--- End jit-asm ---");
        }

        Ok(ct)
    }
}

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self {}))
    }
}

pub(crate) fn yk_ir_section() -> Result<&'static [u8], Box<dyn Error>> {
    let start = symbol_vaddr(&CString::new("ykllvm.yk_ir.start").unwrap())
        .ok_or("couldn't find ykllvm.yk_ir.start")?;
    let stop = symbol_vaddr(&CString::new("ykllvm.yk_ir.stop").unwrap())
        .ok_or("couldn't find ykllvm.yk_ir.stop")?;
    Ok(unsafe { slice::from_raw_parts(start as *const u8, stop - start) })
}
