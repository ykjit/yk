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
use std::{collections::HashSet, env, error::Error, slice, sync::Arc};
use ykaddr::addr::symbol_to_ptr;

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
            _ => Err(format!("Invalid YKD_LOG_IR value: {s}").into()),
        }
    }
}

pub(crate) struct JITCYk {
    phases_to_print: HashSet<IRPhase>,
}

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        let mut phases_to_print = HashSet::new();
        if let Ok(stages) = env::var("YKD_LOG_IR") {
            for x in stages.split(',') {
                phases_to_print.insert(IRPhase::from_str(x)?);
            }
        };
        Ok(Arc::new(Self { phases_to_print }))
    }

    fn default_codegen<'a>(
        jit_mod: &'a jit_ir::Module,
    ) -> Result<Box<dyn CodeGen<'a> + 'a>, CompilationError> {
        #[cfg(target_arch = "x86_64")]
        {
            let ra = Box::new(SpillAllocator::new(StackDirection::GrowsDown));
            Ok(codegen::x86_64::X64CodeGen::new(jit_mod, ra)?)
        }
        #[cfg(not(target_arch = "x86_64"))]
        panic!("No code generator available for this platform");
    }
}

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

        if self.phases_to_print.contains(&IRPhase::AOT) {
            eprintln!("--- Begin aot ---");
            aot_mod.dump();
            eprintln!("--- End aot ---");
        }

        let jit_mod = trace_builder::build(&aot_mod, aottrace_iter.0)?;

        if self.phases_to_print.contains(&IRPhase::PreOpt) {
            eprintln!("--- Begin jit-pre-opt ---");
            // FIXME: If the `unwrap` fails, something rather bad has happened: does recovery even
            // make sense?
            jit_mod.dump().unwrap();
            eprintln!("--- End jit-pre-opt ---");
        }

        let cg = Box::new(Self::default_codegen(&jit_mod)?);
        let ct = cg.codegen()?;

        #[cfg(any(debug_assertions, test))]
        if self.phases_to_print.contains(&IRPhase::Asm) {
            eprintln!("--- Begin jit-asm ---");
            // If this unwrap fails, something went wrong in codegen.
            eprintln!("{}", ct.disassemble().unwrap());
            eprintln!("--- End jit-asm ---");
        }

        Ok(ct)
    }
}

pub(crate) fn yk_ir_section() -> Result<&'static [u8], Box<dyn Error>> {
    let start = symbol_to_ptr("ykllvm.yk_ir.start")? as *const u8;
    let stop = symbol_to_ptr("ykllvm.yk_ir.stop")? as *const u8;
    debug_assert!(start < stop);
    Ok(unsafe { slice::from_raw_parts(start as *const u8, stop.sub_ptr(start)) })
}
