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
    log::{log_ir, should_log_ir, IRPhase},
    mt::{SideTraceInfo, MT},
    trace::AOTTraceIterator,
};

use parking_lot::Mutex;
use std::{error::Error, slice, sync::Arc};
use ykaddr::addr::symbol_to_ptr;

pub mod aot_ir;
mod codegen;
#[cfg(any(debug_assertions, test))]
mod gdb;
pub mod jit_ir;
mod opt;
mod trace_builder;

pub(crate) struct JITCYk;

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self))
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
        mt: Arc<MT>,
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

        if should_log_ir(IRPhase::AOT) {
            log_ir(&format!("--- Begin aot ---\n{}\n--- End aot ---", aot_mod));
        }

        let jit_mod = trace_builder::build(mt.next_compiled_trace_id(), &aot_mod, aottrace_iter.0)?;

        if should_log_ir(IRPhase::PreOpt) {
            log_ir(&format!(
                "--- Begin jit-pre-opt ---\n{jit_mod}\n--- End jit-pre-opt ---",
            ));
        }

        // FIXME: This should be enabled when optimisations can cope with all the inputs we see.
        // let jit_mod = opt::opt(jit_mod)?;
        // if should_log_ir(IRPhase::PostOpt) {
        //     log_ir(&format!(
        //         "--- Begin jit-post-opt ---\n{jit_mod}\n--- End jit-pre-opt ---",
        //     ));
        // }

        let cg = Box::new(Self::default_codegen(&jit_mod)?);
        let ct = cg.codegen()?;

        #[cfg(any(debug_assertions, test))]
        if should_log_ir(IRPhase::Asm) {
            log_ir(&format!(
                "--- Begin jit-asm ---\n{}\n--- End jit-asm ---",
                ct.disassemble().unwrap()
            ));
        }

        Ok(ct)
    }
}

pub(crate) fn yk_ir_section() -> Result<&'static [u8], Box<dyn Error>> {
    let start = symbol_to_ptr("ykllvm.yk_ir.start")? as *const u8;
    let stop = symbol_to_ptr("ykllvm.yk_ir.stop")? as *const u8;
    debug_assert!(start < stop);
    Ok(unsafe { slice::from_raw_parts(start, stop.sub_ptr(start)) })
}
