//! Yk's built-in trace compiler.

use super::CompilationError;
use crate::{
    compile::{
        jitc_yk::codegen::CodeGen, jitc_yk::trace_builder::Frame, CompiledTrace, Compiler,
        SideTraceInfo,
    },
    location::HotLocation,
    log::{log_ir, should_log_ir, IRPhase},
    mt::MT,
    trace::AOTTraceIterator,
};
use parking_lot::Mutex;
use std::{
    env,
    error::Error,
    slice,
    sync::{Arc, LazyLock},
};
use ykaddr::addr::symbol_to_ptr;

pub mod aot_ir;
mod codegen;
#[cfg(any(debug_assertions, test))]
mod gdb;
pub mod jit_ir;
mod opt;
mod trace_builder;

/// Should we turn trace optimisations on or off? Currently defaults to "off".
static YKD_OPT: LazyLock<bool> = LazyLock::new(|| {
    let x = env::var("YKD_OPT");
    match x.as_ref().map(|x| x.as_str()) {
        Ok("1" | "2" | "3") => true,
        Ok(_) | Err(_) => false,
    }
});

pub(crate) static AOT_MOD: LazyLock<aot_ir::Module> = LazyLock::new(|| {
    let ir_slice = yk_ir_section().unwrap();
    aot_ir::deserialise_module(ir_slice).unwrap()
});

struct YkSideTraceInfo {
    callframes: Vec<Frame>,
    aotlives: Vec<aot_ir::InstID>,
}

impl SideTraceInfo for YkSideTraceInfo {
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
        self
    }
}

impl YkSideTraceInfo {
    /// Return the live call frames which are required to setup the trace builder during
    /// side-tracing.
    fn callframes(&self) -> &[Frame] {
        &self.callframes
    }

    /// Return the live AOT variables for this guard. Used to write live values to during deopt.
    fn aotlives(&self) -> &[aot_ir::InstID] {
        &self.aotlives
    }
}

pub(crate) struct JITCYk {
    codegen: Arc<dyn CodeGen>,
}

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self {
            codegen: codegen::default_codegen()?,
        }))
    }
}

impl Compiler for JITCYk {
    fn compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        sti: Option<Arc<dyn SideTraceInfo>>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[usize]>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        // If either `unwrap` fails, there is no chance of the system working correctly.
        let aot_mod = &*AOT_MOD;

        if should_log_ir(IRPhase::AOT) {
            log_ir(&format!(
                "--- Begin aot ---\n{}\n--- End aot ---\n",
                aot_mod
            ));
        }

        let sti = sti.map(|s| s.as_any().downcast::<YkSideTraceInfo>().unwrap());

        let mut jit_mod = trace_builder::build(
            mt.next_compiled_trace_id(),
            aot_mod,
            aottrace_iter,
            sti,
            promotions,
        )?;

        if should_log_ir(IRPhase::PreOpt) {
            log_ir(&format!(
                "--- Begin jit-pre-opt ---\n{jit_mod}\n--- End jit-pre-opt ---\n",
            ));
        }

        if *YKD_OPT {
            jit_mod = opt::opt(jit_mod)?;
            if should_log_ir(IRPhase::PostOpt) {
                log_ir(&format!(
                    "--- Begin jit-post-opt ---\n{jit_mod}\n--- End jit-post-opt ---\n",
                ));
            }
        }

        let ct = self.codegen.codegen(jit_mod, mt, hl)?;

        #[cfg(any(debug_assertions, test))]
        if should_log_ir(IRPhase::Asm) {
            log_ir(&format!(
                "--- Begin jit-asm ---\n{}\n--- End jit-asm ---\n",
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
