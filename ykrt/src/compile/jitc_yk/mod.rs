//! Yk's built-in trace compiler.

use super::CompilationError;
use crate::{
    compile::{CompiledTrace, Compiler},
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::AOTTraceIterator,
};
use parking_lot::Mutex;
use std::{
    collections::HashSet,
    env,
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
}

impl IRPhase {
    fn from_str(s: &str) -> Result<Self, CompilationError> {
        match s {
            "aot" => Ok(Self::AOT),
            "jit-pre-opt" => Ok(Self::PreOpt),
            "jit-post-opt" => Ok(Self::PostOpt),
            _ => Err(CompilationError::Unrecoverable(format!(
                "Invalid YKD_PRINT_IR value: {}",
                s
            ))),
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
        let ir_slice = yk_ir_section()?;
        // FIXME: Cache deserialisation, so we don't load it afresh each time.
        let aot_mod = aot_ir::deserialise_module(ir_slice)?;

        if PHASES_TO_PRINT.contains(&IRPhase::AOT) {
            eprintln!("--- Begin aot ---");
            aot_mod.dump();
            eprintln!("--- End aot ---");
        }

        let jit_mod = trace_builder::build(&aot_mod, aottrace_iter.0)?;

        if PHASES_TO_PRINT.contains(&IRPhase::PreOpt) {
            eprintln!("--- Begin pre-opt ---");
            jit_mod.dump();
            eprintln!("--- End pre-opt ---");
        }

        todo!("new codegen doesn't work yet");
    }
}

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, CompilationError> {
        Ok(Arc::new(Self {}))
    }
}

pub(crate) fn yk_ir_section() -> Result<&'static [u8], CompilationError> {
    let start = symbol_vaddr(&CString::new("ykllvm.yk_ir.start").unwrap()).ok_or(
        CompilationError::Unrecoverable("couldn't find ykllvm.yk_ir.start".into()),
    )?;
    let stop = symbol_vaddr(&CString::new("ykllvm.yk_ir.stop").unwrap()).ok_or(
        CompilationError::Unrecoverable("couldn't find ykllvm.yk_ir.stop".into()),
    )?;
    Ok(unsafe { slice::from_raw_parts(start as *const u8, stop - start) })
}
