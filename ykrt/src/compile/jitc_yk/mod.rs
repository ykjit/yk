//! Yk's built-in trace compiler.

use crate::{
    compile::{CompiledTrace, Compiler},
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::TracedAOTBlock,
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

#[derive(Eq, Hash, PartialEq)]
enum IRPhase {
    AOT,
    PreOpt,
    PostOpt,
}

impl IRPhase {
    fn from_str(s: &str) -> Result<Self, String> {
        let ret = match s {
            "aot" => Self::AOT,
            "jit-pre-opt" => Self::PreOpt,
            "jit-post-opt" => Self::PostOpt,
            _ => return Err(format!("Invalid YKD_PRINT_IR value: {}", s)),
        };
        Ok(ret)
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

pub mod aot_ir;
pub mod jit_ir;
mod trace_builder;

pub(crate) struct JITCYk;

impl Compiler for JITCYk {
    fn compile(
        &self,
        _mt: Arc<MT>,
        mtrace: Vec<TracedAOTBlock>,
        sti: Option<SideTraceInfo>,
        _hl: Arc<Mutex<HotLocation>>,
    ) -> Result<CompiledTrace, Box<dyn Error>> {
        if sti.is_some() {
            todo!();
        }
        let ir_slice = yk_ir_section();
        // FIXME: Cache deserialisation, so we don't load it afresh each time.
        let aot_mod = aot_ir::deserialise_module(ir_slice)?;

        if PHASES_TO_PRINT.contains(&IRPhase::AOT) {
            eprintln!("--- Begin aot ---");
            aot_mod.dump();
            eprintln!("--- End aot ---");
        }

        let jit_mod = trace_builder::build(&aot_mod, &mtrace)?;

        if PHASES_TO_PRINT.contains(&IRPhase::PreOpt) {
            eprintln!("--- Begin pre-opt ---");
            jit_mod.dump();
            eprintln!("--- End pre-opt ---");
        }

        todo!("new codegen doesn't work yet");
    }
}

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self {}))
    }
}

pub(crate) fn yk_ir_section() -> &'static [u8] {
    let start = symbol_vaddr(&CString::new("ykllvm.yk_ir.start").unwrap()).unwrap();
    let stop = symbol_vaddr(&CString::new("ykllvm.yk_ir.stop").unwrap()).unwrap();
    unsafe { slice::from_raw_parts(start as *const u8, stop - start) }
}
