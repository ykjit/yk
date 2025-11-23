//! Yk's built-in trace compiler.

use super::CompilationError;
use crate::{
    compile::{CompiledTrace, Compiler, GuardId, jitc_yk::codegen::CodeGen},
    location::HotLocation,
    log::{IRPhase, log_ir, should_log_ir},
    mt::{MT, TraceId},
    trace::AOTTraceIterator,
};
use jit_ir::TraceEndFrame;
use parking_lot::Mutex;
use std::{
    error::Error,
    fmt,
    marker::PhantomData,
    slice,
    sync::{Arc, LazyLock},
};
use ykaddr::addr::symbol_to_ptr;
use yksmp::Location;

pub mod aot_ir;
pub(super) mod arbbitint;
mod codegen;
#[cfg(any(debug_assertions, test))]
mod gdb;
mod int_signs;
pub mod jit_ir;
mod opt;
mod trace_builder;

pub(crate) static AOT_MOD: LazyLock<aot_ir::Module> = LazyLock::new(|| {
    let ir_slice = yk_ir_section().unwrap();
    aot_ir::deserialise_module(ir_slice).unwrap()
});

/// Contains information required for side-tracing.
pub(crate) struct YkSideTraceInfo<Register: Send + Sync> {
    /// The AOT IR block the failing guard originated from.
    bid: aot_ir::BBlockId,
    /// Inlined calls tracked by [trace_builder] during processing of a trace. Required for
    /// side-tracing in order setup a new [trace_builder] and process a side-trace.
    callframes: Vec<jit_ir::InlinedFrame>,
    /// Mapping of AOT variables to their current location. Used to pass variables from a parent
    /// trace into a side-trace.
    lives: Vec<(aot_ir::InstId, Location)>,
    /// The live variables at the entry point of the root trace.
    entry_vars: Vec<codegen::reg_alloc::VarLocation<Register>>,
    /// Stack pointer offset from the base pointer of the interpreter frame including the
    /// interpreter frame itself and all parent traces. Since all traces execute in the interpreter
    /// frame, each trace adds to this value, making extra space on the stack. This then forms the
    /// new `sp_offset` for any side-traces of this trace.
    sp_offset: usize,
    /// The trace to jump to at the end of this sidetrace.
    target_ctr: Arc<dyn CompiledTrace>,
}

impl<Register: Send + Sync> YkSideTraceInfo<Register> {
    /// Return the live call frames which are required to setup the trace builder during
    /// side-tracing.
    fn callframes(&self) -> &[jit_ir::InlinedFrame] {
        &self.callframes
    }

    /// Return the live AOT variables for this guard. Used to write live values to during deopt.
    fn lives(&self) -> &[(aot_ir::InstId, Location)] {
        &self.lives
    }
}

impl<Register: Send + Sync> fmt::Debug for YkSideTraceInfo<Register> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "YkSideTraceInfo {{ ... }}")
    }
}

pub(crate) struct JITCYk<Register> {
    codegen: Arc<dyn CodeGen>,
    phantom: PhantomData<Register>,
}

impl JITCYk<codegen::x64::Register> {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self {
            codegen: codegen::default_codegen()?,
            phantom: PhantomData,
        }))
    }
}

impl<Register: Send + Sync + 'static> JITCYk<Register> {
    // FIXME: This should probably be split into separate root / sidetrace functions.
    fn compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        ctrid: TraceId,
        sti: Option<Arc<YkSideTraceInfo<codegen::x64::Register>>>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        connector_ctr: Option<Arc<dyn CompiledTrace>>,
        endframe: TraceEndFrame,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        // If either `unwrap` fails, there is no chance of the system working correctly.
        let aot_mod = &*AOT_MOD;

        if should_log_ir(IRPhase::AOT) {
            log_ir(&format!("--- Begin aot ---\n{aot_mod}\n--- End aot ---\n"));
        }

        let mut jit_mod = trace_builder::build(
            &mt,
            aot_mod,
            ctrid,
            aottrace_iter,
            sti,
            promotions,
            debug_strs,
            connector_ctr,
            endframe,
        )?;

        let ds = if let Some(x) = &hl.lock().debug_str {
            format!(": {}", x.as_str())
        } else {
            "".to_owned()
        };

        if should_log_ir(IRPhase::DebugStrs) {
            let kind = match jit_mod.tracekind() {
                jit_ir::TraceKind::HeaderOnly => "header",
                jit_ir::TraceKind::HeaderAndBody => unreachable!(),
                jit_ir::TraceKind::Connector(_) => "connector",
                jit_ir::TraceKind::Sidetrace(_) => "side-trace",
                jit_ir::TraceKind::DifferentFrames => "call-recursion",
            };
            let mut out = String::new();
            out.push_str(&format!("--- Begin debugstrs: {kind}{ds} ---\n"));
            for (_, inst) in jit_mod.iter_skipping_insts() {
                if let jit_ir::Inst::DebugStr(x) = inst {
                    out.push_str(&format!("  {}\n", x.msg(&jit_mod)));
                }
            }
            out.push_str("--- End debugstrs ---\n");
            log_ir(&out);
        }

        if should_log_ir(IRPhase::PreOpt) {
            log_ir(&format!(
                "--- Begin jit-pre-opt{ds} ---\n{jit_mod}\n--- End jit-pre-opt ---\n",
            ));
        }

        if mt.opt_level() > 0 {
            jit_mod = opt::opt(jit_mod)?;
            if should_log_ir(IRPhase::PostOpt) {
                jit_mod.dead_code_elimination();
                log_ir(&format!(
                    "--- Begin jit-post-opt{ds} ---\n{jit_mod}\n--- End jit-post-opt ---\n",
                ));
            }
        }

        // FIXME: This needs to be the combined stacksize of all parent traces.
        let ct = self.codegen.codegen(jit_mod, mt.clone(), hl)?;

        if should_log_ir(IRPhase::Asm) {
            log_ir(&format!(
                "--- Begin jit-asm{ds} ---\n{}\n--- End jit-asm ---\n",
                ct.disassemble(false).unwrap()
            ));
        }
        if should_log_ir(IRPhase::AsmFull) {
            log_ir(&format!(
                "--- Begin jit-asm-full{ds} ---\n{}\n--- End jit-asm-full ---\n",
                ct.disassemble(true).unwrap()
            ));
        }

        // Register JITted code (if required).
        mt.trace_profiler().register_ctr(&ct).map_err(|e| {
            CompilationError::General(format!("failed to register jitted code with profiler: {e}"))
        })?;

        Ok(ct)
    }
}

impl<Register: Send + Sync + 'static> Compiler for JITCYk<Register> {
    fn root_compile(
        self: Arc<Self>,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        ctrid: TraceId,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        connector_ctr: Option<Arc<dyn CompiledTrace>>,
        endframe: TraceEndFrame,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        self.compile(
            mt,
            aottrace_iter,
            ctrid,
            None,
            hl,
            promotions,
            debug_strs,
            connector_ctr,
            endframe,
        )
    }

    fn sidetrace_compile(
        self: Arc<Self>,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        ctrid: TraceId,
        parent_ctr: Arc<dyn CompiledTrace>,
        gid: GuardId,
        target_ctr: Arc<dyn CompiledTrace>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        endframe: TraceEndFrame,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let parent_ctr = parent_ctr
            .as_any()
            .downcast::<codegen::x64::X64CompiledTrace>()
            .unwrap();
        let sti = parent_ctr.sidetraceinfo(gid, Arc::clone(&target_ctr));
        self.compile(
            mt,
            aottrace_iter,
            ctrid,
            Some(sti),
            hl,
            promotions,
            debug_strs,
            None,
            endframe,
        )
    }
}

pub(crate) fn yk_ir_section() -> Result<&'static [u8], Box<dyn Error>> {
    let start = symbol_to_ptr("ykllvm.yk_ir.start")? as *const u8;
    let stop = symbol_to_ptr("ykllvm.yk_ir.stop")? as *const u8;
    debug_assert!(start < stop);
    Ok(unsafe { slice::from_raw_parts(start, stop.sub(start as usize) as usize) })
}
