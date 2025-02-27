//! Yk's built-in trace compiler.

use super::{CompilationError, GuardIdx};
use crate::{
    compile::{jitc_yk::codegen::CodeGen, CompiledTrace, Compiler, SideTraceInfo},
    location::HotLocation,
    log::{log_ir, should_log_ir, IRPhase},
    mt::MT,
    trace::AOTTraceIterator,
};
use parking_lot::Mutex;
use std::{
    env,
    error::Error,
    fmt,
    marker::PhantomData,
    slice,
    sync::{Arc, LazyLock},
};
use ykaddr::addr::symbol_to_ptr;
use yksmp::Location;

pub mod aot_ir;
mod arbbitint;
mod codegen;
#[cfg(any(debug_assertions, test))]
mod gdb;
mod int_signs;
pub mod jit_ir;
mod opt;
mod trace_builder;

/// Should we turn trace optimisations on or off? Defaults to "on".
static YKD_OPT: LazyLock<bool> = LazyLock::new(|| {
    let x = env::var("YKD_OPT");
    match x.as_ref().map(|x| x.as_str()) {
        Ok("0") => false,
        Ok(_) | Err(_) => true,
    }
});

pub(crate) static AOT_MOD: LazyLock<aot_ir::Module> = LazyLock::new(|| {
    let ir_slice = yk_ir_section().unwrap();
    aot_ir::deserialise_module(ir_slice).unwrap()
});

struct RootTracePtr(*const libc::c_void);
unsafe impl Send for RootTracePtr {}
unsafe impl Sync for RootTracePtr {}

/// Contains information required for side-tracing.
struct YkSideTraceInfo<Register: Send + Sync> {
    /// The AOT IR block the failing guard originated from.
    bid: aot_ir::BBlockId,
    /// The `GuardIdx`s of all failing guards leading up to here.
    guards: Vec<GuardIdx>,
    /// Inlined calls tracked by [trace_builder] during processing of a trace. Required for
    /// side-tracing in order setup a new [trace_builder] and process a side-trace.
    callframes: Vec<jit_ir::InlinedFrame>,
    /// Mapping of AOT variables to their current location. Used to pass variables from a parent
    /// trace into a side-trace.
    lives: Vec<(aot_ir::InstID, Location)>,
    /// The address of the root trace. This is where the side-trace needs to jump back to after it
    /// finished its execution.
    root_addr: RootTracePtr,
    /// Stack pointer offset of the root trace from the interpreter frame. Required to reset RSP to
    /// the root trace's frame, before jumping back to it.
    root_offset: usize,
    /// The live variables at the entry point of the root trace.
    entry_vars: Vec<codegen::reg_alloc::VarLocation<Register>>,
    /// Stack pointer offset from the base pointer of the interpreter frame including the
    /// interpreter frame itself and all parent traces. Since all traces execute in the interpreter
    /// frame, each trace adds to this value, making extra space on the stack. This then forms the
    /// new `sp_offset` for any side-traces of this trace.
    sp_offset: usize,
}

impl<Register: Send + Sync + 'static> SideTraceInfo for YkSideTraceInfo<Register> {
    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
        self
    }
}

impl<Register: Send + Sync> YkSideTraceInfo<Register> {
    /// Return the live call frames which are required to setup the trace builder during
    /// side-tracing.
    fn callframes(&self) -> &[jit_ir::InlinedFrame] {
        &self.callframes
    }

    /// Return the live AOT variables for this guard. Used to write live values to during deopt.
    fn lives(&self) -> &[(aot_ir::InstID, Location)] {
        &self.lives
    }

    /// Get the address of the root trace. This is where we need jump to at the end of a
    /// side-trace.
    fn root_addr(&self) -> *const libc::c_void {
        self.root_addr.0
    }

    fn root_offset(&self) -> usize {
        self.root_offset
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
        sti: Option<Arc<dyn SideTraceInfo>>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        // If either `unwrap` fails, there is no chance of the system working correctly.
        let aot_mod = &*AOT_MOD;

        if should_log_ir(IRPhase::AOT) {
            log_ir(&format!(
                "--- Begin aot ---\n{}\n--- End aot ---\n",
                aot_mod
            ));
        }

        let sti = sti.map(|s| s.as_any().downcast::<YkSideTraceInfo<Register>>().unwrap());
        let sp_offset = sti.as_ref().map(|x| x.sp_offset);
        let root_offset = sti.as_ref().map(|x| x.root_offset);
        let guards = sti.as_ref().map(|x| x.guards.clone());

        let mut jit_mod =
            trace_builder::build(&mt, aot_mod, aottrace_iter, sti, promotions, debug_strs)?;

        if should_log_ir(IRPhase::PreOpt) {
            log_ir(&format!(
                "--- Begin jit-pre-opt ---\n{jit_mod}\n--- End jit-pre-opt ---\n",
            ));
        }

        if *YKD_OPT {
            jit_mod = opt::opt(jit_mod)?;
            if should_log_ir(IRPhase::PostOpt) {
                jit_mod.dead_code_elimination();
                log_ir(&format!(
                    "--- Begin jit-post-opt ---\n{jit_mod}\n--- End jit-post-opt ---\n",
                ));
            }
        }

        // FIXME: This needs to be the combined stacksize of all parent traces.
        let ct = self
            .codegen
            .codegen(jit_mod, mt, hl, sp_offset, root_offset, guards)?;

        if should_log_ir(IRPhase::Asm) {
            log_ir(&format!(
                "--- Begin jit-asm ---\n{}\n--- End jit-asm ---\n",
                ct.disassemble(false).unwrap()
            ));
        }
        if should_log_ir(IRPhase::AsmFull) {
            log_ir(&format!(
                "--- Begin jit-asm-full ---\n{}\n--- End jit-asm-full ---\n",
                ct.disassemble(true).unwrap()
            ));
        }

        Ok(ct)
    }
}

impl<Register: Send + Sync + 'static> Compiler for JITCYk<Register> {
    fn root_compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        self.compile(mt, aottrace_iter, None, hl, promotions, debug_strs)
    }

    fn sidetrace_compile(
        &self,
        mt: Arc<MT>,
        aottrace_iter: Box<dyn AOTTraceIterator>,
        sti: Arc<dyn SideTraceInfo>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        self.compile(mt, aottrace_iter, Some(sti), hl, promotions, debug_strs)
    }
}

pub(crate) fn yk_ir_section() -> Result<&'static [u8], Box<dyn Error>> {
    let start = symbol_to_ptr("ykllvm.yk_ir.start")? as *const u8;
    let stop = symbol_to_ptr("ykllvm.yk_ir.stop")? as *const u8;
    debug_assert!(start < stop);
    Ok(unsafe { slice::from_raw_parts(start, stop.sub(start as usize) as usize) })
}
