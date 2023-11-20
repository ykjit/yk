//! The trace builder.
//!
//! Given a mapped trace and an AOT module, assembles an in-memory Yk IR trace by copying blocks
//! from the AOT IR. The output of this process will be the input to the code generator.

use super::aot_ir::{self, Module};
use crate::trace::{MappedTrace, TracedAOTBlock};
use std::error::Error;

struct TraceBuilder<'a> {
    aot_mod: &'a Module,
    jit_mod: Module,
    mtrace: &'a MappedTrace,
}

impl<'a> TraceBuilder<'a> {
    fn new(aot_mod: &'a Module, mtrace: &'a MappedTrace) -> Self {
        Self {
            aot_mod,
            mtrace,
            jit_mod: Module::default(),
        }
    }

    fn lookup_aot_block(&self, tb: &TracedAOTBlock) -> Option<&aot_ir::Block> {
        match tb {
            TracedAOTBlock::Mapped { func_name, bb } => {
                let func_name = func_name.to_str().unwrap(); // safe: func names are valid UTF-8.
                let func = self.aot_mod.func_by_name(func_name)?;
                func.block(*bb)
            }
            TracedAOTBlock::Unmappable { .. } => None,
        }
    }

    fn build(self) -> Result<Module, Box<dyn Error>> {
        for tblk in self.mtrace.blocks() {
            match self.lookup_aot_block(tblk) {
                Some(_blk) => {
                    // Mapped block
                    todo!();
                }
                None => {
                    // Unmappable block
                    todo!();
                }
            }
        }
        Ok(self.jit_mod)
    }
}

/// Given a mapped trace (through `aot_mod`), assemble and return a Yk IR trace.
pub(super) fn build(aot_mod: &Module, mtrace: &MappedTrace) -> Result<Module, Box<dyn Error>> {
    TraceBuilder::new(aot_mod, mtrace).build()
}
