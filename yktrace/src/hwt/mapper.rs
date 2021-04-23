//! The mapper translates a PT trace into an IR trace.

use crate::IRBlock;
use hwtracer::{HWTracerError, Trace};
use phdrs::objects;
use std::convert::TryFrom;

pub struct HWTMapper {
    phdr_offset: u64,
}

impl HWTMapper {
    pub(super) fn new() -> HWTMapper {
        let phdr_offset = get_phdr_offset();
        HWTMapper { phdr_offset }
    }

    /// Maps each entry of a hardware trace back the IR block from whence it was compiled.
    pub(super) fn map_trace(&self, trace: Box<dyn Trace>) -> Result<Vec<IRBlock>, HWTracerError> {
        let blocks = Vec::new();
        for block in trace.iter_blocks() {
            let block = block?;

            // FIXME If the mapping using LLVM is precise, we might be able to get rid of end_addr?
            let _start_addr = usize::try_from(block.first_instr() - self.phdr_offset).unwrap();
            let _end_addr = usize::try_from(block.last_instr() - self.phdr_offset).unwrap();
            todo!(); // FIXME use LLVM mapping info to find the right block in IR.
        }
        Ok(blocks)
    }
}

/// Extract the program header offset.
fn get_phdr_offset() -> u64 {
    (&objects()[0]).addr() as u64
}
