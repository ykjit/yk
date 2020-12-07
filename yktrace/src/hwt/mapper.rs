use object::{Object, ObjectSection};
use phdrs::objects;

use crate::sir::SirLoc;
use hwtracer::{HWTracerError, Trace};
use lazy_static::lazy_static;
use std::{convert::TryFrom, env, fs};
use ykpack::SirLabel;

lazy_static! {
    /// Maps a label address to its symbol name and block index.
    ///
    /// We use a vector here since we never actually look up entries by address; we only iterate
    /// over the labels checking if each address is within the range of a block.
    ///
    /// The labels are the same for each trace, and they are immutable, so it makes sense for this
    /// to be a lazy static, loaded only once and shared.
    ///
    /// FIXME if we want to support dlopen(), we will have to rethink this.
    static ref LABELS: Vec<SirLabel> = load_labels();
}

pub struct HWTMapper {
    phdr_offset: u64
}

impl HWTMapper {
    pub fn new() -> HWTMapper {
        let phdr_offset = get_phdr_offset();
        HWTMapper { phdr_offset }
    }

    /// Maps each entry of a hardware trace to the appropriate SirLoc.
    pub fn map(&self, trace: Box<dyn Trace>) -> Result<Vec<SirLoc>, HWTracerError> {
        let mut annotrace = Vec::new();
        for block in trace.iter_blocks() {
            let block = block?;

            let start_addr = usize::try_from(block.first_instr() - self.phdr_offset).unwrap();
            let end_addr = usize::try_from(block.last_instr() - self.phdr_offset).unwrap();
            // Each block reported by the hardware tracer corresponds to one or more SIR
            // blocks, so we collect them in a vector here. This is safe because:
            //
            // a) We know that the SIR blocks were compiled (by LLVM) to straight-line
            // code, otherwise a control-flow instruction would have split the code into
            // multiple PT blocks.
            //
            // b) `labels` is sorted, so the blocks will be appended to the trace in the
            // correct order.
            let mut locs = Vec::new();
            for lbl in &*LABELS {
                if lbl.off >= start_addr && lbl.off <= end_addr {
                    // Found matching label.
                    // Store the virtual address alongside the first basic block, so we can turn
                    // inlined functions into calls during tracing.
                    let vaddr = if lbl.bb == 0 {
                        Some(block.first_instr())
                    } else {
                        None
                    };
                    locs.push(SirLoc::new(lbl.symbol_name.clone(), lbl.bb, vaddr));
                } else if lbl.off > end_addr {
                    // `labels` is sorted by address, so once we see one with an address
                    // higher than `end_addr`, we know there can be no further hits.
                    break;
                }
            }
            annotrace.extend(locs);
        }
        Ok(annotrace)
    }
}

/// Extract the program header offset. This offset can be used to translate the address of a trace
/// block to a program address, allowing us to find the correct SIR location.
fn get_phdr_offset() -> u64 {
    (&objects()[0]).addr() as u64
}

/// Loads SIR location labels from the executable.
///
/// The returned vector is sorted by label address ascending (due to the encode ordering).
fn load_labels() -> Vec<SirLabel> {
    let pathb = env::current_exe().unwrap();
    let file = fs::File::open(&pathb.as_path()).unwrap();
    let mmap = unsafe { memmap::Mmap::map(&file).unwrap() };
    let object = object::File::parse(&*mmap).unwrap();
    let sec = object.section_by_name(ykpack::YKLABELS_SECTION).unwrap();
    let ret = bincode::deserialize::<Vec<SirLabel>>(sec.data().unwrap()).unwrap();
    ret
}
