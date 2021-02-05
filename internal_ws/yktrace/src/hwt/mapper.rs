//! XXX needs a top-level doc comment

use object::{Object, ObjectSection};
use phdrs::objects;

use crate::sir::SirLoc;
use hwtracer::{HWTracerError, Trace};
use intervaltree::IntervalTree;
use lazy_static::lazy_static;
use std::{convert::TryFrom, env, fs, iter::FromIterator};
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
    static ref LABELS: IntervalTree<usize, SirLabel> = load_labels();
}

pub struct HWTMapper {
    phdr_offset: u64
}

impl HWTMapper {
    pub(super) fn new() -> HWTMapper {
        let phdr_offset = get_phdr_offset();
        HWTMapper { phdr_offset }
    }

    /// Maps each entry of a hardware trace to the appropriate SirLoc.
    ///
    /// For each block in the trace, the interval tree is queried for labels coinciding with the
    /// block. Label addresses which coincide are therefore contained within the block, and are
    /// thus part of the SIR trace.
    pub(super) fn map_trace(&self, trace: Box<dyn Trace>) -> Result<Vec<SirLoc>, HWTracerError> {
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
            let mut locs = LABELS
                .query(start_addr..(end_addr + 1))
                .map(|e| {
                    let lbl = &e.value;
                    let vaddr = if lbl.bb == 0 {
                        Some(block.first_instr())
                    } else {
                        None
                    };
                    (lbl.off, SirLoc::new(&lbl.symbol_name, lbl.bb, vaddr))
                })
                .collect::<Vec<_>>();
            locs.sort_by_key(|l| l.0);
            annotrace.extend(locs.into_iter().map(|l| l.1));
        }
        Ok(annotrace)
    }
}

/// Extract the program header offset. This offset can be used to translate the address of a trace
/// block to a program address, allowing us to find the correct SIR location.
/// XXX if this is an address should it be usize?
fn get_phdr_offset() -> u64 {
    (&objects()[0]).addr() as u64
}

/// Loads SIR location labels from the executable.
///
/// We load the labels into an interval tree for fast queries later. Each label address is stored
/// in the tree as a "point interval" (an interval with only one member, e.g. `0..1`).
fn load_labels() -> IntervalTree<usize, SirLabel> {
    let pathb = env::current_exe().unwrap();
    let file = fs::File::open(&pathb.as_path()).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let object = object::File::parse(&*mmap).unwrap();
    let sec = object.section_by_name(ykpack::YKLABELS_SECTION).unwrap();
    let vec = bincode::deserialize::<Vec<SirLabel>>(sec.data().unwrap()).unwrap();
    IntervalTree::from_iter(vec.into_iter().map(|l| (l.off..(l.off + 1), l)))
}
