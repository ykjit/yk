//! The mapper translates a PT trace into an IR trace.

use crate::IRBlock;
use byteorder::{NativeEndian, ReadBytesExt};
use hwtracer::{HWTracerError, Trace};
use intervaltree::{self, IntervalTree};
use memmap2;
use object::{Object, ObjectSection};
use once_cell::sync::Lazy;
use std::{
    convert::TryFrom,
    env, fs,
    io::{prelude::*, Cursor, SeekFrom},
};
use ykllvmwrap::symbolizer::Symbolizer;
use ykutil::addr::code_vaddr_to_off;

const BLOCK_MAP_SEC: &str = ".llvm_bb_addr_map";
static BLOCK_MAP: Lazy<BlockMap> = Lazy::new(|| BlockMap::new());

/// The information for one basic block, as per:
/// https://llvm.org/docs/Extensions.html#sht-llvm-bb-addr-map-section-basic-block-address-map
#[derive(Debug)]
#[allow(dead_code)]
struct BlockMapEntry {
    /// Function offset.
    f_off: u64,
    /// Basic block number.
    bb: usize,
}

/// Maps (unrelocated) block offsets to their corresponding block map entry.
pub struct BlockMap {
    tree: IntervalTree<u64, BlockMapEntry>,
}

impl BlockMap {
    /// Parse the LLVM blockmap section of the current executable and return a struct holding the
    /// mappings.
    pub fn new() -> Self {
        let mut elems = Vec::new();
        let pathb = env::current_exe().unwrap();
        let file = fs::File::open(&pathb.as_path()).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let object = object::File::parse(&*mmap).unwrap();
        let sec = object.section_by_name(BLOCK_MAP_SEC).unwrap();
        let sec_size = sec.size();
        let mut crsr = Cursor::new(sec.data().unwrap());

        // Keep reading records until we fall outside of the section's bounds.
        while crsr.position() < sec_size {
            let f_off = crsr.read_u64::<NativeEndian>().unwrap();
            let n_blks = leb128::read::unsigned(&mut crsr).unwrap();
            for _ in 0..n_blks {
                let b_off = leb128::read::unsigned(&mut crsr).unwrap();
                // Skip the block size. We still have to parse the field, as it's variable-size.
                let b_sz = leb128::read::unsigned(&mut crsr).unwrap();
                // Skip over block meta-data.
                crsr.seek(SeekFrom::Current(1)).unwrap();
                let b_idx = leb128::read::unsigned(&mut crsr).unwrap();

                let lo = f_off + b_off;
                let hi = lo + b_sz;
                elems.push((
                    (lo..hi),
                    BlockMapEntry {
                        f_off,
                        bb: usize::try_from(b_idx).unwrap(),
                    },
                ));
            }
        }
        Self {
            tree: elems.into_iter().collect::<IntervalTree<_, _>>(),
        }
    }

    pub fn len(&self) -> usize {
        self.tree.iter().count()
    }

    /// Queries the blockmap for blocks whose address range coincides with `start_off..end_off`.
    fn query(
        &self,
        start_off: u64,
        end_off: u64,
    ) -> intervaltree::QueryIter<'_, u64, BlockMapEntry> {
        self.tree.query(start_off..end_off)
    }
}

pub struct HWTMapper {
    symb: Symbolizer,
}

impl HWTMapper {
    pub(super) fn new() -> HWTMapper {
        Self {
            symb: Symbolizer::new(),
        }
    }

    /// Maps each entry of a hardware trace back the IR block from whence it was compiled.
    pub(super) fn map_trace(&self, trace: Box<dyn Trace>) -> Result<Vec<IRBlock>, HWTracerError> {
        let mut ret_irblocks = Vec::new();
        let mut itr = trace.iter_blocks();
        while let Some(block) = itr.next() {
            let block = block?;
            let irblocks = self.map_block(&block);
            if !ret_irblocks.is_empty() && irblocks.is_empty() {
                // Once we have seen the last block that can be mapped we are done.
                break;
            } else {
                ret_irblocks.extend(irblocks);
            }
        }
        // No remaining blocks in the iterator should be mappable. If any can be, then we have
        // unmappable blocks in the middle of the trace and something is wrong.
        debug_assert!(itr.all(|block| self.map_block(&block.unwrap()).is_empty()));

        Ok(ret_irblocks)
    }

    /// Maps one PT block to one or many LLVM IR blocks.
    ///
    /// The reason that there may be many corresponding blocks is due to the following scenario.
    ///
    /// Suppose that the LLVM IR looked like this:
    ///
    ///   bb1:
    ///     ...
    ///     br bb2;
    ///   bb2:
    ///     ...
    ///
    /// During codegen LLVM may remove the unconditional jump and simply place bb1 and bb2
    /// consecutively, allowing bb1 to fall-thru to bb2. In the eyes of the PT block decoder, a
    /// fall-thru does not terminate a block, so whereas LLVM sees two blocks, PT sees only one.
    fn map_block(&self, block: &hwtracer::Block) -> Vec<IRBlock> {
        let block_vaddr = block.first_instr();
        let (obj_name, block_off) = code_vaddr_to_off(block_vaddr as usize).unwrap();
        let block_len = block.last_instr() - block_vaddr;

        let mut ret = Vec::new();
        let mut ents = BLOCK_MAP
            .query(block_off, block_off + block_len)
            .collect::<Vec<_>>();

        // If a PT block maps to multiple IR blocks, then the IR blocks should be at consecutive
        // addresses (they should be related only by "fall-thru", without control flow dispatch, as
        // depicted in the above doc string). For debug builds, we check this.
        #[cfg(debug_assertions)]
        let mut prev_ent: Option<&intervaltree::Element<_, _>> = None;

        ents.sort_by(|x, y| x.range.start.partial_cmp(&y.range.start).unwrap());
        for ent in ents {
            #[cfg(debug_assertions)]
            {
                if let Some(prev) = prev_ent {
                    debug_assert!(ent.range.start == prev.range.end);
                }
                prev_ent = Some(ent);
            }

            let func_name = self.symb.find_code_sym(obj_name, ent.value.f_off).unwrap();
            ret.push(IRBlock {
                func_name,
                bb: ent.value.bb,
            });
        }
        ret
    }
}
