//! The mapper translates a PT trace into an IR trace.

use crate::IRBlock;
use byteorder::{NativeEndian, ReadBytesExt};
use hwtracer::{decode::TraceDecoderBuilder, HWTracerError, Trace};
use intervaltree::{self, IntervalTree};
use libc::c_void;
use std::{
    collections::HashMap,
    convert::TryFrom,
    ffi::CString,
    io::{prelude::*, Cursor, SeekFrom},
    slice,
    sync::LazyLock,
};
use ykutil::{
    addr::{code_vaddr_to_off, vaddr_to_sym_and_obj},
    obj::SELF_BIN_PATH,
};

static BLOCK_MAP: LazyLock<BlockMap> = LazyLock::new(BlockMap::new);

/// The information for one LLVM MachineBasicBlock, as per:
/// https://llvm.org/docs/Extensions.html#sht-llvm-bb-addr-map-section-basic-block-address-map
#[derive(Debug)]
#[allow(dead_code)]
struct BlockMapEntry {
    /// Function offset.
    f_off: u64,
    /// Indices of corresponding BasicBlocks.
    corr_bbs: Vec<u64>,
}

// ykllvm inserts a symbol pair marking the extent of the `.llvm_bb_addr_map` section.
extern "C" {
    #[link_name = "ykllvm.bbaddrmaps.start"]
    static BBMAPS_START_BYTE: u8;
    #[link_name = "ykllvm.bbaddrmaps.stop"]
    static BBMAPS_STOP_BYTE: u8;
}

/// Maps (unrelocated) block offsets to their corresponding block map entry.
pub struct BlockMap {
    tree: IntervalTree<u64, BlockMapEntry>,
}

impl BlockMap {
    /// Parse the LLVM blockmap section of the current executable and return a struct holding the
    /// mappings.
    pub fn new() -> Self {
        let start_addr = unsafe { &BBMAPS_START_BYTE as *const u8 };
        let stop_addr = unsafe { &BBMAPS_STOP_BYTE as *const u8 };
        debug_assert!(stop_addr > start_addr);
        let bbaddrmap_data =
            unsafe { slice::from_raw_parts(start_addr, stop_addr.sub_ptr(start_addr)) };

        // Keep reading blockmap records until we fall outside of the section's bounds.
        let mut elems = Vec::new();
        let mut crsr = Cursor::new(bbaddrmap_data);
        while crsr.position() < u64::try_from(bbaddrmap_data.len()).unwrap() {
            let _version = crsr.read_u8().unwrap();
            let _feature = crsr.read_u8().unwrap();
            let f_off = crsr.read_u64::<NativeEndian>().unwrap();
            let n_blks = leb128::read::unsigned(&mut crsr).unwrap();
            let mut last_off = f_off;
            for _ in 0..n_blks {
                let mut corr_bbs = Vec::new();
                let b_off = leb128::read::unsigned(&mut crsr).unwrap();
                // Skip the block size. We still have to parse the field, as it's variable-size.
                let b_sz = leb128::read::unsigned(&mut crsr).unwrap();
                // Skip over block meta-data.
                crsr.seek(SeekFrom::Current(1)).unwrap();
                // Read the indices of the BBs corresponding with this MBB.
                let num_corr = leb128::read::unsigned(&mut crsr).unwrap();
                for _ in 0..num_corr {
                    corr_bbs.push(leb128::read::unsigned(&mut crsr).unwrap());
                }

                let lo = last_off + b_off;
                let hi = lo + b_sz;
                elems.push(((lo..hi), BlockMapEntry { f_off, corr_bbs }));
                last_off = hi;
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
    faddrs: HashMap<CString, *const c_void>,
}

impl HWTMapper {
    pub(super) fn new() -> HWTMapper {
        Self {
            faddrs: HashMap::new(),
        }
    }

    /// Maps each entry of a hardware trace back to the IR block from which it was compiled.
    pub(super) fn map_trace(
        mut self,
        trace: Box<dyn Trace>,
    ) -> Result<(Vec<IRBlock>, HashMap<CString, *const c_void>), HWTracerError> {
        let tdec = TraceDecoderBuilder::new().build().unwrap();
        let mut ret_irblocks: Vec<IRBlock> = Vec::new();
        let mut itr = tdec.iter_blocks(&*trace);
        while let Some(block) = itr.next() {
            let block = block?;
            let irblocks = self.map_block(&block);
            if irblocks.is_empty() {
                // The block is unmappable. Insert a IRBlock that indicates this, but only if the
                // trace isn't empty. We also take care to collapse repeated unmappable blocks into
                // a single unmappable IRBlock.
                if let Some(lb) = ret_irblocks.last() {
                    if !lb.is_unmappable() {
                        ret_irblocks.push(IRBlock::unmappable());
                    }
                }
            } else {
                for irblock in irblocks.into_iter() {
                    if let Some(irblock) = irblock {
                        // The `BlockDisambiguate` pass in ykllvm ensures that no high-level LLVM
                        // IR block ever branches straight back to itself, so if we see the same
                        // high-level block more than once consecutively in a trace, then we know
                        // that the IR block has been lowered to multiple machine blocks during
                        // code-gen, and that we should only push the IR block once.
                        if ret_irblocks.last() != Some(&irblock) {
                            ret_irblocks.push(irblock);
                        }
                    } else {
                        // Part of a PT block mapped to a machine block in the LLVM block address
                        // map, but the machine block has no corresponding IR blocks.
                        //
                        // FIXME: https://github.com/ykjit/yk/issues/388
                        // We *think* this happens because LLVM can introduce extra
                        // `MachineBasicBlock`s to help with laying out machine code. If that's the
                        // case, then for our purposes these extra blocks can be ignored. However,
                        // we should really investigate to be sure.
                    }
                }
            }
        }
        // Strip any trailing unmappable blocks.
        if !ret_irblocks.is_empty() && ret_irblocks.last().unwrap().is_unmappable() {
            ret_irblocks.pop();
        }

        #[cfg(debug_assertions)]
        {
            if !ret_irblocks.is_empty() {
                debug_assert!(!ret_irblocks.first().unwrap().is_unmappable());
                debug_assert!(!ret_irblocks.last().unwrap().is_unmappable());
            }
        }

        Ok((ret_irblocks, self.faddrs))
    }

    /// Maps one PT block to one or many LLVM IR blocks.
    ///
    /// Mapping a PT block to IRBlocks occurs in two phases. First the mapper tries to find machine
    /// blocks whose address ranges overlap with the address range of the PT block (by using the
    /// LLVM block address map section). Once machine blocks have been found, the mapper then tries
    /// to find which LLVM IR blocks the machine blocks are part of.
    ///
    /// A `Some` element in the returned vector means that the mapper found a machine block that
    /// maps to part of the PT block and that the machine block could be directly mapped
    /// to an IR block.
    ///
    /// A `None` element in the returned vector means that the mapper found a machine block that
    /// corresponds with part of the PT block but that the machine block could *not* be directly
    /// mapped to an IR block. This happens when `MachineBasicBlock::getBasicBlock()` returns
    /// `nullptr`.
    ///
    /// This function returns an empty vector if the PT block was unmappable (no matching machine
    /// blocks could be found).
    ///
    /// The reason we cannot simply ignore the `None` case is that it is important to differentiate
    /// "there were no matching machine blocks" from "there were matching machine blocks, but we
    /// were unable to find IR blocks for them".
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
    fn map_block(&mut self, block: &hwtracer::Block) -> Vec<Option<IRBlock>> {
        let block_vaddr = block.first_instr();
        let (obj_name, block_off) = code_vaddr_to_off(block_vaddr as usize).unwrap();

        // Currently we only read in a block map and IR for the currently running binary (and not
        // for dynamically linked shared objects). Thus, if we see code from another object, we
        // can't map it.
        //
        // FIXME: https://github.com/ykjit/yk/issues/413
        // In the future we could inline code from shared objects if they were built for use with
        // yk (i.e. they have a blockmap and IR embedded).
        if obj_name != *SELF_BIN_PATH {
            return Vec::new();
        }

        // The HW mapper gives us the start address of the last instruction in a block, which gives
        // us an exclusive upper bound when the interval tree expects an inclusive upper bound.
        // This is a problem when the last block in our sequence has a single 1 byte instruction:
        // the exclusive upper bound will not include any of that block's bytes. Fortunately, we
        // don't need to calculate the precise end address of the last instruction in a block (on
        // variable width instruction sets that's impractical!): instead we just need to ensure
        // we're at least 1 byte into that last instruction (hence the +1 below). Since every
        // instruction (and thus every block) is at least 1 byte long, this is always safe, and
        // ensures that we generate an inclusive upper bound.
        let block_len = (block.last_instr() - block_vaddr) + 1;

        let mut ret = Vec::new();
        let mut ents = BLOCK_MAP
            .query(block_off, block_off + block_len)
            .collect::<Vec<_>>();

        // In the case that a PT block maps to multiple machine blocks, it may be tempting to check
        // that they are at consecutive address ranges. Unfortunately we can't do this because LLVM
        // sometimes appends `nop` sleds (e.g. `nop word cs:[rax + rax]; nop`) to the ends of
        // blocks for alignment. This padding is not reflected in the LLVM block address map, so
        // blocks may not appear consecutive.
        ents.sort_by(|x, y| x.range.start.partial_cmp(&y.range.start).unwrap());
        for ent in ents {
            if !ent.value.corr_bbs.is_empty() {
                // OPT: This could probably be sped up with caching. If we use an interval tree
                // keyed virtual address ranges, then we could take advantage of the fact that all
                // blocks belonging to the same function will fall within the address range of the
                // function's symbol. If the cache knows that block A and B are from the same
                // function, and a block X has a start address between blocks A and B, then X must
                // also belong to the same function and there's no need to query the linker.
                let sio = vaddr_to_sym_and_obj(usize::try_from(block_vaddr).unwrap()).unwrap();
                debug_assert_eq!(obj_name.to_str().unwrap(), sio.obj_name().to_str().unwrap());
                if !self.faddrs.contains_key(sio.sym_name()) {
                    self.faddrs
                        .insert(sio.sym_name().to_owned(), sio.sym_vaddr());
                }
                for bb in &ent.value.corr_bbs {
                    ret.push(Some(IRBlock::new(
                        sio.sym_name().to_owned(),
                        usize::try_from(*bb).unwrap(),
                    )));
                }
            } else {
                ret.push(None);
            }
        }
        ret
    }
}
