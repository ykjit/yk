//! The mapper translates a PT trace into an IR trace.

use crate::trace::IRBlock;
use libc::c_void;
use perftracer::llvm_blockmap::LLVM_BLOCK_MAP;
use perftracer::{Block, HWTracerError};
use std::{collections::HashMap, convert::TryFrom, ffi::CString};
use ykutil::{
    addr::{vaddr_to_obj_and_off, vaddr_to_sym_and_obj},
    obj::SELF_BIN_PATH,
};

/// Maps each entry of a hardware trace back to the IR block from which it was compiled.
pub struct HWTMapper {
    faddrs: HashMap<CString, *const c_void>,
}

impl<'a> HWTMapper {
    pub fn new() -> Self {
        Self {
            faddrs: HashMap::new(),
        }
    }

    pub fn faddrs(self) -> HashMap<CString, *const c_void> {
        self.faddrs
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
    fn map_block(&mut self, block: &perftracer::Block) -> Vec<Option<IRBlock>> {
        let b_rng = block.vaddr_range();
        if b_rng.is_none() {
            // If the address range of the block isn't known, then it follows that we can't map
            // back to an IRBlock. We return the empty vector to flag this.
            return Vec::new();
        }
        let (block_vaddr, block_last_instr) = b_rng.unwrap();

        let (obj_name, block_off) = vaddr_to_obj_and_off(block_vaddr as usize).unwrap();

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

        let block_len = block_last_instr - block_vaddr;
        let mut ret = Vec::new();
        let mut ents = LLVM_BLOCK_MAP
            .query(block_off, block_off + block_len)
            .collect::<Vec<_>>();

        // In the case that a PT block maps to multiple machine blocks, it may be tempting to check
        // that they are at consecutive address ranges. Unfortunately we can't do this because LLVM
        // sometimes appends `nop` sleds (e.g. `nop word cs:[rax + rax]; nop`) to the ends of
        // blocks for alignment. This padding is not reflected in the LLVM block address map, so
        // blocks may not appear consecutive.
        ents.sort_by(|x, y| x.range.start.partial_cmp(&y.range.start).unwrap());
        for ent in ents {
            if !ent.value.corr_bbs().is_empty() {
                // OPT: This could probably be sped up with caching. If we use an interval tree
                // keyed virtual address ranges, then we could take advantage of the fact that all
                // blocks belonging to the same function will fall within the address range of the
                // function's symbol. If the cache knows that block A and B are from the same
                // function, and a block X has a start address between blocks A and B, then X must
                // also belong to the same function and there's no need to query the linker.
                // FIXME: Is this `unwrap` safe?
                let sio = vaddr_to_sym_and_obj(usize::try_from(block_vaddr).unwrap()).unwrap();
                debug_assert_eq!(
                    obj_name.to_str().unwrap(),
                    sio.dli_fname().unwrap().to_str().unwrap()
                );
                if let Some(sym_name) = sio.dli_sname() {
                    if !self.faddrs.contains_key(sym_name) {
                        self.faddrs
                            .insert(sym_name.to_owned(), sio.dli_saddr() as *const c_void);
                    }
                    for bb in ent.value.corr_bbs() {
                        ret.push(Some(IRBlock::new_mapped(
                            sym_name.to_owned(),
                            usize::try_from(*bb).unwrap(),
                        )));
                    }
                } else {
                    ret.push(None);
                }
            } else {
                ret.push(None);
            }
        }
        ret
    }

    /// Map the *machine* blocks of the specified trace into LLVM IR blocks.
    ///
    /// Each entry in the returned trace is either a "mapped block" identifying a successfully
    /// mapped LLVM IR block, or an unsuccessfully mapped "unmappable block" (an unknown region of
    /// code spanning at least one machine block).
    ///
    /// In the returned trace, unmappable blocks never appear consecutively.
    ///
    /// The returned trace will always start with a mapped block (the unmappable prefix of the
    /// foreign "turn on tracing" routine is omitted).
    pub fn map_trace(
        &mut self,
        mut trace_iter: &'a mut dyn Iterator<Item = Result<Block, HWTracerError>>,
    ) -> Result<Vec<IRBlock>, HWTracerError> {
        let mut ret: Vec<IRBlock> = Vec::new();

        for block in &mut trace_iter {
            let block = block?;
            let irblocks = self.map_block(&block);
            if irblocks.is_empty() {
                // The block is unmappable. Insert a IRBlock that indicates this, but only if the
                // trace isn't empty (we never report the leading unmappable code in a trace). We
                // also take care to collapse consecutive unmappable blocks into one.
                if let Some(last) = ret.last_mut() {
                    if !last.is_unmappable() {
                        ret.push(IRBlock::new_unmappable(block.stack_adjust().unwrap()));
                    } else {
                        // The previous entry in the trace is already and unmappable region. Don't
                        // push, thus collapsing repeated unmappable blocks into one. We do have to
                        // sum together the stack adjust values though!
                        *last.stack_adjust_mut() += block.stack_adjust().unwrap();
                    }
                }
            } else {
                for irblock in irblocks.into_iter() {
                    if let Some(irb) = irblock {
                        match ret.last() {
                            Some(last) if &irb != last => ret.push(irb),
                            Some(_) => {
                                // The `BlockDisambiguate` pass in ykllvm ensures that no
                                // high-level LLVM IR block ever branches straight back to itself,
                                // so if we see the same high-level block more than once
                                // consecutively in a trace, then we know that the IR block has
                                // been lowered to multiple machine blocks during code-gen, and
                                // that we should only push the IR block once.
                            }
                            None => {
                                // The returned trace is empty thus far and `irb` is mappable, so
                                // we always want to push that.
                                ret.push(irb);
                            }
                        }
                    } else {
                        // Part of a PT block mapped to a machine block in the LLVM block
                        // address map, but the machine block has no corresponding IR blocks.
                        //
                        // FIXME: https://github.com/ykjit/yk/issues/388
                        // We *think* this happens because LLVM can introduce extra
                        // `MachineBasicBlock`s to help with laying out machine code. If that's
                        // the case, then for our purposes these extra blocks can be ignored.
                        // However, we should really investigate to be sure.
                    }
                }
            }
        }
        Ok(ret)
    }
}
