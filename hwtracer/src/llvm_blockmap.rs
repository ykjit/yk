//! Parser for ykllvm's extended `.llvm_bb_addr_map`.

use byteorder::{NativeEndian, ReadBytesExt};
use intervaltree::IntervalTree;
use object::{Object, ObjectSection};
use std::{
    io::{prelude::*, Cursor, SeekFrom},
    sync::LazyLock,
};
use ykaddr::obj::SELF_BIN_MMAP;

pub static LLVM_BLOCK_MAP: LazyLock<BlockMap> = LazyLock::new(|| {
    let object = object::File::parse(&**SELF_BIN_MMAP).unwrap();
    let sec = object.section_by_name(".llvm_bb_addr_map").unwrap();
    BlockMap::new(sec.data().unwrap())
});

/// Describes the successors (if any) of an LLVM `MachineBlock`.
#[derive(Debug)]
pub enum SuccessorKind {
    /// One successor.
    Unconditional {
        /// The successor's offset, or `None` if control flow is divergent.
        target: Option<u64>,
    },
    /// Choice of two successors.
    Conditional {
        /// The number of conditional branch instructions terminating the block.
        ///
        /// This isn't necessarily 1 as you might expect. E.g. LLVM uses the `X86::COND_NE_OR_P`
        /// and `X86::COND_E_AND_NP` terminators, which are actually two consecutive conditional
        /// branches.
        num_cond_brs: u8,
        /// The offset of the "taken" successor.
        taken_target: u64,
        /// The offset of the "not taken" successor, or `None` if control flow is divergent.
        not_taken_target: Option<u64>,
    },
    /// A return edge.
    Return,
    /// Any other control flow edge known only at runtime, e.g. an indirect branch.
    Dynamic,
}

/// Information about an machine-level LLVM call instruction.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct CallInfo {
    /// Offset of the call instruction
    callsite_off: u64,
    /// Offset of the return address (should the call return conventionally).
    return_off: u64,
    /// Offset of the target of the call (if known statically).
    target_off: Option<u64>,
    /// Indicates if the call is direct (true) or indirect (false).
    direct: bool,
}

impl CallInfo {
    pub fn callsite_off(&self) -> u64 {
        self.callsite_off
    }

    pub fn return_off(&self) -> u64 {
        self.return_off
    }

    pub fn target_off(&self) -> Option<u64> {
        self.target_off
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }
}

/// The information for one LLVM `MachineBasicBlock`.
#[derive(Debug)]
pub struct BlockMapEntry {
    /// Indices of corresponding BasicBlocks.
    corr_bbs: Vec<u64>,
    /// Successor information.
    succ: SuccessorKind,
    /// Offsets of call instructions.
    call_offs: Vec<CallInfo>,
}

impl BlockMapEntry {
    pub fn corr_bbs(&self) -> &Vec<u64> {
        &self.corr_bbs
    }

    pub fn successor(&self) -> &SuccessorKind {
        &self.succ
    }

    pub fn call_offs(&self) -> &Vec<CallInfo> {
        &self.call_offs
    }
}

/// Maps (unrelocated) block offsets to their corresponding block map entry.
pub struct BlockMap {
    tree: IntervalTree<u64, BlockMapEntry>,
}

impl BlockMap {
    /// Parse the LLVM blockmap section of the current executable and return a struct holding the
    /// mappings.
    pub fn new(data: &'static [u8]) -> Self {
        // Keep reading blockmap records until we fall outside of the section's bounds.
        let mut elems = Vec::new();
        let mut crsr = Cursor::new(data);
        while crsr.position() < u64::try_from(data.len()).unwrap() {
            let version = crsr.read_u8().unwrap();
            let _feature = crsr.read_u8().unwrap();
            let mut last_off = crsr.read_u64::<NativeEndian>().unwrap();
            let n_blks = leb128::read::unsigned(&mut crsr).unwrap();
            for _ in 0..n_blks {
                let mut corr_bbs = Vec::new();
                if version > 1 {
                    let _bbid = leb128::read::unsigned(&mut crsr).unwrap();
                }
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

                // Read call information.
                let num_calls = leb128::read::unsigned(&mut crsr).unwrap();
                let mut call_offs = Vec::new();
                for _ in 0..num_calls {
                    let callsite_off = crsr.read_u64::<NativeEndian>().unwrap();
                    let return_off = crsr.read_u64::<NativeEndian>().unwrap();
                    debug_assert!(callsite_off < return_off);
                    let target = crsr.read_u64::<NativeEndian>().unwrap();
                    let target_off = if target != 0 { Some(target) } else { None };
                    let direct = crsr.read_u8().unwrap() == 1;
                    call_offs.push(CallInfo {
                        callsite_off,
                        return_off,
                        target_off,
                        direct,
                    })
                }

                let read_maybe_divergent_target = |crsr: &mut Cursor<_>| {
                    let target = crsr.read_u64::<NativeEndian>().unwrap();
                    if target != 0 {
                        Some(target)
                    } else {
                        None // Divergent.
                    }
                };

                // Read successor info.
                // unconditional, target-address
                // conditional, num-cond-brs, taken-address, not-taken-address
                let succ = match crsr.read_u8().unwrap() {
                    0 => SuccessorKind::Unconditional {
                        target: read_maybe_divergent_target(&mut crsr),
                    },
                    1 => SuccessorKind::Conditional {
                        num_cond_brs: crsr.read_u8().unwrap(),
                        taken_target: crsr.read_u64::<NativeEndian>().unwrap(),
                        not_taken_target: read_maybe_divergent_target(&mut crsr),
                    },
                    2 => SuccessorKind::Return,
                    3 => SuccessorKind::Dynamic,
                    _ => unreachable!(),
                };

                let lo = last_off + b_off;
                let hi = lo + b_sz;
                elems.push((
                    (lo..hi),
                    BlockMapEntry {
                        corr_bbs,
                        call_offs,
                        succ,
                    },
                ));
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
    pub fn query(
        &self,
        start_off: u64,
        end_off: u64,
    ) -> intervaltree::QueryIter<'_, u64, BlockMapEntry> {
        self.tree.query(start_off..end_off)
    }
}
