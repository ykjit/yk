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
///
/// All code addresses are virtual addresses, since the LLVM blockmap section is relocated.
#[derive(Debug)]
pub enum SuccessorKind {
    /// One successor.
    Unconditional {
        /// The successor's virtual address, or `None` if control flow is divergent.
        target: Option<usize>,
    },
    /// Choice of two successors.
    Conditional {
        /// The number of conditional branch instructions terminating the block.
        ///
        /// This isn't necessarily 1 as you might expect. E.g. LLVM uses the `X86::COND_NE_OR_P`
        /// and `X86::COND_E_AND_NP` terminators, which are actually two consecutive conditional
        /// branches.
        num_cond_brs: u8,
        /// The virtual address of the "taken" successor.
        taken_target: usize,
        /// The virtual address of the "not taken" successor, or `None` if control flow is
        /// divergent.
        not_taken_target: Option<usize>,
    },
    /// A return edge.
    Return,
    /// Any other control flow edge known only at runtime, e.g. an indirect branch.
    Dynamic,
}

/// Information about an machine-level LLVM call instruction.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct CallInfo {
    /// The virtual address of the call instruction
    callsite_vaddr: usize,
    /// The virtual address of the return address (should the call return conventionally).
    return_vaddr: usize,
    /// The virtual address of the target of the call (if known statically).
    target_vaddr: Option<usize>,
    /// Indicates if the call is direct (true) or indirect (false).
    direct: bool,
}

impl CallInfo {
    pub fn callsite_vaddr(&self) -> usize {
        self.callsite_vaddr
    }

    pub fn return_vaddr(&self) -> usize {
        self.return_vaddr
    }

    pub fn target_vaddr(&self) -> Option<usize> {
        self.target_vaddr
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
    /// Virtual addresses of call instructions.
    call_vaddrs: Vec<CallInfo>,
}

impl BlockMapEntry {
    pub fn corr_bbs(&self) -> &Vec<u64> {
        &self.corr_bbs
    }

    pub fn successor(&self) -> &SuccessorKind {
        &self.succ
    }

    pub fn call_vaddrs(&self) -> &Vec<CallInfo> {
        &self.call_vaddrs
    }
}

/// Maps block virtual addressed to their corresponding block map entry.
#[derive(Debug)]
pub struct BlockMap {
    tree: IntervalTree<usize, BlockMapEntry>,
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
            let mut last_vaddr = usize::try_from(crsr.read_u64::<NativeEndian>().unwrap()).unwrap();
            let n_blks = leb128::read::unsigned(&mut crsr).unwrap();
            for _ in 0..n_blks {
                let mut corr_bbs = Vec::new();
                if version > 1 {
                    let _bbid = leb128::read::unsigned(&mut crsr).unwrap();
                }
                let b_off = usize::try_from(leb128::read::unsigned(&mut crsr).unwrap()).unwrap();
                // Skip the block size. We still have to parse the field, as it's variable-size.
                let b_sz = usize::try_from(leb128::read::unsigned(&mut crsr).unwrap()).unwrap();
                // Skip over block meta-data.
                crsr.seek(SeekFrom::Current(1)).unwrap();
                // Read the indices of the BBs corresponding with this MBB.
                let num_corr = leb128::read::unsigned(&mut crsr).unwrap();
                for _ in 0..num_corr {
                    corr_bbs.push(leb128::read::unsigned(&mut crsr).unwrap());
                }

                // Read call information.
                let num_calls = leb128::read::unsigned(&mut crsr).unwrap();
                let mut call_vaddrs = Vec::new();
                for _ in 0..num_calls {
                    let callsite_vaddr =
                        usize::try_from(crsr.read_u64::<NativeEndian>().unwrap()).unwrap();
                    let return_vaddr =
                        usize::try_from(crsr.read_u64::<NativeEndian>().unwrap()).unwrap();
                    debug_assert!(callsite_vaddr < return_vaddr);
                    let target = usize::try_from(crsr.read_u64::<NativeEndian>().unwrap()).unwrap();
                    let target_vaddr = if target != 0 { Some(target) } else { None };
                    let direct = crsr.read_u8().unwrap() == 1;
                    call_vaddrs.push(CallInfo {
                        callsite_vaddr,
                        return_vaddr,
                        target_vaddr,
                        direct,
                    })
                }

                let read_maybe_divergent_target = |crsr: &mut Cursor<_>| {
                    let target = crsr.read_u64::<NativeEndian>().unwrap();
                    if target != 0 {
                        Some(usize::try_from(target).unwrap())
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
                        taken_target: usize::try_from(crsr.read_u64::<NativeEndian>().unwrap())
                            .unwrap(),
                        not_taken_target: read_maybe_divergent_target(&mut crsr),
                    },
                    2 => SuccessorKind::Return,
                    3 => SuccessorKind::Dynamic,
                    _ => unreachable!(),
                };

                let lo = last_vaddr + b_off;
                let hi = lo + b_sz;
                elems.push((
                    (lo..hi),
                    BlockMapEntry {
                        corr_bbs,
                        call_vaddrs,
                        succ,
                    },
                ));
                last_vaddr = hi;
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
        start_off: usize,
        end_off: usize,
    ) -> intervaltree::QueryIter<'_, usize, BlockMapEntry> {
        self.tree.query(start_off..end_off)
    }
}
