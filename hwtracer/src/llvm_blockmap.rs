//! Parser for ykllvm's extended `.llvm_bb_addr_map`.

use byteorder::{NativeEndian, ReadBytesExt};
use intervaltree::IntervalTree;
use libc::{dlsym, RTLD_DEFAULT};
use std::{
    convert::TryFrom,
    ffi::CString,
    io::{prelude::*, Cursor, SeekFrom},
    slice,
    sync::LazyLock,
};

pub static LLVM_BLOCK_MAP: LazyLock<BlockMap> = LazyLock::new(BlockMap::new);

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

// ykllvm inserts a symbol pair marking the extent of the `.llvm_bb_addr_map` section.
// This function returns a byte slice of the memory between these two marker symbols.
//
// Note that in the "common use case" this lookup could be done statically (without `dlsym`) using:
//
// ```
// extern "C" {
//   #[link(name = "ykllvm.bbaddrmaps.start")]
//   BBMAPS_START_BYTE: u8;
//   #[link(name = "ykllvm.bbaddrmaps.stop")]
//   BBMAPS_STOP_BYTE: u8;
// }
// ```
//
// however, this would force every binary that uses this crate to provide the symbols. This is not
// desirable, e.g. Rust test binaries.
fn find_blockmap_section() -> &'static [u8] {
    let start_sym = CString::new("ykllvm.bbaddrmaps.start").unwrap();
    let start_addr = unsafe { dlsym(RTLD_DEFAULT, start_sym.as_ptr()) } as *const u8;
    if start_addr.is_null() {
        panic!("can't find ykllvm.bbaddrmaps.start");
    }

    let stop_sym = CString::new("ykllvm.bbaddrmaps.stop").unwrap();
    let stop_addr = unsafe { dlsym(RTLD_DEFAULT, stop_sym.as_ptr()) } as *const u8;
    if stop_addr.is_null() {
        panic!("can't find ykllvm.bbaddrmaps.stop");
    }

    debug_assert!(stop_addr > start_addr);
    unsafe { slice::from_raw_parts(start_addr, stop_addr.sub_ptr(start_addr)) }
}

/// Maps (unrelocated) block offsets to their corresponding block map entry.
pub struct BlockMap {
    tree: IntervalTree<u64, BlockMapEntry>,
}

impl BlockMap {
    /// Parse the LLVM blockmap section of the current executable and return a struct holding the
    /// mappings.
    pub fn new() -> Self {
        let bbaddrmap_data = find_blockmap_section();

        // Keep reading blockmap records until we fall outside of the section's bounds.
        let mut elems = Vec::new();
        let mut crsr = Cursor::new(bbaddrmap_data);
        while crsr.position() < u64::try_from(bbaddrmap_data.len()).unwrap() {
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
                // conditional, taken-address, not-taken-address
                let succ = match crsr.read_u8().unwrap() {
                    0 => SuccessorKind::Unconditional {
                        target: read_maybe_divergent_target(&mut crsr),
                    },
                    1 => SuccessorKind::Conditional {
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
