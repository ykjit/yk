//! The mapper translates a PT trace into an IR trace.

use crate::IRBlock;
use byteorder::{NativeEndian, ReadBytesExt};
use fxhash::FxHashMap;
use hwtracer::{HWTracerError, Trace};
use memmap2;
use object::{Object, ObjectSection};
use phdrs::objects;
use std::{
    convert::TryFrom,
    env, fs,
    io::{prelude::*, Cursor, SeekFrom},
};

const BLOCK_MAP_SEC: &str = ".llvm_bb_addr_map";

/// The information for one basic block, as per:
/// https://llvm.org/docs/Extensions.html#sht-llvm-bb-addr-map-section-basic-block-address-map
#[allow(dead_code)]
struct BlockMapEntry {
    /// Function offset.
    f_off: u64,
    /// Basic block number.
    bb: u8,
}

/// Maps (unrelocated) block offsets to their corresponding block map entry.
pub struct BlockMap {
    map: FxHashMap<usize, BlockMapEntry>,
}

impl BlockMap {
    /// Parse the LLVM blockmap section of the current executable and return a struct holding the
    /// mappings.
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
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
            let n_blks = crsr.read_u8().unwrap();
            for bb in 0..n_blks {
                let b_off = leb128::read::unsigned(&mut crsr).unwrap();
                // Skip the block size. We still have to parse the field, as it's variable-size.
                let _b_sz = leb128::read::unsigned(&mut crsr).unwrap();
                // Skip over block meta-data.
                crsr.seek(SeekFrom::Current(1)).unwrap();

                map.insert(
                    usize::try_from(f_off + b_off).unwrap(),
                    BlockMapEntry { f_off, bb },
                );
            }
        }
        Self { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

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
