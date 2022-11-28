use std::fmt;

#[cfg(target_arch = "x86_64")]
type BlockAddr = u64;

/// Information about a trace decoder's notion of a basic block.
///
/// The exact definition of a basic block will vary from collector to collector.
#[derive(Eq, PartialEq)]
pub enum Block {
    /// An address range that captures at least the first byte of every machine instruction in the
    /// block.
    VAddrRange {
        /// Virtual address of the start of the first instruction in this block.
        first_instr: BlockAddr,
        /// Virtual address of *any* byte of the last instruction in this block.
        last_instr: BlockAddr,
    },
    /// An unkonwn address range.
    Unknown,
}

impl fmt::Debug for Block {
    /// Format virtual addresses using hexidecimal.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Self::VAddrRange {
            first_instr,
            last_instr,
        } = self
        {
            write!(f, "Block({:x}..={:x})", first_instr, last_instr)
        } else {
            write!(f, "UnkonwnBlock(???..=???)")
        }
    }
}

impl Block {
    /// Creates a new basic block from the virtual addresses of:
    ///   * the start of the first instruction in the basic block.
    ///   * the start of the last instruction in the basic block.
    pub fn from_vaddr_range(first_instr: BlockAddr, last_instr: BlockAddr) -> Self {
        Self::VAddrRange {
            first_instr,
            last_instr,
        }
    }

    pub fn unknown() -> Self {
        Self::Unknown
    }

    /// If `self` represents a known address range, returns the address range, otherwise `None`.
    pub fn vaddr_range(&self) -> Option<(BlockAddr, BlockAddr)> {
        if let Self::VAddrRange {
            first_instr,
            last_instr,
        } = self
        {
            Some((*first_instr, *last_instr))
        } else {
            None
        }
    }
}
