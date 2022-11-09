#[cfg(target_arch = "x86_64")]
type BlockAddr = u64;

/// Information about a basic block.
#[derive(Debug, Eq, PartialEq)]
pub struct Block {
    /// Virtual address of the start of the first instruction in this block.
    first_instr: BlockAddr,
    /// Virtual address of the start of the last instruction in this block.
    last_instr: BlockAddr,
}

impl Block {
    /// Creates a new basic block from the virtual addresses of:
    ///   * the start of the first instruction in the basic block.
    ///   * the start of the last instruction in the basic block.
    pub fn new(first_instr: BlockAddr, last_instr: BlockAddr) -> Self {
        Self {
            first_instr,
            last_instr,
        }
    }

    /// Returns the virtual address of the start of the first instruction in this block.
    pub fn first_instr(&self) -> BlockAddr {
        self.first_instr
    }

    /// Returns the virtual address of the start of the last instruction in this block.
    pub fn last_instr(&self) -> BlockAddr {
        self.last_instr
    }
}
