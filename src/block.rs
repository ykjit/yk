/// Information about a basic block.
#[derive(Debug, Eq, PartialEq)]
pub struct Block {
    /// Virtual address of the first instruction in this block.
    first_instr: u64,
    /// Virtual address of the last instruction in this block.
    last_instr: u64,
}

impl Block {
    /// Creates a new basic block from a start address and a length in bytes.
    pub fn new(first_instr: u64, last_instr: u64) -> Self {
        Self {
            first_instr,
            last_instr,
        }
    }

    /// Returns the virtual address of the first instruction in this block.
    pub fn first_instr(&self) -> u64 {
        self.first_instr
    }

    /// Returns the virtual address of the last instruction in this block.
    pub fn last_instr(&self) -> u64 {
        self.last_instr
    }
}
