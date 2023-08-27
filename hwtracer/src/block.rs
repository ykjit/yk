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
    /// An unknown virtual address range.
    ///
    /// This is required because decoders don't have perfect knowledge about every block
    /// in the virtual address space.
    Unknown {
        /// The stack adjustment required as a consequence of executing this unknown code.
        stack_adjust: isize,
    },
}

impl fmt::Debug for Block {
    /// Format virtual addresses using hexadecimal.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::VAddrRange {
                first_instr,
                last_instr,
            } => {
                write!(f, "Block({:x}..={:x})", first_instr, last_instr)
            }
            Self::Unknown { stack_adjust } => {
                write!(f, "UnknownBlock(stack_adjust={stack_adjust})")
            }
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

    /// Returns `true` if `self` represents an unknown virtual address range.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
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

    /// Create an unknown block.
    pub fn new_unknown() -> Self {
        Self::Unknown { stack_adjust: 0 }
    }

    /// Return the stack adjustment value, if applicable.
    pub fn stack_adjust(&self) -> Option<isize> {
        if let Self::Unknown { stack_adjust } = self {
            Some(*stack_adjust)
        } else {
            None
        }
    }

    /// Return a mutable reference to the stack adjustment value, if applicable.
    pub fn stack_adjust_mut(&mut self) -> Option<&mut isize> {
        if let Self::Unknown {
            ref mut stack_adjust,
        } = self
        {
            Some(stack_adjust)
        } else {
            None
        }
    }
}
