//! The spill allocator.
//!
//! This is a register allocator that always allocates to the stack, so in fact it's not much of a
//! register allocator at all.

use super::{
    super::{abs_stack::AbstractStack, jit_ir},
    RegisterAllocator, StackDirection, VarLocation,
};
use std::collections::HashMap;

pub(crate) struct SpillAllocator {
    /// Maps a local variable (the instruction that defines it) to its allocation.
    allocs: HashMap<jit_ir::InstIdx, VarLocation>,
    stack_dir: StackDirection,
}

impl RegisterAllocator for SpillAllocator {
    fn new(stack_dir: StackDirection) -> SpillAllocator {
        Self {
            allocs: Default::default(),
            stack_dir,
        }
    }

    fn allocate(
        &mut self,
        local: jit_ir::InstIdx,
        size: usize,
        stack: &mut AbstractStack,
    ) -> VarLocation {
        // Align the stack to the size of the allocation.
        //
        // FIXME: perhaps we should align to the largest alignment of the constituent fields?
        // To do this we need to first finish proper type sizes.
        let post_align_off = stack.align(size);

        // Make space for the allocation.
        let post_grow_off = stack.grow(size);

        // If the stack grows up, then the allocation's offset is the stack height *before* we've
        // made space on the stack, otherwise it's the stack height *after*.
        let alloc_off = match self.stack_dir {
            StackDirection::GrowsUp => post_align_off,
            StackDirection::GrowsDown => post_grow_off,
        };

        let alloc = VarLocation::new_stack(alloc_off, size);
        self.allocs.insert(local, alloc);
        alloc
    }

    /// Returns the location for an SSA variable.
    ///
    /// # Panics
    ///
    /// Panics if there is no such SSA variable.
    fn location(&self, idx: jit_ir::InstIdx) -> &VarLocation {
        &self.allocs[&idx]
    }
}

#[cfg(test)]
mod tests {
    use crate::compile::jitc_yk::{
        codegen::{
            abs_stack::AbstractStack,
            reg_alloc::{RegisterAllocator, SpillAllocator, StackDirection, VarLocation},
        },
        jit_ir::InstIdx,
    };

    #[test]
    fn grow_down() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsDown);

        let idx = InstIdx::new(0).unwrap();
        sa.allocate(idx, 8, &mut stack);
        debug_assert_eq!(stack.size(), 8);
        debug_assert_eq!(
            sa.location(idx),
            &VarLocation::Stack {
                frame_off: 8,
                size: 8
            }
        );

        let idx = InstIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 9);
        debug_assert_eq!(
            sa.location(idx),
            &VarLocation::Stack {
                frame_off: 9,
                size: 1
            }
        );
    }

    #[test]
    fn grow_up() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsUp);

        let idx = InstIdx::new(0).unwrap();
        sa.allocate(idx, 8, &mut stack);
        debug_assert_eq!(stack.size(), 8);
        debug_assert_eq!(
            sa.location(idx),
            &VarLocation::Stack {
                frame_off: 0,
                size: 8
            }
        );

        let idx = InstIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 9);
        debug_assert_eq!(
            sa.location(idx),
            &VarLocation::Stack {
                frame_off: 8,
                size: 1
            }
        );
    }

    #[test]
    fn compose_alloc_and_align_down() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsDown);

        sa.allocate(InstIdx::new(0).unwrap(), 8, &mut stack);
        stack.align(32);

        let idx = InstIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 33);
        debug_assert_eq!(
            sa.location(idx),
            &VarLocation::Stack {
                frame_off: 33,
                size: 1
            }
        );
    }

    #[test]
    fn compose_alloc_and_align_up() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsUp);

        sa.allocate(InstIdx::new(0).unwrap(), 8, &mut stack);
        stack.align(32);

        let idx = InstIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 33);
        debug_assert_eq!(
            sa.location(idx),
            &VarLocation::Stack {
                frame_off: 32,
                size: 1
            }
        );
    }
}
