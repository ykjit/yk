//! The spill allocator.
//!
//! This is a register allocator that always allocates to the stack, so in fact it's not much of a
//! register allocator at all.

use super::{
    super::{abs_stack::AbstractStack, jit_ir},
    LocalAlloc, RegisterAllocator, StackDirection,
};
use typed_index_collections::TiVec;

pub(crate) struct SpillAllocator {
    allocs: TiVec<jit_ir::InstrIdx, LocalAlloc>,
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
        local: jit_ir::InstrIdx,
        size: usize,
        stack: &mut AbstractStack,
    ) -> LocalAlloc {
        // Under the current design, there can't be gaps in [self.allocs] and local variable
        // allocations happen sequentially. So the local we are currently allocating should be the
        // next unallocated index.
        debug_assert!(jit_ir::InstrIdx::new(self.allocs.len()).unwrap() == local);

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

        let alloc = LocalAlloc::new_stack(alloc_off);
        self.allocs.push(alloc);
        alloc
    }

    fn allocation<'a>(&'a self, idx: jit_ir::InstrIdx) -> &'a LocalAlloc {
        &self.allocs[idx]
    }
}

#[cfg(test)]
mod tests {
    use crate::compile::jitc_yk::{
        codegen::{
            abs_stack::AbstractStack,
            reg_alloc::{LocalAlloc, RegisterAllocator, SpillAllocator, StackDirection},
        },
        jit_ir::InstrIdx,
    };

    #[test]
    fn grow_down() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsDown);

        let idx = InstrIdx::new(0).unwrap();
        sa.allocate(idx, 8, &mut stack);
        debug_assert_eq!(stack.size(), 8);
        debug_assert_eq!(sa.allocation(idx), &LocalAlloc::Stack { frame_off: 8 });

        let idx = InstrIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 9);
        debug_assert_eq!(sa.allocation(idx), &LocalAlloc::Stack { frame_off: 9 });
    }

    #[test]
    fn grow_up() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsUp);

        let idx = InstrIdx::new(0).unwrap();
        sa.allocate(idx, 8, &mut stack);
        debug_assert_eq!(stack.size(), 8);
        debug_assert_eq!(sa.allocation(idx), &LocalAlloc::Stack { frame_off: 0 });

        let idx = InstrIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 9);
        debug_assert_eq!(sa.allocation(idx), &LocalAlloc::Stack { frame_off: 8 });
    }

    #[cfg(debug_assertions)]
    #[should_panic]
    #[test]
    fn allocate_out_of_order() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsUp);
        // panics because the backing store for local allocations are a "unsparse vector" and Local
        // 0 hasn't been allocated yet.
        sa.allocate(InstrIdx::new(1).unwrap(), 1, &mut stack);
    }

    #[test]
    fn compose_alloc_and_align_down() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsDown);

        sa.allocate(InstrIdx::new(0).unwrap(), 8, &mut stack);
        stack.align(32);

        let idx = InstrIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 33);
        debug_assert_eq!(sa.allocation(idx), &LocalAlloc::Stack { frame_off: 33 });
    }

    #[test]
    fn compose_alloc_and_align_up() {
        let mut stack = AbstractStack::default();
        let mut sa = SpillAllocator::new(StackDirection::GrowsUp);

        sa.allocate(InstrIdx::new(0).unwrap(), 8, &mut stack);
        stack.align(32);

        let idx = InstrIdx::new(1).unwrap();
        sa.allocate(idx, 1, &mut stack);
        debug_assert_eq!(stack.size(), 33);
        debug_assert_eq!(sa.allocation(idx), &LocalAlloc::Stack { frame_off: 32 });
    }
}
