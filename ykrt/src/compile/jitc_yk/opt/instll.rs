//! A linked list of instruction types.
//!
//! This allows one to iterate backwards over a trace starting at instruction X and efficiently
//! view all previous instructions of the same kind as X.
//!
//! For example given the trace:
//!
//! ```text
//! %0: i8 = param reg
//! %1: ptr = param reg
//! %2: i8 = add %0, %0
//! %3: i8 = add %0, %0
//! %4: i8 = load %1
//! %5: i8 = add %0, %0
//! ```
//!
//! If we iterate backwards from `%5`, this linked list will successively return the values `%3`
//! and `%2` (i.e. the other `BinOp` functions).
//!
//! The internal data structure is modelled on an equivalent in LuaJIT. It is highly efficient,
//! requiring only a single memory allocation with one `InstIdx` per instruction in a trace.

use crate::compile::jitc_yk::jit_ir::{Inst, InstIdx, Module};
use strum::EnumCount;

pub(super) struct InstLinkedList {
    /// The head of the linked list with one entry for each [Inst] discriminant. This points to the
    /// most recent entry in [Self::predecessors].
    heads: [InstIdx; Inst::COUNT],
    /// The linked list with one entry for each instruction in a trace.
    predecessors: Vec<InstIdx>,
}

impl InstLinkedList {
    /// Create an empty linked list for a module `m`. This does not prepopulate the linked-list:
    /// instead call [Self::push] for each element in the trace as you encounter it.
    pub(super) fn new(m: &Module) -> Self {
        // Because we use `X::MAX` as our "we haven't seen anything here yet" value, we have to
        // make sure we won't overlap with a genuine value. These are deliberately `assert`s,
        // because we don't yet guarantee elsewhere that the max value of either is actually
        // `max-1`. The second `assert` might seem like it should be `debug_assert`, but we don't
        // know if `Inst::COUNT` has the same size in debug/release builds, so better safe than
        // sorry (and, probably, the optimiser will prove it's a constant and remove it).
        assert!(usize::from(InstIdx::max()) > m.insts_len());
        assert!(usize::from(u8::MAX) > Inst::COUNT);
        Self {
            heads: [InstIdx::max(); Inst::COUNT],
            predecessors: vec![InstIdx::max(); m.insts_len()],
        }
    }

    /// As a trace is being optimised, this method should be called with each new instruction
    /// generated.
    pub(super) fn push(&mut self, iidx: InstIdx, inst: Inst) {
        let dim_off = inst.discriminant();
        if self.heads[dim_off] != InstIdx::max() {
            self.predecessors[usize::from(iidx)] = self.heads[dim_off];
        }
        self.heads[dim_off] = iidx;
    }

    /// Successively generate the backwardly reachable instructions of the same kind as `inst`.
    pub(super) fn rev_iter<'a>(&'a self, m: &'a Module, inst: Inst) -> InstLLRevIterator<'a> {
        InstLLRevIterator {
            m,
            instll: self,
            next: self.heads[inst.discriminant()],
        }
    }
}

pub(super) struct InstLLRevIterator<'a> {
    m: &'a Module,
    instll: &'a InstLinkedList,
    next: InstIdx,
}

impl Iterator for InstLLRevIterator<'_> {
    type Item = (InstIdx, Inst);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next != InstIdx::max() {
            let x = self.next;
            self.next = self.instll.predecessors[usize::from(x)];
            match self.m.inst_nocopy(x) {
                None | Some(Inst::Const(_)) | Some(Inst::Tombstone) => (),
                Some(y) => return Some((x, y)),
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::jitc_yk::jit_ir::InstDiscriminants;

    #[test]
    fn recents_predecessors() {
        let m = Module::from_str(
            "
            entry:
              %0: i8 = param reg
              %1: ptr = param reg
              %2: i8 = add %0, %0
              %3: i8 = add %0, %0
              %4: i8 = load %1
              %5: i8 = add %0, %0
            ",
        );
        let mut instll = InstLinkedList::new(&m);
        for (iidx, inst) in m.iter_skipping_insts() {
            instll.push(iidx, inst);
        }
        assert_eq!(
            instll
                .heads
                .iter()
                .filter(|x| **x == InstIdx::max())
                .count(),
            Inst::COUNT - 3
        );
        assert_eq!(
            instll.heads[InstDiscriminants::Param as usize],
            InstIdx::unchecked_from(1)
        );
        assert_eq!(
            instll.heads[InstDiscriminants::BinOp as usize],
            InstIdx::unchecked_from(5)
        );
        assert_eq!(
            instll.heads[InstDiscriminants::Load as usize],
            InstIdx::unchecked_from(4)
        );
        assert_eq!(
            instll.predecessors,
            vec![
                InstIdx::max(),
                InstIdx::unchecked_from(0),
                InstIdx::max(),
                InstIdx::unchecked_from(2),
                InstIdx::max(),
                InstIdx::unchecked_from(3)
            ]
        );
        assert_eq!(
            instll
                .rev_iter(&m, m.inst(InstIdx::unchecked_from(1)))
                .map(|(iidx, _)| usize::from(iidx))
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
        assert_eq!(
            instll
                .rev_iter(&m, m.inst(InstIdx::unchecked_from(5)))
                .map(|(iidx, _)| usize::from(iidx))
                .collect::<Vec<_>>(),
            vec![5, 3, 2]
        );
        assert_eq!(
            instll
                .rev_iter(&m, m.inst(InstIdx::unchecked_from(4)))
                .map(|(iidx, _)| usize::from(iidx))
                .collect::<Vec<_>>(),
            vec![4]
        );
    }
}
