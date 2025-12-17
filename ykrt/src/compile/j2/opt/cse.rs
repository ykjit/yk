//! Common subexpression elimination.
//!
//! ## Efficiency
//!
//! CSE is simple but if implemented naively it has O(n^2) costs. We use two tricks to reduce the
//! cost.
//!
//! First, we take a lesson from LuaJIT and maintain a linked list of instruction kinds: if we're
//! checking CSE for, say [PtrAdd]s, that means we only check `PtrAdd`s (and not, say, [And] etc).
//!
//! Second, we can use an instruction's operand indices to work out if we've gone implausibly
//! far back. For example, if we're checking `add %102, %105` for CSE, there is no chance of a
//! match at an [InstIdx] below 102 and we can stop checking.
//!
//! This allows one to iterate backwards over a trace starting at instruction X and efficiently
//! view all previous instructions of the same kind as X, typically bailing out early if there is
//! no match.

use crate::compile::j2::{
    hir::{Inst, InstDiscriminants, InstIdx, InstT},
    opt::OptT,
};
use strum::EnumCount;

/// Common supexpression elimination.
///
/// Internally this uses a linked list for each [Inst] encoded in `heads` and `predecessors`.
pub(super) struct CSE {
    /// The head of the linked list with one entry for each [Inst] discriminant. This points to the
    /// most recent entry in [Self::predecessors]. `InstIDX::MAX_INDEX` is used to represent "no
    /// previous entry".
    heads: [InstIdx; Inst::COUNT],
    /// The linked list with one entry for each instruction in a trace. `InstIDX::MAX_INDEX` is
    /// used to represent "no previous entry".
    predecessors: Vec<InstIdx>,
}

impl CSE {
    /// Create an empty CSE object. As new instructions are committed to the eventual [Module],
    /// they must also be `push`ed here so that CSE can be in-sync with the [Module].
    pub(super) fn new() -> Self {
        // Because we use `HEAD_UINT::MAX` as our "we haven't seen anything here yet" value, we
        // have to make sure we won't overlap with a genuine value.
        const {
            assert!(InstIdx::MAX_INDEX > Inst::COUNT);
        }
        Self {
            heads: [InstIdx::from_usize(InstIdx::MAX_INDEX); Inst::COUNT],
            predecessors: Vec::new(),
        }
    }

    pub(super) fn is_equiv(&self, opt: &dyn OptT, inst: &Inst) -> Option<InstIdx> {
        // FIXME: This is a hack for "does this instruction have side effects".
        if let Inst::Arg(_)
        | Inst::Call(_)
        | Inst::Const(_)
        | Inst::Exit(_)
        | Inst::Guard(_)
        | Inst::Load(_)
        | Inst::MemCpy(_)
        | Inst::MemSet(_)
        | Inst::Return(_)
        | Inst::Store(_) = inst
        {
            return None;
        }

        #[cfg(test)]
        {
            if let Inst::BlackBox(_) = inst {
                return None;
            }
        }

        // FIXME: This is an inefficient way of working out what the maximum operand reference is,
        // but `for_each_iidx` needs to be changed for that to be possible.
        let max_ref = std::rc::Rc::new(std::cell::RefCell::new(InstIdx::from_usize(0)));
        inst.for_each_iidx(|x| {
            if x > *max_ref.borrow() {
                *max_ref.borrow_mut() = x;
            }
        });
        let max_ref = *max_ref.borrow();
        let mut cur = self.heads[InstDiscriminants::from(inst) as usize];
        while cur != InstIdx::MAX_INDEX && cur >= max_ref {
            let equiv = opt.equiv_iidx(cur);
            if equiv >= max_ref && opt.inst(equiv).cse_eq(opt, inst) {
                return Some(equiv);
            }
            cur = self.predecessors[usize::from(cur)];
        }
        None
    }

    /// As a [Module] is being built-up, this method must be called with each new instruction
    /// generated.
    pub(super) fn push(&mut self, inst: &Inst) {
        let dim_off = InstDiscriminants::from(inst) as usize;
        let prev = self.heads[dim_off];
        self.predecessors.push(prev);
        self.heads[dim_off] = InstIdx::from_usize(self.predecessors.len() - 1);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::{
        opt::{OptOutcome, test::opt_and_test},
        strength_fold::strength_fold,
    };
    use std::{cell::RefCell, rc::Rc};

    fn test_cse(mod_s: &str, ptn: &str) {
        let cse = Rc::new(RefCell::new(CSE::new()));
        opt_and_test(
            mod_s,
            |opt, mut inst| {
                if let Inst::Guard(_) = inst {
                    match strength_fold(opt, inst) {
                        OptOutcome::NotNeeded => OptOutcome::NotNeeded,
                        OptOutcome::Rewritten(inst) => {
                            let mut cse = cse.borrow_mut();
                            cse.push(&inst);
                            OptOutcome::Rewritten(inst)
                        }
                        OptOutcome::Equiv(iidx) => OptOutcome::Equiv(iidx),
                    }
                } else {
                    inst.canonicalise(opt);
                    let mut cse = cse.borrow_mut();
                    if let Some(iidx) = cse.is_equiv(opt, &inst) {
                        OptOutcome::Equiv(iidx)
                    } else {
                        cse.push(&inst);
                        OptOutcome::Rewritten(inst)
                    }
                }
            },
            ptn,
        );
    }

    #[test]
    fn cse() {
        test_cse(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = add %0, %1
          %3: i8 = add %0, %1
          blackbox %2
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = add %0, %1
          blackbox %2
          blackbox %2
        ",
        );

        // Test that CSE isn't impacted by intermediate instructions.
        test_cse(
            "
          extern abort()

          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = add %0, %1
          %3: ptr = @abort
          call abort %3()
          %5: i8 = add %0, %1
          blackbox %2
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = add %0, %1
          %3: ptr = ...
          call %3()
          blackbox %2
          blackbox %2
        ",
        );
    }

    #[test]
    fn cse_equiv() {
        // Test that deep instruction equivalence works
        test_cse(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          %4: i8 = add %0, %0
          %5: i8 = add %1, %1
          blackbox %4
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          %4: i8 = add %0, %0
          blackbox %4
          blackbox %4
        ",
        );

        test_cse(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = add %0, %0
          %3: i1 = icmp eq %0, %1
          guard true, %3, []
          %5: i8 = add %1, %1
          blackbox %2
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = add %0, %0
          %3: i1 = icmp eq %0, %1
          guard true, %3, []
          blackbox %2
          blackbox %2
        ",
        );
    }

    #[test]
    fn cse_never_equiv() {
        // For each instruction kind that CSE should never be able to remove, a test should be
        // added here.

        // arg
        test_cse(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
        ",
        );

        // blackbox
        test_cse(
            "
          %0: i8 = arg [reg]
          blackbox %0
          blackbox %0
        ",
            "
          %0: i8 = arg
          blackbox %0
          blackbox %0
        ",
        );

        // call
        test_cse(
            "
          extern abort()

          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: ptr = @abort
          call abort %2()
          call abort %2()
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: ptr = ...
          call %2()
          call %2()
        ",
        );

        // guard
        test_cse(
            "
          %0: i1 = arg [reg]
          guard true, %0, []
          guard true, %0, []
        ",
            "
          %0: i1 = arg
          guard true, %0, []
          guard true, %0, []
        ",
        );

        // load
        test_cse(
            "
          %0: ptr = arg [reg]
          %1: i8 = load %0
          %2: i8 = load %0
        ",
            "
          %0: ptr = arg
          %1: i8 = load %0
          %2: i8 = load %0
        ",
        );

        // memcpy
        test_cse(
            "
          %0: ptr = arg [reg]
          %1: ptr = arg [reg]
          %2: i64 = arg [reg]
          memcpy %0, %1, %2, true
          memcpy %0, %1, %2, true
        ",
            "
          %0: ptr = arg
          %1: ptr = arg
          %2: i64 = arg
          memcpy %0, %1, %2, true
          memcpy %0, %1, %2, true
        ",
        );

        // memset
        test_cse(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          %2: i32 = arg [reg]
          memset %0, %1, %2, true
          memset %0, %1, %2, true
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          %2: i32 = arg
          memset %0, %1, %2, true
          memset %0, %1, %2, true
        ",
        );

        // store
        test_cse(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          store %1, %0
          store %1, %0
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          store %1, %0
          store %1, %0
        ",
        );
    }
}
