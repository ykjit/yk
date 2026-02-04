//! Redundant load/store optimisation.
//!
//! Loads and stores are often unnecessary and can be removed. For example if we have this input
//! trace:
//!
//! ```text
//! %0: ptr = arg [reg]
//! %1: i8 = load %0
//! store %1, %0
//! %2: i8 = load %0
//! ```
//!
//! both the store and the second load are redundant (assuming they are non-`volatile`): we know
//! that the byte stored at %0's pointer has the value in %1 and it cannot be changed.
//!
//! This pass is currently very naive in two main ways:
//!   * Its approach to alias analysis is to only trust pointers derived from the same base: so
//!     `ptradd %0, 4` and `ptradd %0, 8` are distinct pointers but it can never prove that two
//!     pointers `%0` and `%1` that derive from different bases do not alias.
//!   * It has no notion of "effect" other than "things like calls are barriers". As soon as it
//!     encounters a barrier, all of its knowledge of heap values is destroyed.

use crate::compile::j2::{
    effects::Effects,
    hir::*,
    opt::{
        BlockLikeT, EquivIIdxT, ModLikeT,
        fullopt::{CommitInstOpt, OptOutcome, PassOpt, PassT},
    },
};
use std::collections::HashMap;

/// Load/Store elimination
pub(super) struct LoadStore {
    /// A map telling us what value we know is currently stored at a given [Address].
    hv: HashMap<Address, InstIdx>,
}

impl LoadStore {
    pub(super) fn new() -> Self {
        Self { hv: HashMap::new() }
    }
}

impl PassT for LoadStore {
    fn feed(&mut self, opt: &mut PassOpt, inst: Inst) -> OptOutcome {
        match inst {
            Inst::Load(Load {
                tyidx,
                ptr,
                is_volatile: false,
            }) => {
                let addr = Address::from(opt, ptr);
                if let Some(hv_iidx) = self.hv.get(&addr) {
                    let inst_ty = opt.ty(tyidx);
                    let hv_iidx = opt.equiv_iidx(*hv_iidx);
                    let hv_tyidx = opt.inst(hv_iidx).tyidx(opt);
                    let hv_ty = opt.ty(hv_tyidx);
                    // We currently only allow the same number of bytes to lead to load
                    // elimination. We could relax this to allow <= the number of known bits.
                    if inst_ty.bitw() == hv_ty.bitw() {
                        // If type punning has occurred, we can either do a type conversion
                        // ourselves, or give up.
                        if hv_ty == inst_ty {
                            // No type punning has occurred: the easy case!
                            return OptOutcome::Equiv(hv_iidx);
                        }
                    }
                }
                OptOutcome::Rewritten(inst)
            }
            Inst::Store(Store {
                ptr,
                val,
                is_volatile: false,
            }) => {
                let val = opt.equiv_iidx(val);
                let addr = Address::from(opt, ptr);
                if let Some(iidx) = self.hv.get(&addr) {
                    let iidx = opt.equiv_iidx(*iidx);
                    if val == iidx {
                        // We currently only allow the same number of bits to lead to load
                        // elimination. We could relax this to allow <= the number of known bits.
                        if opt.inst_bitw(opt, val) == opt.inst_bitw(opt, iidx) {
                            return OptOutcome::NotNeeded;
                        }
                    }
                }
                OptOutcome::Rewritten(inst)
            }
            _ => OptOutcome::Rewritten(inst),
        }
    }

    fn inst_committed(&mut self, opt: &CommitInstOpt, iidx: InstIdx, inst: &Inst) {
        match inst {
            Inst::Load(Load {
                tyidx: _,
                ptr,
                is_volatile: _,
            }) => {
                self.hv.insert(Address::from(opt, *ptr), iidx);
            }
            Inst::Store(Store {
                ptr,
                val,
                is_volatile: _,
            }) => {
                let addr = Address::from(opt, *ptr);
                let val = opt.equiv_iidx(*val);
                // We are ultra conservative here: we only say "these don't overlap" if two stores
                // ultimately reference the same SSA variable with pointer adds. In other words, if
                // we're writing 8 bytes and we're storing to `%3 + 8` and `%3 + 24` we can be
                // entirely sure the stores don't overlap: in any other situation, we assume
                // overlap is possible.
                match addr {
                    Address::PtrOff(ptr, off) => {
                        let ptr = opt.equiv_iidx(ptr);
                        let off = isize::try_from(off).unwrap();
                        let bytew = isize::try_from(opt.inst_bitw(opt, val).div_ceil(8)).unwrap();
                        self.hv.retain(|hv_addr, hv_val| match hv_addr {
                            Address::PtrOff(hv_ptr, hv_off) => {
                                let hv_ptr = opt.equiv_iidx(*hv_ptr);
                                let hv_off = isize::try_from(*hv_off).unwrap();
                                let hv_val = opt.equiv_iidx(*hv_val);
                                let hv_bytew =
                                    isize::try_from(opt.inst_bitw(opt, hv_val).div_ceil(8))
                                        .unwrap();
                                ptr == hv_ptr && (off + bytew <= hv_off || hv_off + hv_bytew <= off)
                            }
                            Address::Const(_) => false,
                        });
                    }
                    Address::Const(addr) => {
                        self.hv.retain(|hv_addr, hv_val| match hv_addr {
                            Address::PtrOff(_, _) => false,
                            Address::Const(hv_addr) => {
                                let bytew =
                                    usize::try_from(opt.inst_bitw(opt, val).div_ceil(8)).unwrap();
                                let hv_val = opt.equiv_iidx(*hv_val);
                                let hv_bytew =
                                    usize::try_from(opt.inst_bitw(opt, hv_val).div_ceil(8))
                                        .unwrap();
                                addr + bytew <= *hv_addr || hv_addr + hv_bytew <= addr
                            }
                        });
                    }
                }
                self.hv.insert(addr, val);
            }
            _ => {
                if inst
                    .read_write_effects()
                    .interferes(Effects::none().add_heap().add_volatile())
                {
                    self.hv.clear();
                }
            }
        }
    }

    fn equiv_committed(&mut self, _equiv1: InstIdx, _equiv2: InstIdx) {}
}

/// An abstract "address" representing a location in RAM.
#[derive(Debug, Eq, Hash, PartialEq)]
enum Address {
    PtrOff(InstIdx, i32),
    Const(usize),
}

impl Address {
    /// Create a new `address` for the instruction at `ptr`. Note: `ptr` does not need to have
    /// been `equiv_iidx`ed.
    fn from<T: BlockLikeT + EquivIIdxT>(opt: &T, mut ptr: InstIdx) -> Self {
        // We now chase the instruction at `ptr` backwards if it's a chain of `PtrAdd`s. So
        // if we have:
        //
        // ```
        // %0: ptr = arg [reg]
        // %1: ptradd %0, 8
        // %2: ptradd %1, 4
        // ```
        //
        // and we call `Address::from(%2)` then `Address::PtrOff(%0, 12)` is returned.

        let mut cum_off: i32 = 0; // Cumulative offset over the chain of `PtrAdd`s.
        ptr = opt.equiv_iidx(ptr);
        while let Inst::PtrAdd(PtrAdd {
            ptr: child_ptr,
            off,
            nusw,
            nuw,
            in_bounds,
        }) = opt.inst(ptr)
        {
            // We don't support nusw or nuw yet
            assert!(!nusw && !nuw && !in_bounds);
            cum_off = cum_off.checked_add(*off).unwrap();
            ptr = opt.equiv_iidx(*child_ptr);
        }
        if let Inst::Const(c) = opt.inst(ptr) {
            assert_eq!(cum_off, 0);
            let Const {
                tyidx: _,
                kind: ConstKind::Ptr(addr),
            } = c
            else {
                panic!()
            };
            Address::Const(*addr)
        } else {
            Address::PtrOff(ptr, cum_off)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::{fullopt::test::opt_and_test, strength_fold::StrengthFold};
    use std::{cell::RefCell, rc::Rc};

    fn test_ls(mod_s: &str, ptn: &str) {
        let ls = Rc::new(RefCell::new(LoadStore::new()));
        let strength_fold = Rc::new(RefCell::new(StrengthFold::new()));
        opt_and_test(
            mod_s,
            |opt, mut inst| {
                if let Inst::Guard(_) = inst {
                    strength_fold.borrow_mut().feed(opt, inst)
                } else {
                    inst.canonicalise(opt);
                    ls.borrow_mut().feed(opt, inst)
                }
            },
            |opt, iidx, inst| ls.borrow_mut().inst_committed(opt, iidx, inst),
            |_, _| (),
            ptn,
        );
    }

    #[test]
    fn basic() {
        // Optimise away a store following a load
        test_ls(
            "
          %0: ptr = 0x1234
          %1: i8 = load %0
          store %1, %0
          blackbox %1
        ",
            "
          %0: ptr = 0x1234
          %1: i8 = load %0
          blackbox %1
        ",
        );

        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = load %0
          store %1, %0
          blackbox %1
        ",
            "
          %0: ptr = arg
          %1: i8 = load %0
          blackbox %1
        ",
        );

        // Optimise away a load following a store
        test_ls(
            "
          %0: i8 = arg [reg]
          %1: ptr = 0x1234
          store %0, %1
          %3: i8 = load %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: ptr = 0x1234
          store %0, %1
          blackbox %0
        ",
        );

        test_ls(
            "
          %0: i8 = arg [reg]
          %1: ptr = arg [reg]
          store %0, %1
          %3: i8 = load %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: ptr = arg
          store %0, %1
          blackbox %0
        ",
        );
    }

    #[test]
    fn load_store_equiv() {
        // Test that deep instruction equivalence works
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: ptr = arg [reg]
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          %4: i8 = load %0
          %5: i8 = load %1
          blackbox %4
          blackbox %5
          term [%0, %1]
        ",
            "
          %0: ptr = arg
          %1: ptr = arg
          %2: i1 = icmp eq %0, %1
          guard true, %2, []
          %4: i8 = load %0
          blackbox %4
          blackbox %4
          term [%0, %0]
          ...
        ",
        );
    }

    #[test]
    fn overlaps() {
        // Test that overlapping loads/stores aren't optimised away.

        // const
        test_ls(
            "
          %0: ptr = 0x1234
          %1: i8 = load %0
          %2: i16 = load %0
          blackbox %1
          blackbox %2
        ",
            "
          %0: ptr = 0x1234
          %1: i8 = load %0
          %2: i16 = load %0
          blackbox %1
          blackbox %2
        ",
        );

        test_ls(
            "
          %0: i8 = arg [reg]
          %1: ptr = 0x1234
          store %0, %1
          %3: i16 = load %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: ptr = 0x1234
          store %0, %1
          %3: i16 = load %1
          blackbox %3
        ",
        );

        // ptradd
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = load %0
          %2: i16 = load %0
          blackbox %1
          blackbox %2
        ",
            "
          %0: ptr = arg
          %1: i8 = load %0
          %2: i16 = load %0
          blackbox %1
          blackbox %2
        ",
        );

        test_ls(
            "
          %0: i8 = arg [reg]
          %1: ptr = arg [reg]
          store %0, %1
          %3: i16 = load %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: ptr = arg
          store %0, %1
          %3: i16 = load %1
          blackbox %3
        ",
        );
    }

    #[test]
    fn overlap_edges() {
        // Test that non-overlapping stores don't invalidate previous loads.
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = load %0
          %3: ptr = ptradd %0, -8
          store %1, %3
          %5: i8 = load %0
          blackbox %2
          blackbox %5
          term [%0, %1]
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          %2: i8 = load %0
          %3: ptr = ptradd %0, -8
          store %1, %3
          blackbox %2
          blackbox %2
          term [%0, %1]
        ",
        );
    }

    #[test]
    fn ptradd_collapse() {
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          store %1, %0
          %3: ptr = ptradd %0, 4
          %4: ptr = ptradd %3, -4
          %5: i8 = load %4
          blackbox %5
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          store %1, %0
          %3: ptr = ptradd %0, 4
          %4: ptr = ptradd %3, -4
          blackbox %1
        ",
        );
    }

    #[test]
    fn barriers() {
        // Test that the things that should be barriers really do act as barriers.

        // Calls
        test_ls(
            "
          extern f()

          %0: ptr = arg [reg]
          %1: i8 = load %0
          %2: ptr = 0x1234
          call f %2()
          store %1, %0
        ",
            "
          %0: ptr = arg
          %1: i8 = load %0
          %2: ptr = 0x1234
          call %2()
          store %1, %0
        ",
        );

        // memcpy
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: ptr = arg [reg]
          %2: i64 = arg [reg]
          %3: i8 = load %0
          memcpy %0, %1, %2, true
          %5: i8 = load %0
        ",
            "
          %0: ptr = arg
          %1: ptr = arg
          %2: i64 = arg
          %3: i8 = load %0
          memcpy %0, %1, %2, true
          %5: i8 = load %0
        ",
        );

        // memset
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          %2: i32 = arg [reg]
          %3: i8 = load %0
          memset %0, %1, %2, true
          %5: i8 = load %0
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          %2: i32 = arg
          %3: i8 = load %0
          memset %0, %1, %2, true
          %5: i8 = load %0
        ",
        );
    }
}
