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
use index_vec::IndexVec;
use std::{collections::HashMap, mem};

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
                        match (hv_ty, inst_ty) {
                            (x, y) if x == y => {
                                // No type punning has occurred: the easy case!
                                return OptOutcome::Equiv(hv_iidx);
                            }
                            // The cases below are for type punning: we have to translate these
                            // into the appropriate HIR instructions.
                            (Ty::Int(64), Ty::Double) | (Ty::Double, Ty::Int(64)) => {
                                return OptOutcome::Rewritten(
                                    BitCast {
                                        tyidx,
                                        val: hv_iidx,
                                    }
                                    .into(),
                                );
                            }
                            (Ty::Int(_), Ty::Ptr(0)) => {
                                return OptOutcome::Rewritten(
                                    IntToPtr {
                                        tyidx,
                                        val: hv_iidx,
                                    }
                                    .into(),
                                );
                            }
                            (Ty::Ptr(0), Ty::Int(_)) => {
                                return OptOutcome::Rewritten(
                                    PtrToInt {
                                        tyidx,
                                        val: hv_iidx,
                                    }
                                    .into(),
                                );
                            }
                            (x, y) => todo!("{x:?} {y:?}"),
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

    fn equiv_committed(&mut self, equiv1: InstIdx, equiv2: InstIdx) {
        // FIXME: This is of a hack until `prepare_for_peel` can properly determine equivalences.
        // At that point, there's no need for this method to do anything.
        for (_hv_addr, hv_val) in self.hv.iter_mut() {
            if *hv_val == equiv1 {
                *hv_val = equiv2;
            }
        }
    }

    fn prepare_for_peel(
        &mut self,
        opt: &mut PassOpt,
        entry: &Block,
        map: &IndexVec<InstIdx, InstIdx>,
    ) {
        let mut new_hv = HashMap::new();

        for (hv_addr, hv_val) in mem::take(&mut self.hv) {
            let new_addr = match hv_addr {
                Address::PtrOff(iidx, off) => {
                    if map[iidx] == InstIdx::MAX {
                        continue;
                    }
                    Address::PtrOff(map[iidx], off)
                }
                Address::Const(_) => hv_addr,
            };
            if map[hv_val] != InstIdx::MAX {
                new_hv.insert(new_addr, opt.equiv_iidx(map[hv_val]));
            } else if let Inst::Const(x) = entry.inst(hv_val) {
                new_hv.insert(new_addr, opt.push_pre_inst(Inst::Const(x.to_owned())));
            }
        }
        self.hv = new_hv;
    }
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
    use crate::compile::j2::opt::{
        fullopt::test::{full_opt_test, user_defined_opt_test},
        strength_fold::StrengthFold,
    };
    use std::{cell::RefCell, rc::Rc};

    fn test_ls(mod_s: &str, ptn: &str) {
        let ls = Rc::new(RefCell::new(LoadStore::new()));
        let strength_fold = Rc::new(RefCell::new(StrengthFold::new()));
        user_defined_opt_test(
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
    fn load_punning() {
        // i64 -> double
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i64 = load %0
          %2: double = load %0
          blackbox %1
          blackbox %2
        ",
            "
          %0: ptr = arg
          %1: i64 = load %0
          %2: double = bitcast %1
          blackbox %1
          blackbox %2
        ",
        );

        #[cfg(not(target_pointer_width = "64"))]
        todo!();

        #[cfg(target_pointer_width = "64")]
        {
            // i64 -> ptr
            test_ls(
                "
          %0: ptr = arg [reg]
          %1: i64 = load %0
          %2: ptr = load %0
          blackbox %1
          blackbox %2
        ",
                "
          %0: ptr = arg
          %1: i64 = load %0
          %2: ptr = inttoptr %1
          blackbox %1
          blackbox %2
        ",
            );

            // ptr -> i64
            test_ls(
                "
          %0: ptr = arg [reg]
          %1: ptr = load %0
          %2: i64 = load %0
          blackbox %1
          blackbox %2
        ",
                "
          %0: ptr = arg
          %1: ptr = load %0
          %2: i64 = ptrtoint %1
          blackbox %1
          blackbox %2
        ",
            );
        }
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
          memcpy %0, %1, %2, false
          %5: i8 = load %0
        ",
            "
          %0: ptr = arg
          %1: ptr = arg
          %2: i64 = arg
          %3: i8 = load %0
          memcpy %0, %1, %2, false
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
          memset %0, %1, %2, false
          %5: i8 = load %0
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          %2: i32 = arg
          %3: i8 = load %0
          memset %0, %1, %2, false
          %5: i8 = load %0
        ",
        );
    }

    #[test]
    fn peeling() {
        // Optimise away a load of a known constant across a peel.
        full_opt_test(
            r#"
          %0: ptr = arg [reg("GPR0", undefined)]
          %1: i8 = load %0
          %2: i8 = 2
          %3: i1 = icmp eq %1, %2
          guard true, %3, []
          blackbox %1
          term [%0]
        "#,
            "
          %0: ptr = arg
          %1: i8 = load %0
          %2: i8 = 2
          %3: i1 = icmp eq %1, %2
          guard true, %3, []
          blackbox %2
          term [%0]
          ; peel
          %0: ptr = arg
          %2: i8 = 2
          blackbox %2
          term [%0]
        ",
        );

        // Optimise away a store of a known constant across a peel.
        full_opt_test(
            r#"
          %0: ptr = arg [reg("GPR0", undefined)]
          %1: i8 = 2
          store %1, %0
          term [%0]
        "#,
            "
          %0: ptr = arg
          %1: i8 = 2
          store %1, %0
          term [%0]
          ; peel
          %0: ptr = arg
          term [%0]
        ",
        );

        // FIXME: This test shouldn't pass! We _should_ be capable of passing the equivalence of %2
        // and %1 into the peel, but we need to extend the term_vars to do so.
        full_opt_test(
            r#"
          %0: ptr = arg [reg("GPR0", undefined)]
          %1: i8 = arg [reg("GPR1", undefined)]
          %2: i8 = load %0
          %3: i1 = icmp eq %2, %1
          guard true, %3, []
          blackbox %1
          %6: i8 = 2
          term [%0, %6]
        "#,
            "
          %0: ptr = arg
          %1: i8 = arg
          %2: i8 = load %0
          %3: i1 = icmp eq %2, %1
          guard true, %3, []
          blackbox %2
          %6: i8 = 2
          term [%0, %6]
          ; peel
          %0: ptr = arg
          %1: i8 = 2
          %2: i8 = load %0
          %3: i1 = icmp eq %2, %1
          guard true, %3, []
          blackbox %1
          term [%0, %1]
        ",
        );
    }

    #[test]
    fn volatiles() {
        // Volatile loads fill the cache but aren't removed
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = load volatile %0
          %2: i8 = load %0
          %3: i8 = load volatile %0
          term [%0]
        ",
            "
          %0: ptr = arg
          %1: i8 = load volatile %0
          %2: i8 = load volatile %0
          term [%0]
        ",
        );

        // Volatile stores fill the cache but aren't removed
        test_ls(
            "
          %0: ptr = arg [reg]
          %1: i8 = arg [reg]
          store volatile %1, %0
          store %1, %0
          store volatile %1, %0
          term [%0, %1]
        ",
            "
          %0: ptr = arg
          %1: i8 = arg
          store volatile %1, %0
          store volatile %1, %0
          term [%0, %1]
        ",
        );
    }
}
