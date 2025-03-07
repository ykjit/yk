//! Build up knowledge about values on the heap. As we optimise and analyse a trace, we build up
//! knowledge about the possible values that can be stored at different addresses. This module
//! provides a way of keeping track of what values we know about at a given point in the trace.
//!
//! Broadly speaking, loads add new information; stores tend to remove most old information and add
//! new information; and barriers remove all information.
use super::super::jit_ir::{Const, Inst, InstIdx, Module, Operand};
use std::collections::HashMap;

/// An abstract "address" representing a location in RAM.
///
/// Users of this module should not need to use this `enum`'s variants: if you find yourself
/// needing to do so, then consider whether a refactoring of this code might be in order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum Address {
    /// This address is derived from a pointer stored in `InstIdx` + the constant offset `i32`
    /// values in bytes.
    PtrPlusOff(InstIdx, i32),
    /// This address is a constant.
    #[allow(unused)]
    Const(usize),
}

impl Address {
    /// Create an [Address] from an [Operand].
    pub(super) fn from_operand(m: &Module, op: Operand) -> Address {
        match op {
            Operand::Var(mut iidx) => {
                // We canonicalise pointers as "instruction + off" so that we can cut out any
                // intermediate `PtrAdd`s.
                let mut off = 0;
                while let Inst::PtrAdd(pa_inst) = m.inst_nocopy(iidx).unwrap() {
                    match pa_inst.ptr(m) {
                        Operand::Var(ptr_iidx) => {
                            off += pa_inst.off();
                            iidx = ptr_iidx;
                        }
                        Operand::Const(_) => todo!(),
                    }
                }
                Address::PtrPlusOff(iidx, off)
            }
            Operand::Const(cidx) => {
                let Const::Ptr(v) = m.const_(cidx) else {
                    panic!();
                };
                Address::Const(*v)
            }
        }
    }
}

/// The currently known values on the heap.
///
/// This must be used as part of a linear scan over a trace: as you move from one instruction to
/// the next, knowledge will be gained (e.g. because of a new load) and lost (e.g. because a store
/// invalidates some or all of our previous knowledge).
pub(super) struct HeapValues {
    /// The heap values we currently know about.
    hv: HashMap<Address, Operand>,
}

impl HeapValues {
    pub(super) fn new() -> Self {
        HeapValues { hv: HashMap::new() }
    }

    /// What is the currently known value at `addr` of `bytesize` bytes? Returns `None` if no value
    /// of that size is known at that address.
    pub(super) fn get(&self, m: &Module, addr: Address, bytesize: usize) -> Option<Operand> {
        match self.hv.get(&addr) {
            Some(x) if x.byte_size(m) == bytesize => Some(x.clone()),
            _ => None,
        }
    }

    /// Record the value `v` as known to be at `addr` as a result of a load. Note: `v`'s bytesize
    /// *must* match the number of bytes stored at `addr`.
    pub(super) fn load(&mut self, _m: &Module, addr: Address, v: Operand) {
        // We don't need to invalidate anything for loads: we can safely have aliases.
        self.hv.insert(addr, v);
    }

    /// Record the value `v` as known to be at `addr` as a result of a store. Note: `v`'s bytesize
    /// *must* match the number of bytes stored at `addr`.
    pub(super) fn store(&mut self, m: &Module, addr: Address, v: Operand) {
        // We now need to perform alias analysis to see if this new value invalidates some or all
        // of our previous knowledge.
        match addr {
            Address::PtrPlusOff(iidx, off) => {
                let off = isize::try_from(off).unwrap();
                let op_bytesize = isize::try_from(v.byte_size(m)).unwrap();
                // We are ultra conservative here: we only say "these don't overlap" if two stores
                // ultimately reference the same SSA variable with pointer adds. In other words, if
                // we're writing 8 bytes and we're storing to `%3 + 8` and `%3 + 24` we can be
                // entirely sure the stores don't overlap: in any other situation, we assume
                // overlap is possible. This can be relaxed in the future.
                self.hv.retain(|hv_addr, _| match hv_addr {
                    Address::PtrPlusOff(hv_iidx, hv_off) => {
                        let hv_off = isize::try_from(*hv_off).unwrap();
                        let hv_bytesize =
                            isize::try_from(m.inst_nocopy(*hv_iidx).unwrap().def_byte_size(m))
                                .unwrap();
                        iidx == *hv_iidx
                            && (off + op_bytesize <= hv_off || hv_off + hv_bytesize <= off)
                    }
                    Address::Const(_) => false,
                });
                self.hv.insert(addr, v);
            }
            Address::Const(_) => todo!(),
        }
    }

    /// Record a barrier instruction as having been encountered. This will invalidate all of our
    /// existing heap knowledge.
    pub(super) fn barrier(&mut self) {
        self.hv.clear();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::jitc_yk::{
        arbbitint::ArbBitInt,
        jit_ir::{Const, Ty},
    };

    #[test]
    fn basic() {
        // We only need to have the `ptr_add`s in the module for this test.
        let mut m = Module::from_str(
            "
          entry:
            %0: ptr = param reg
            %1: ptr = ptr_add %0, 1
            %2: ptr = ptr_add %1, -1
            %3: ptr = ptr_add %0, 8
            %4: ptr = load %3
        ",
        );
        let mut hv = HeapValues::new();

        // Add a single load
        let addr0 = Address::from_operand(&m, Operand::Var(InstIdx::unchecked_from(0)));
        assert!(hv.get(&m, addr0.clone(), 1).is_none());
        let cidx0 = m
            .insert_const(Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 0)))
            .unwrap();
        hv.load(&m, addr0.clone(), Operand::Const(cidx0));
        assert_eq!(hv.hv.len(), 1);
        assert_eq!(hv.get(&m, addr0.clone(), 1), Some(Operand::Const(cidx0)));

        // Add a non-overlapping load
        let addr1 = Address::from_operand(&m, Operand::Var(InstIdx::unchecked_from(1)));
        let cidx1 = m
            .insert_const(Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 1)))
            .unwrap();
        hv.load(&m, addr1.clone(), Operand::Const(cidx1));
        assert_eq!(hv.hv.len(), 2);
        assert_eq!(hv.get(&m, addr0.clone(), 1), Some(Operand::Const(cidx0)));
        assert_eq!(hv.get(&m, addr1.clone(), 1), Some(Operand::Const(cidx1)));

        // Check that ptr_adds are canonicalised.
        let addr2 = Address::from_operand(&m, Operand::Var(InstIdx::unchecked_from(2)));
        assert_eq!(hv.get(&m, addr2.clone(), 1), Some(Operand::Const(cidx0)));
        assert!(hv.get(&m, addr2.clone(), 2).is_none());

        // Add a store that replaces our knowledge of the second load but preserves the first.
        let cidx2 = m
            .insert_const(Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 2)))
            .unwrap();
        hv.store(&m, addr2.clone(), Operand::Const(cidx2));
        assert_eq!(hv.hv.len(), 2);
        assert_eq!(hv.get(&m, addr0.clone(), 1), Some(Operand::Const(cidx2)));
        assert_eq!(hv.get(&m, addr1.clone(), 1), Some(Operand::Const(cidx1)));

        // Add an overlapping i64 store which should remove information about both preceding loads.
        let int64_tyidx = m.insert_ty(Ty::Integer(64)).unwrap();
        let cidx3 = m
            .insert_const(Const::Int(int64_tyidx, ArbBitInt::from_u64(64, 3)))
            .unwrap();
        hv.store(&m, addr2.clone(), Operand::Const(cidx3));
        assert_eq!(hv.hv.len(), 1);
        assert_eq!(hv.get(&m, addr0.clone(), 8), Some(Operand::Const(cidx3)));
        assert!(hv.get(&m, addr0.clone(), 1).is_none());
        assert!(hv.get(&m, addr1.clone(), 1).is_none());

        // Add an overlapping i8 store which should remove information about the i64 load.
        let cidx4 = m
            .insert_const(Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 4)))
            .unwrap();
        hv.store(&m, addr1.clone(), Operand::Const(cidx4));
        assert_eq!(hv.hv.len(), 1);
        assert_eq!(hv.get(&m, addr1.clone(), 1), Some(Operand::Const(cidx4)));
        assert!(hv.get(&m, addr0.clone(), 1).is_none());
        assert!(hv.get(&m, addr0.clone(), 8).is_none());

        // Add a store which we can't prove doesn't alias.
        let addr4 = Address::from_operand(&m, Operand::Var(InstIdx::unchecked_from(4)));
        let cidx5 = m
            .insert_const(Const::Int(m.int8_tyidx(), ArbBitInt::from_u64(8, 5)))
            .unwrap();
        hv.store(&m, addr4.clone(), Operand::Const(cidx5));
        assert_eq!(hv.hv.len(), 1);
        assert_eq!(hv.get(&m, addr4.clone(), 1), Some(Operand::Const(cidx5)));

        hv.barrier();
        assert_eq!(hv.hv.len(), 0);
    }
}
