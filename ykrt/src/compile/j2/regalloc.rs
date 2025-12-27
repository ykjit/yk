//! The architecture independent part of register allocation.
//!
//! This register allocator fundamentally assumes it is involved in backwards code generation. A
//! forward register allocator ensures that, when allocating registers for instruction *n*, the
//! system is put into the correct state for the instruction *n*. In contrast, a backwards register
//! allocator ensures that, when allocating registers for instruction *n*, the system is put into
//! the correct state for instruction *n+1*. This can be rather confusing!
//!
//!
//! ## Register fills
//!
//! The allocator keeps track of register fill bits: i.e. the upper bits of a register that may not
//! be used directly by a particular value, but which some operations may depend on. For example,
//! if a register contains an `i8` and we want to do a signed subtraction, the CPU may only allow
//! 64-bit signed subtraction: we will need to set sign-extend the upper 56 bits to get a correct
//! result.
//!
//! [RegFill::Undefined] means "we don't care what the upper bits are set to"; [RegFill::Signed]
//! and [RegFill::Zeroed] says "the upper bits must be signed / zero extended".
//!
//! By carefully tracking fills, we can avoid unnecessary sign / zero extension. Where possible,
//! operations should aim to take in [RegFill::Undefined] and output [RegFill::Signed] or
//! [RegFilled::Zeroed], as this requires the fewest explicit sign / zero extensions.
//!
//!
//! ## Merging of different sized values
//!
//! When the value from instruction M can be cast from the value from instruction N (where the
//! `iidx(M) > iidx(N)`), the register allocator may be able to merge the two values into one
//! register.
//!
//! For example if an `i1` is sign-extended to an `i64`, the latter can always be derived from the
//! former by re-sign-extending the `i1`. Similarly, if an `i64` is truncated to an `i1`, the
//! latter is trivially rederivable from the former.
//!
//! The rules for what can be merged are somewhat subtle. For example consider:
//!
//! ```text
//! %0: i64 = arg [reg]
//! %1: i1 = trunc %0
//! %2: i64 = zext %1
//! ...
//! ```
//!
//! `%0` and `%1` can be merged together; `%1` and `%2` can be merged together; but `%0` and `%2`
//! cannot be merged together.
//!
//!
//! ## Register fill expectations
//!
//! The allocator assumes that values on entry to a trace have a [RegFill::Undefined] fill. Spills
//! are always zero-extended in the heap and we assume that all spills in the heap (including those
//! made by LLVM, not j2) are zero-extended too.

#[cfg(test)]
use crate::compile::j2::hir::Ty;
use crate::compile::{
    CompilationError,
    j2::{
        hir::{Block, BlockLikeT, Const, ConstKind, Inst, InstIdx, Mod, ModKind},
        hir_to_asm::HirToAsmBackend,
    },
};
use index_vec::{Idx, IndexVec, index_vec};
use smallvec::{SmallVec, smallvec};
use std::{
    assert_matches::assert_matches,
    fmt::{Debug, Display, Formatter},
};
use vob::Vob;

pub(super) struct RegAlloc<'a, AB: HirToAsmBackend + ?Sized> {
    m: &'a Mod<AB::Reg>,
    b: &'a Block,
    /// The state of each instruction.
    istates: IndexVec<InstIdx, IState>,
    /// The state of each register.
    rstates: RStates<AB::Reg>,
    /// For each instruction, where is its last use? A value of 0 means, by definition, "not yet
    /// used" because both (a) an instruction cannot use itself (b) value 0 is the last possible
    /// value.
    is_used: IndexVec<InstIdx, InstIdx>,
    /// The offset of the current stack: this must be exactly equal to the end of the last byte
    /// used in the stack.
    stack_off: u32,
    snapshots: IndexVec<SnapshotIdx, Snapshot<AB>>,
}

impl<'a, AB: HirToAsmBackend> RegAlloc<'a, AB> {
    pub(super) fn new(m: &'a Mod<AB::Reg>, b: &'a Block, stack_off: u32) -> Self {
        Self {
            m,
            b,
            istates: index_vec![IState::None; b.insts_len()],
            rstates: RStates::new(),
            is_used: index_vec![InstIdx::from_usize(0); b.insts_len()],
            stack_off,
            snapshots: index_vec![],
        }
    }

    /// Create a new register allocator from a previous [Snapshot] for guard restores.
    ///
    /// FIXME: This currently doesn't restore `is_used`! That isn't currently a problem, but it
    /// might become so in the future.
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn from_snapshot(&self, sidx: SnapshotIdx) -> Self {
        Self {
            m: self.m,
            b: self.b,
            istates: self.snapshots[sidx].istates.clone(),
            rstates: self.snapshots[sidx].rstates.clone(),
            is_used: index_vec![], // FIXME!
            stack_off: self.stack_off,
            snapshots: index_vec![],
        }
    }

    /// Snapshot this register allocator in a way that is suitable for a
    /// [super::hir::GuardRestore].
    pub(super) fn snapshot(&mut self, iidx: InstIdx) -> SnapshotIdx {
        self.snapshots.push(Snapshot {
            istates: self.istates[..iidx].to_vec(),
            rstates: self.rstates.clone(),
        })
    }

    /// Before processing the main body of a trace, set the stack offset (if any) of entry
    /// variables, so that we don't end up unnecessarily spilling them twice during execution.
    /// This is an optimisation rather than a necessity.
    pub(super) fn set_entry_stacks_at_end(&mut self, entry_vlocs: &[VarLocs<AB::Reg>]) {
        for (iidx, vlocs) in entry_vlocs
            .iter()
            .enumerate()
            .map(|(i, x)| (InstIdx::from_usize(i), x))
        {
            for vloc in vlocs.iter() {
                match vloc {
                    VarLoc::Stack(stack_off) => {
                        assert_eq!(self.istates[iidx], IState::None);
                        self.istates[iidx] = IState::Stack(*stack_off);
                    }
                    VarLoc::StackOff(stack_off) => {
                        assert_eq!(self.istates[iidx], IState::None);
                        self.istates[iidx] = IState::StackOff(*stack_off);
                    }
                    VarLoc::Reg(_) => (),
                    VarLoc::Const(_) => (),
                }
            }
        }
    }

    /// After processing the main body of a trace, set the [VarLocs]s of the entry variables.
    pub(super) fn set_entry_vlocs_at_start(
        &mut self,
        be: &mut AB,
        entry_vlocs: &[VarLocs<AB::Reg>],
    ) {
        // In essence, this is a simple, special case of normal register allocation. First we work
        // out what the rstate after trace entry will be, diff that, and generate the appropriate
        // code.

        let mut in_rstate = RStates::<AB::Reg>::new();
        for (iidx, vlocs) in entry_vlocs
            .iter()
            .enumerate()
            .map(|(x, y)| (InstIdx::from(x), y))
        {
            for vloc in vlocs.iter() {
                if let VarLoc::Reg(reg) = vloc {
                    if let ModKind::Coupler { .. } | ModKind::Loop { .. } = self.m.kind {
                        // Because of the way we call traces (see bc59d8bff411931440459fa3377a137e8537a32f
                        // for details), caller saved registers are potentially corrupted at the very start
                        // of a loop trace.
                        //
                        // FIXME: This is a horrible hack and assumes that all [Block]s in a loop
                        // trace are subject to the same restriction.
                        if reg.is_caller_saved() {
                            continue;
                        }
                    }
                    if !in_rstate.iidxs(*reg).is_empty() {
                        let bitw = self.b.inst_bitw(self.m, iidx);
                        if bitw > iidxs_maxbitw(self.m, self.b, in_rstate.iidxs(*reg)) {
                            todo!();
                        }
                    }
                    in_rstate.set_fill_iidxs(*reg, RegFill::Undefined, smallvec![iidx]);
                }
            }
        }

        let mut ractions = RegActions::new();
        self.rstate_diff_to_action(&in_rstate, &mut ractions);
        self.toposort_distinct_copies(&mut ractions).unwrap();
        self.asm_ractions(be, &ractions).unwrap();

        // Because we are, in a sense, allocating registers for multiple instructions in one go, we
        // now need to find all `arg` instructions that we need to spill i.e. those where (1) they
        // aren't spilt coming into the trace (2) during register allocation we've marked them down
        // as needing spilling.
        for (iidx, vlocs) in entry_vlocs
            .iter()
            .enumerate()
            .map(|(x, y)| (InstIdx::from(x), y))
        {
            if let IState::Stack(stack_off) = self.istates[iidx]
                && !vlocs.iter().any(|vloc| matches!(vloc, VarLoc::Stack(_)))
            {
                let Some(VarLoc::Reg(reg)) =
                    vlocs.iter().find(|vloc| matches!(vloc, VarLoc::Reg(_)))
                else {
                    panic!("{iidx:?}")
                };
                let bitw = self.b.inst_bitw(self.m, iidx);
                be.spill(*reg, RegFill::Undefined, stack_off, bitw).unwrap();
            }
        }
    }

    /// Set the [VarLocs] of a [super::hir::Exit] instruction.
    pub(super) fn set_exit_vlocs(
        &mut self,
        be: &mut AB,
        is_loop: bool,
        all_entry_vlocs: &[VarLocs<AB::Reg>],
        exit_iidx: InstIdx,
        all_exit_vars: &[InstIdx],
        all_exit_vlocs: &[VarLocs<AB::Reg>],
    ) -> Result<(), CompilationError> {
        assert_eq!(all_exit_vars.len(), all_exit_vlocs.len());

        // At a trace's exit, we potentially have to shuffle the stack around. In most cases we
        // have to "move" a value to/from the same stack location, but not always. Consider a trace
        // such as:
        //
        // ```
        // %1 = arg ; stack_off = 16
        // %2 = ... ;
        // ...
        // ; spill %2
        // call f(%1)
        // exit [%10]
        // ```
        //
        // At the trace's end, %2 needs to be spilt into `stack_off=16`, but we can't do that until
        // after the last use of %1. That means that %2 will be spilt somewhere that isn't
        // `stack_off=16` and will need to be moved to that before `exit`.
        //
        // We then have the potential for moves to overlap and so on. The basic idea of this loop
        // is that we calculate all the moves, and then perform them in a safe order where "safe"
        // might require us to spill extra values or use temporary registers.

        // Always obtaining a temporary register is bad, and this approach is particularly bad: it
        // could fail to find a candidate register!
        let tmp_reg = self
            .rstates
            .iter()
            .find_map(|(reg, rstate)| {
                if rstate.iidxs.is_empty() {
                    Some(reg)
                } else {
                    None
                }
            })
            .unwrap();

        // Push constants into registers as the very last thing (these registers are excellent
        // "temporary" candidates, so do them last).
        for (iidx, exit_vlocs) in all_exit_vars.iter().zip(all_exit_vlocs.iter()) {
            if let Inst::Const(Const { kind, .. }) = self.b.inst(*iidx) {
                for vloc in exit_vlocs.iter() {
                    let bitw = self.b.inst_bitw(self.m, *iidx);
                    match vloc {
                        VarLoc::Stack(_) => (),
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(reg) => {
                            be.move_const(*reg, None, bitw, RegFill::Zeroed, kind)?;
                        }
                        VarLoc::Const(_) => (),
                    }
                }
            }
        }

        // Push constants into the stack as the penultimate thing (these may require up to two
        // temporary registers).
        for (iidx, exit_vlocs) in all_exit_vars.iter().zip(all_exit_vlocs.iter()) {
            if let Inst::Const(Const { kind, .. }) = self.b.inst(*iidx) {
                for vloc in exit_vlocs.iter() {
                    let bitw = self.b.inst_bitw(self.m, *iidx);
                    match vloc {
                        VarLoc::Stack(stack_off) => {
                            if let IState::Stack(spill_stack_off) = self.istates[*iidx]
                                && *stack_off == spill_stack_off
                            {
                                continue;
                            }
                            be.spill(tmp_reg, RegFill::Zeroed, *stack_off, bitw)?;
                            be.move_const(tmp_reg, None, bitw, RegFill::Zeroed, kind)?;
                        }
                        VarLoc::StackOff(_) => todo!(),
                        VarLoc::Reg(_) => (),
                        VarLoc::Const(_) => (),
                    }
                }
            }
        }

        let mut moves = Vec::new();
        for (iidx, exit_vlocs) in all_exit_vars.iter().zip(all_exit_vlocs.iter()) {
            if let Inst::Const(_) = self.b.inst(*iidx) {
                // We handled these above.
                continue;
            }
            self.is_used[*iidx] = exit_iidx;
            let bitw = self.b.inst_bitw(self.m, *iidx);
            for vloc in exit_vlocs.iter() {
                if exit_vlocs
                    .iter()
                    .any(|vloc| matches!(vloc, VarLoc::Const(_)))
                {
                    todo!();
                }
                match vloc {
                    VarLoc::Stack(to_stack_off) => {
                        if usize::from(*iidx) < all_entry_vlocs.len()
                            && let Some(VarLoc::Stack(from_stack_off)) = all_entry_vlocs
                                [usize::from(*iidx)]
                            .iter()
                            .find(|vloc| matches!(vloc, VarLoc::Stack(_)))
                        {
                            moves.push((bitw, *from_stack_off, *to_stack_off));
                            continue;
                        }

                        let tmp_stack_off = be.align_spill(self.stack_off, bitw);
                        self.stack_off = tmp_stack_off;
                        assert_eq!(self.istates[*iidx], IState::None);
                        self.istates[*iidx] = IState::Stack(tmp_stack_off);
                        moves.push((bitw, tmp_stack_off, *to_stack_off));
                    }
                    VarLoc::StackOff(stack_off) => {
                        if is_loop {
                            assert_eq!(self.istates[*iidx], IState::StackOff(*stack_off));
                        } else if self.istates[*iidx] != IState::StackOff(*stack_off) {
                            todo!();
                        }
                    }
                    VarLoc::Reg(reg) => {
                        assert!(!self.rstates.iidxs(*reg).contains(iidx));
                        self.rstates.set_fill(*reg, RegFill::Undefined);
                        self.rstates.iidxs_mut(*reg).push(*iidx);
                    }
                    VarLoc::Const(_) => (),
                }
            }
        }

        moves.sort_unstable_by_key(|x| x.1);
        for (bitw, from_stack_off, to_stack_off) in moves.into_iter() {
            if from_stack_off == to_stack_off {
                continue;
            }
            be.move_stack_val(bitw, from_stack_off, to_stack_off, tmp_reg);
        }

        Ok(())
    }

    /// Topologically sort `ractions.distinct_copies`, breaking cycles as necessary, such that no
    /// [RegAction::Copy] can overwrite a value needed by a later [RegAction::Copy]. This will
    /// update the `ractions.spills` and `raction.unspills` as necessary. Note:
    /// `ractions.distinct_copies` will be sorted in reverse-order (i.e. suitable for reverse code
    /// generation).
    fn toposort_distinct_copies(
        &self,
        ractions: &mut RegActions<AB::Reg>,
    ) -> Result<(), CompilationError> {
        // If there are 0 or 1 distinct [RegCopy]s, there is no chance of overlap, and no need to
        // do a topological sort (which is, relatively speaking, fairly slow).
        if ractions.distinct_copies.len() <= 1 {
            return Ok(());
        }

        // This is a fixed-point algorithm which produces an ordered set of copies without cycles.
        //
        // To do that, on each iteration we attempt a topological sort on `ractions.distinct_copies`
        // using Kahn's algorithm. If the topological sort fails, we have detected a cycle; we then
        // take one of the registers identified in the cycle and use it to break that cycle; and
        // try again. If the topological sort succeeds, we can then use its output to sort
        // `ractions.distinct_copies`.
        loop {
            // Stage 1: a topological sort using Kahn's algorithm. Registers are nodes and
            // [RegCopy]s define edges.

            // Work out all the nodes (i.e. registers) and the number of incoming edges (i.e.
            // register copies) to those nodes.
            let mut nodes = Vob::from_elem(false, AB::Reg::MAX_REGIDX.index());
            let mut indegrees = index_vec![0; AB::Reg::MAX_REGIDX.index()];
            for RegCopy {
                src_reg, dst_reg, ..
            } in &ractions.distinct_copies
            {
                nodes.set(src_reg.regidx().index(), true);
                nodes.set(dst_reg.regidx().index(), true);
                indegrees[dst_reg.regidx()] += 1;
            }

            // Create the initial queue. Note: this must not contain duplicate nodes. Fortunately,
            // that happens naturally because of our `Vob` representation of `nodes`.
            let mut queue = nodes
                .iter_set_bits(..)
                .map(|x| AB::Reg::from_regidx(<AB::Reg as RegT>::RegIdx::from_usize(x)))
                .filter(|reg| indegrees[reg.regidx()] == 0)
                .collect::<Vec<_>>();

            // The topologically sorted output we will build up.
            let mut ordered = Vec::new();
            while let Some(reg) = queue.pop() {
                ordered.push(reg);
                for RegCopy {
                    src_reg: nbr_src_reg,
                    dst_reg,
                    ..
                } in &ractions.distinct_copies
                {
                    if reg == *nbr_src_reg {
                        assert!(indegrees[dst_reg.regidx()] > 0);
                        indegrees[dst_reg.regidx()] -= 1;
                        if indegrees[dst_reg.regidx()] == 0 {
                            queue.push(*dst_reg);
                        }
                    }
                }
            }

            // Stage 2: did the topological sort succeed?
            if ordered.len() == nodes.iter_set_bits(..).count() {
                // It succeeded (i.e. so no cycles were detected). However, the topological sort
                // has given us an ordered sequence of _nodes_ but we want an ordered sequence of
                // _edges_. Fortunately we can easily derive the latter from the former. Note: we
                // deliberately sort these in "reverse" order due to the general context of
                // backwards code generation.
                ractions.distinct_copies.sort_by_key(
                    |RegCopy {
                         src_reg, dst_reg, ..
                     }| {
                        (
                            ordered.iter().position(|x| x == src_reg).unwrap(),
                            ordered.iter().position(|x| x == dst_reg).unwrap(),
                        )
                    },
                );
                return Ok(());
            } else {
                // It failed. Each register in the cycle -- of which there must be at least two --
                // will have an indegree greater than zero. Arbitrarily pick one such register, and
                // use it to break the cycle.
                let break_reg = AB::Reg::from_regidx(
                    indegrees
                        .iter_enumerated()
                        .find(|(_, degree)| **degree > 0)
                        .unwrap()
                        .0,
                );

                ractions.spills.push(RegSpill {
                    iidxs: self.rstates.iidxs(break_reg).clone(),
                });

                let mut new_unspills = Vec::new();
                ractions.distinct_copies.retain(|rcopy| {
                    if rcopy.dst_reg != break_reg {
                        true
                    } else {
                        new_unspills.push(RegUnspill {
                            iidxs: self.rstates.iidxs(rcopy.dst_reg).clone(),
                            reg: rcopy.dst_reg,
                            fill: rcopy.src_fill,
                        });
                        false
                    }
                });
                ractions.unspills.extend(new_unspills);
            }
        }
    }

    /// Generate the code for a [RegActions]. This guarantees to generate (bearing in mind this is
    /// in the context of reverse code generation) in the following order: unspills, copies, and
    /// spills.
    fn asm_ractions(
        &mut self,
        be: &mut AB,
        ractions: &RegActions<AB::Reg>,
    ) -> Result<(), CompilationError> {
        'a: for RegUnspill { iidxs, reg, fill } in ractions.unspills.iter().rev() {
            let (max_bitw, mut max_bitw_iter) = iter_maxbitw_iidxs(self.m, self.b, iidxs);
            // If we can unspill a constant, that's likely to be quicker than unspilling from
            // memory.
            for iidx in max_bitw_iter.clone() {
                if let Inst::Const(Const { kind, .. }) = self.b.inst(iidx) {
                    let tmp_reg = if let Some(mut tmp_reg_iter) = be.const_needs_tmp_reg(*reg, kind)
                    {
                        match tmp_reg_iter.find(|reg| self.rstates.iidxs(*reg).is_empty()) {
                            Some(x) => Some(x),
                            None => todo!(),
                        }
                    } else {
                        None
                    };
                    be.move_const(*reg, tmp_reg, max_bitw, *fill, kind)?;
                    continue 'a;
                } else if let IState::StackOff(stack_off) = self.istates[iidx] {
                    be.move_stackoff(*reg, stack_off)?;
                    continue 'a;
                }
            }

            // Arbitrarily pick one of the maximum bit width `iidx`s and unspill it.
            let unspill_iidx = max_bitw_iter.nth(0).unwrap();
            let stack_off = match self.istates[unspill_iidx] {
                IState::None => {
                    let stack_off = be.align_spill(self.stack_off, max_bitw);
                    self.stack_off = stack_off;
                    assert_eq!(self.istates[unspill_iidx], IState::None);
                    self.istates[unspill_iidx] = IState::Stack(stack_off);
                    stack_off
                }
                IState::Stack(stack_off) => stack_off,
                IState::StackOff(_) => unreachable!(),
            };
            be.unspill(stack_off, *reg, *fill, max_bitw)?;
        }

        for RegCopy {
            bitw,
            src_reg,
            src_fill,
            dst_reg,
            dst_fill,
        } in ractions.distinct_copies.iter()
        {
            be.arrange_fill(*dst_reg, *src_fill, *bitw, *dst_fill);
            be.copy_reg(*src_reg, *dst_reg)?;
        }

        for RegCopy {
            bitw,
            src_reg: _,
            src_fill,
            dst_reg,
            dst_fill,
        } in ractions.self_copies.iter()
        {
            be.arrange_fill(*dst_reg, *src_fill, *bitw, *dst_fill);
        }

        for RegSpill { iidxs } in ractions.spills.iter().rev() {
            for iidx in iidxs {
                if matches!(self.b.inst(*iidx), Inst::Const(_))
                    || matches!(self.istates[*iidx], IState::StackOff(_))
                {
                    // We don't need to spill constants or `IState::StackOff`s.
                    continue;
                }

                if let IState::None = self.istates[*iidx] {
                    let bitw = self.b.inst_bitw(self.m, *iidx);
                    let stack_off = be.align_spill(self.stack_off, bitw);
                    self.stack_off = stack_off;
                    self.istates[*iidx] = IState::Stack(stack_off);
                }
            }
        }

        Ok(())
    }

    /// Returns the stack offset, relative to the base of the control point frame, that this block
    /// will need.
    pub(super) fn stack_off(&self) -> u32 {
        self.stack_off
    }

    /// Has the instruction `iidx` been used thus far?
    ///
    /// Note: being used in a guard's entry_vars counts as "being used".
    pub(super) fn is_used(&self, iidx: InstIdx) -> bool {
        usize::from(*self.is_used.get(usize::from(iidx)).unwrap()) > 0
    }

    /// Force the value `iidx` to be marked as used at `cur_iidx`. Must only be used for testing purposes.
    #[cfg(test)]
    pub(super) fn blackbox(&mut self, cur_iidx: InstIdx, iidx: InstIdx) {
        self.is_used[iidx] = cur_iidx;
    }

    /// Return an iterator which will produce all the registers in which `iidx` is contained.
    pub(super) fn iter_reg_for(&self, iidx: InstIdx) -> impl Iterator<Item = AB::Reg> {
        self.rstates
            .iter()
            .filter_map(move |(reg, rstate)| rstate.iidxs.contains(&iidx).then_some(reg))
    }

    /// For an instruction `iidx` in a [GuardRestore], return its [VarLocs].
    ///
    /// Note: currently we assume that all instructions in this situation have been spilt. That
    /// will not always be the case.
    pub(super) fn varlocs_for_deopt(&self, iidx: InstIdx) -> VarLocs<AB::Reg> {
        // To keep deopt simple, we currently don't make use of the fact that an instruction might
        // be in a register: we assume everything has been spilled.
        if let IState::Stack(stack_off) = self.istates[iidx] {
            VarLocs::new(smallvec![VarLoc::Stack(stack_off)])
        } else if let IState::StackOff(stack_off) = self.istates[iidx] {
            VarLocs::new(smallvec![VarLoc::StackOff(stack_off)])
        } else if let Inst::Const(Const { kind, .. }) = self.b.inst(iidx) {
            VarLocs::new(smallvec![VarLoc::Const(kind.clone())])
        } else {
            todo!(
                "{iidx:?} {:?} {:?}\n  {:?}",
                self.b.inst(iidx),
                self.iter_reg_for(iidx).collect::<Vec<_>>(),
                self.rstates
            );
        }
    }

    /// Forcibly spill any unspilt values in `reg` in a way that is *only* suitable for calling at
    /// the point of a deopt call.
    pub(super) fn ensure_spilled_for_deopt(
        &mut self,
        be: &mut AB,
        reg: AB::Reg,
    ) -> Result<(), CompilationError> {
        for iidx in self.rstates.iidxs(reg) {
            // We don't need to spill things that are already spilt, or constants.
            if matches!(&self.istates[*iidx], IState::Stack(_) | IState::StackOff(_))
                || matches!(self.b.inst(*iidx), Inst::Const(_))
            {
                continue;
            }

            let bitw = self.b.inst_bitw(self.m, *iidx);
            self.stack_off = be.align_spill(self.stack_off, bitw);
            assert_eq!(self.istates[*iidx], IState::None);
            self.istates[*iidx] = IState::Stack(self.stack_off);
            be.spill(reg, self.rstates.fill(reg), self.stack_off, bitw)?;
        }
        Ok(())
    }

    /// Allocate registers for the instruction at position `iidx`. Note: this function may leave
    /// CPU flags in an undefined state.
    ///
    /// # Panics
    ///
    /// If `iidx` is an [Inst::Const] or any [RegCnstr]s contain [RegCnstrFill::AnyOf].
    pub(super) fn alloc<const N: usize>(
        &mut self,
        be: &mut AB,
        iidx: InstIdx,
        cnstrs: [RegCnstr<AB::Reg>; N],
    ) -> Result<[AB::Reg; N], CompilationError> {
        // If there are `AnyOf`s, the user should have called `alloc_with_fills`.
        assert!(cnstrs.iter().all(|x| match x {
            RegCnstr::InputOutput {
                in_fill, out_fill, ..
            } =>
                !matches!(in_fill, RegCnstrFill::AnyOf(_))
                    && !matches!(out_fill, RegCnstrFill::AnyOf(_)),
            RegCnstr::Output { out_fill, .. } | RegCnstr::Cast { out_fill, .. } =>
                !matches!(out_fill, RegCnstrFill::AnyOf(_)),
            RegCnstr::Input { in_fill, .. } => !matches!(in_fill, RegCnstrFill::AnyOf(_)),
            RegCnstr::Clobber { .. } | RegCnstr::Temp { .. } | RegCnstr::KeepAlive { .. } => true,
        }));

        Ok(self.alloc_with_fills(be, iidx, cnstrs)?.map(|(reg, _)| reg))
    }

    /// Allocate registers and the required output [RegFill]s for the instruction at position
    /// `iidx`. Note: This function may leave CPU flags in an undefined state.
    ///
    /// The output [RegFill] for a given [RegCnstr] is defined as follows:
    ///
    /// * [RegCnstr::Clobber], [RegCnstr::Temp], and [RegCnstr::KeepAlive] always return
    ///   [RegFill::Undefined]
    /// * [RegCnstrFill::Undefined], [RegCnstrFill::Signed], and [RegCnstrFill::Zeroed] return
    ///   [RegFill::Undefined], [RegFill::Signed], and [RegFill::Zeroed] respectively.
    /// * [RegCnstrFill::AnyOf] returns [RegFill::Undefined], [RegFill::Signed], and
    ///   [RegFill::Zeroed] as appropriate.
    ///
    /// # Panics
    ///
    /// If `iidx` is an [Inst::Const]
    pub(super) fn alloc_with_fills<const N: usize>(
        &mut self,
        be: &mut AB,
        iidx: InstIdx,
        mut cnstrs: [RegCnstr<AB::Reg>; N],
    ) -> Result<[(AB::Reg, RegFill); N], CompilationError> {
        assert!(!matches!(self.b.inst(iidx), Inst::Const(_)));
        // Let us call `self.rstate` rstate *n+1:in* (i.e. the input for instruction *iidx+1*). What
        // we need to do here is multi-fold:
        //
        // 1. Decide which registers the calling instruction should use for output (*regs_out*) and
        //    input (*regs_in*).
        // 2. Calculate rstate *n:out* from *n+1:in* and *regs_out*.
        // 3. Calculate how to get from *n:out* to *n+1:in* and generate the code to do so.
        // 4. Calculate rstate *n:in* from *n:out* and *regs_in*.
        // 5. Set self.rstates to *n:in* and deal with [RegCnstr::KeepAlive]s.
        //
        // Notice the careful differentiation of "output" and "input" rstates!

        // There must be at most 1 output register.
        assert!(
            cnstrs
                .iter()
                .filter(|x| match x {
                    RegCnstr::InputOutput { .. }
                    | RegCnstr::Output { .. }
                    | RegCnstr::Cast { .. } => true,
                    RegCnstr::Clobber { .. }
                    | RegCnstr::Input { .. }
                    | RegCnstr::Temp { .. }
                    | RegCnstr::KeepAlive { .. } => false,
                })
                .count()
                <= 1
        );

        // `AnyOf` (currently) only makes sense in `out_fill`s of non-`Cast`s.
        assert!(cnstrs.iter().all(|x| match x {
            RegCnstr::InputOutput { in_fill, .. } => !matches!(in_fill, RegCnstrFill::AnyOf(_)),
            RegCnstr::Input { in_fill, .. } => !matches!(in_fill, RegCnstrFill::AnyOf(_)),
            RegCnstr::Cast { out_fill, .. } => !matches!(out_fill, RegCnstrFill::AnyOf(_)),
            RegCnstr::Output { .. }
            | RegCnstr::Clobber { .. }
            | RegCnstr::Temp { .. }
            | RegCnstr::KeepAlive { .. } => true,
        }));

        // Phase 1: Find registers for constraints. Note that multiple constraints may end up
        // using the same register. This phase does not mutate `self`.
        let allocs = self.find_regs_for_constraints(iidx, &cnstrs)?;

        // Phase 2: Calculate *n:out*. Counter-intuitively, we need to calculate this "forwards",
        // even though when we generate code we'll do so backwards.
        //
        // We start by using *n+1:in* as our basis because some of these registers will be
        // untouched by the current instruction.
        let mut n_out = self.rstates.clone();

        // 2.1: Update for the state immediately after the instruction has produced outputs and
        // work out what fill this constraint should have.
        let mut output_reg = None; // Needed when outputs can end up in multiple registers.
        let mut rtn_fills = Vec::with_capacity(N);
        for (reg, cnstr) in allocs.iter().cloned().zip(cnstrs.iter_mut()) {
            match cnstr {
                RegCnstr::Clobber { .. } | RegCnstr::Temp { .. } => {
                    n_out.set_fill_iidxs(reg, RegFill::Undefined, smallvec![]);
                    rtn_fills.push(RegFill::Undefined);
                }
                RegCnstr::Input {
                    in_iidx,
                    in_fill,
                    clobber,
                    ..
                } => {
                    if *clobber {
                        n_out.set_fill_iidxs(reg, RegFill::Undefined, smallvec![]);
                        rtn_fills.push(RegFill::Undefined);
                    } else {
                        let in_fill = RegFill::from_regcnstrfill(in_fill);
                        n_out.set_fill_iidxs(reg, in_fill, smallvec![*in_iidx]);
                        rtn_fills.push(in_fill);
                    }
                }
                RegCnstr::InputOutput { out_fill, .. } | RegCnstr::Output { out_fill, .. } => {
                    if let RegCnstrFill::AnyOf(cnd_fills) = out_fill {
                        let fill = if self.rstates.iidxs(reg).contains(&iidx) {
                            self.rstates.fill(reg)
                        } else if let Some(other_reg) = self.iter_reg_for(iidx).nth(0) {
                            self.rstates.fill(other_reg)
                        } else if cnd_fills.has_undefined() {
                            RegFill::Undefined
                        } else {
                            todo!();
                        };
                        assert!(AnyOfFill::from_regfill(fill).intersects_with(cnd_fills));
                        *out_fill = RegCnstrFill::from_regfill(fill);
                    }
                    let out_fill = RegFill::from_regcnstrfill(out_fill);
                    n_out.set_fill_iidxs(reg, out_fill, smallvec![iidx]);
                    rtn_fills.push(out_fill);
                    output_reg = Some(reg);
                }
                RegCnstr::Cast {
                    in_iidx, out_fill, ..
                } => {
                    let out_fill = RegFill::from_regcnstrfill(out_fill);
                    if n_out.iidxs(reg).contains(&iidx) {
                        if n_out.iidxs(reg).len() == 1
                            || n_out.fill(reg) == out_fill
                            || n_out.fill(reg) == RegFill::Undefined
                            || out_fill == RegFill::Undefined
                        {
                            n_out.set_fill_iidxs(reg, out_fill, smallvec![iidx]);
                            rtn_fills.push(out_fill);
                            output_reg = Some(reg);
                        } else {
                            todo!()
                        }
                    } else if n_out.iidxs(reg).contains(in_iidx) {
                        if n_out.iidxs(reg).len() == 1
                            || n_out.fill(reg) == out_fill
                            || n_out.fill(reg) == RegFill::Undefined
                            || out_fill == RegFill::Undefined
                        {
                            n_out.iidxs_mut(reg).push(iidx);
                            n_out.set_fill(reg, out_fill);
                            rtn_fills.push(out_fill);
                            output_reg = Some(reg);
                        } else {
                            todo!()
                        }
                    } else {
                        n_out.set_fill_iidxs(reg, out_fill, smallvec![iidx]);
                        rtn_fills.push(out_fill);
                        output_reg = Some(reg);
                    }
                }
                RegCnstr::KeepAlive { .. } => {
                    rtn_fills.push(RegFill::Undefined);
                }
            }
        }

        // The output, if there is any, can only be in a single register. However, `n_out` might at
        // this point require that value to be duplicated in multiple registers. Rather than be
        // clever and do that duplication here, we simply nullify all of the duplicates, and rely
        // on `rstate_diff_to_actions` to do the duplication for us.
        if let Some(output_reg) = output_reg {
            for (reg, rstate) in n_out.iter_mut() {
                if reg != output_reg && rstate.iidxs.contains(&iidx) {
                    rstate.iidxs.retain(|x| *x != iidx);
                }
            }
        }

        // Phase 3: Calculate how to get from *n:out* to *n+1:in* and generate the code to do so.

        // Phase 3.1: at this point (remember, it's reverse code generation!), all the outputs are
        // in the right registers, but we might need to shuffle lots of other things around, and
        // unspill others, to get into the right state for *n:out*. First we calculate a simple
        // "diff", which will happily give us register copies that overwrite each other...
        let mut ractions = RegActions::new();
        self.rstate_diff_to_action(&n_out, &mut ractions);
        // ...so we then topologically sort the distinct copies, which will break any cycles.
        self.toposort_distinct_copies(&mut ractions)?;
        // We now have an [RegActions] which we can directly generate code from.
        self.asm_ractions(be, &ractions)?;

        // Before we can get the output fills in the correct states, we need to discount anything
        // we'd immediately unspill on top of.
        for RegUnspill { reg, .. } in &ractions.unspills {
            n_out.set_fill_iidxs(*reg, RegFill::Undefined, smallvec![]);
        }

        // Phase 3.2: Spill outputs if they will need to be unspilled later.
        //
        // This also turns out to be a convenient place to calculate which instructions' values are
        // used, so do that at the same time.
        for (reg, cnstr) in allocs.iter().zip(cnstrs.iter()) {
            match cnstr {
                RegCnstr::Clobber { .. } | RegCnstr::Temp { .. } => (),
                RegCnstr::Input {
                    in_iidx,
                    in_fill: _,
                    ..
                } => {
                    self.is_used[*in_iidx] = iidx;
                }
                RegCnstr::InputOutput {
                    in_iidx, out_fill, ..
                }
                | RegCnstr::Cast {
                    in_iidx, out_fill, ..
                } => {
                    if let IState::Stack(stack_off) = self.istates[iidx] {
                        let out_bitw = self.b.inst_bitw(self.m, iidx);
                        be.spill(
                            *reg,
                            RegFill::from_regcnstrfill(out_fill),
                            stack_off,
                            out_bitw,
                        )?;
                    }
                    self.is_used[*in_iidx] = iidx;
                }
                RegCnstr::Output {
                    out_fill,
                    regs: _,
                    can_be_same_as_input: _,
                } => {
                    if let IState::Stack(stack_off) = self.istates[iidx] {
                        let out_bitw = self.b.inst_bitw(self.m, iidx);
                        be.spill(
                            *reg,
                            RegFill::from_regcnstrfill(out_fill),
                            stack_off,
                            out_bitw,
                        )?;
                    }
                }
                RegCnstr::KeepAlive { .. } => (),
            }
        }

        // Phase 4. Calculate rstate *n:in* from *n:out* and *regs_in*.
        let mut n_in = n_out;

        for (reg, cnstr) in allocs.iter().cloned().zip(cnstrs.iter()) {
            match cnstr {
                RegCnstr::Clobber { .. } | RegCnstr::Temp { .. } => {
                    // These are all handled in phase 2.1.
                }
                RegCnstr::Input {
                    in_iidx, in_fill, ..
                }
                | RegCnstr::InputOutput {
                    in_iidx, in_fill, ..
                } => {
                    n_in.set_fill_iidxs(
                        reg,
                        RegFill::from_regcnstrfill(in_fill),
                        smallvec![*in_iidx],
                    );
                }
                RegCnstr::Output {
                    can_be_same_as_input,
                    ..
                } => {
                    if !can_be_same_as_input
                        || !allocs
                            .iter()
                            .cloned()
                            .zip(cnstrs.iter())
                            .any(|(cnd_reg, cnd_cnstr)| {
                                reg == cnd_reg && matches!(cnd_cnstr, RegCnstr::Input { .. })
                            })
                    {
                        n_in.set_fill_iidxs(reg, RegFill::Undefined, smallvec![]);
                    }
                }
                RegCnstr::Cast {
                    in_iidx, out_fill, ..
                } => {
                    n_in.set_fill_iidxs(
                        reg,
                        RegFill::from_regcnstrfill(out_fill),
                        smallvec![*in_iidx],
                    );
                }
                RegCnstr::KeepAlive { .. } => (),
            }
        }

        for (_, rstate) in n_in.iter_mut() {
            rstate.iidxs.retain(|x| *x != iidx);
            if rstate.iidxs.is_empty() {
                rstate.fill = RegFill::Undefined;
            }
        }

        // Phase 5: Set self.rstates to *n:in* and deal with [RegCnstr::KeepAlive]s.
        self.rstates = n_in;

        // Deal with `KeepAlive`: at this point we know what the incoming registers for the "next"
        // rstate will be (because it's now the "current" rstate).
        for (_, cnstr) in allocs.iter().cloned().zip(cnstrs.iter()) {
            if let RegCnstr::KeepAlive { iidxs } = cnstr {
                for ka_iidx in iidxs.iter() {
                    let reg = self.iter_reg_for(*ka_iidx).nth(0);
                    if reg.is_none()
                        && let IState::None = self.istates[*ka_iidx]
                        && !matches!(self.b.inst(*ka_iidx), Inst::Const(_))
                    {
                        let mut found = false;
                        for reg in be.iter_possible_regs(self.b, *ka_iidx) {
                            if self.rstates.iidxs(reg).is_empty() {
                                self.rstates.set_fill_iidxs(
                                    reg,
                                    RegFill::Undefined,
                                    smallvec![*ka_iidx],
                                );
                                found = true;
                                break;
                            }
                        }
                        if !found && !matches!(self.b.inst(*ka_iidx), Inst::Const(_)) {
                            let ka_bitw = self.b.inst_bitw(self.m, *ka_iidx);
                            self.stack_off = be.align_spill(self.stack_off, ka_bitw);
                            self.istates[*ka_iidx] = IState::Stack(self.stack_off);
                        }
                    }
                    self.is_used[*ka_iidx] = iidx;
                }
            }
        }

        assert_eq!(allocs.len(), rtn_fills.len());
        Ok(allocs
            .into_iter()
            .zip(rtn_fills)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap())
    }

    /// For [Inst::Const] instructions only, allocate registers. This function should only be
    /// called by [hir_to_asm]: backends should not call it.
    ///
    /// # Panics
    ///
    /// If this function is called on any other kind of instruction.
    pub(super) fn alloc_const(
        &mut self,
        be: &mut AB,
        iidx: InstIdx,
    ) -> Result<(), CompilationError> {
        assert_matches!(self.b.inst(iidx), Inst::Const(_));
        // Constants should never be spilled.
        assert_eq!(self.istates[iidx], IState::None);

        // The constant might exist in multiple registers: we turn each into a [RegUnspill] and
        // defer to [Self::asm_ractions] to choose how to do that unspilling.
        let mut unspills = Vec::new();
        for (reg, rstate) in self.rstates.iter_mut() {
            if rstate.iidxs.contains(&iidx) {
                rstate.iidxs.retain(|x| *x != iidx);
                // If `rstate.iidxs` isn't empty, that means that multiple instructions' values
                // have been merged into one register, and we can let the earlier value.
                if rstate.iidxs.is_empty() {
                    unspills.push(RegUnspill {
                        iidxs: smallvec![iidx],
                        reg,
                        fill: rstate.fill,
                    });
                }
            }
        }

        if !unspills.is_empty() {
            let mut ractions = RegActions::new();
            ractions.unspills = unspills;
            self.asm_ractions(be, &ractions)?;
        }
        Ok(())
    }

    /// For each constraint in `cnstrs`, return the register it should end up in. Note: multiple
    /// constraints may end up being allocated to the same register.
    ///
    /// Note: `cnstrs` must contain at most one `Output`/`InputOutput`. Violating this will lead to
    /// undefined behaviour.
    fn find_regs_for_constraints<const N: usize>(
        &self,
        iidx: InstIdx,
        cnstrs: &[RegCnstr<AB::Reg>; N],
    ) -> Result<[AB::Reg; N], CompilationError> {
        // This is a somewhat simple minded approach to allocating registers. In particular, it
        // misses opportunities to merge together `Input`s into the same register (this allows us
        // to avoid worrying about things like incompatible fills).

        let mut allocs = [None; N];

        // If the (sole!) output can be the same as an input, we need to a different dance at some
        // points below, so work out if this is the case now.
        let output_can_be_same_as_input = cnstrs.iter().enumerate().find_map(|(i, cnstr)| {
            if matches!(
                cnstr,
                RegCnstr::Output {
                    can_be_same_as_input: true,
                    ..
                }
            ) {
                Some(i)
            } else {
                None
            }
        });

        // Allocate all hard constraints i.e. those where only one register will do.
        for (i, cnstr) in cnstrs.iter().enumerate() {
            match cnstr {
                RegCnstr::Clobber { reg } => {
                    assert!(allocs[i].is_none());
                    allocs[i] = Some(*reg);
                }
                RegCnstr::Input { regs, .. }
                | RegCnstr::InputOutput { regs, .. }
                | RegCnstr::Output { regs, .. }
                | RegCnstr::Cast { regs, .. } => {
                    if regs.len() == 1 {
                        // FIXME: We could in fact deal with some overlaps, but it'll be easier to
                        // do so if/when we see it in practise.
                        assert!(allocs[i].is_none());
                        allocs[i] = Some(regs[0]);
                    }
                }
                RegCnstr::Temp { .. } | RegCnstr::KeepAlive { .. } => (),
            }
        }

        // If values are already allocated to a register, and we aren't going to use that register
        // for anything else, keep the register as-is.
        let find_alloc = |allocs: &mut [Option<AB::Reg>; N], i, regs: &[AB::Reg], find_iidx| {
            if let Some(reg) = self
                .iter_reg_for(find_iidx)
                .find(|reg| regs.contains(reg) && !allocs.contains(&Some(*reg)))
            {
                allocs[i] = Some(reg);
                true
            } else {
                false
            }
        };

        for (i, cnstr) in cnstrs.iter().enumerate() {
            if allocs[i].is_some() {
                continue;
            }
            match cnstr {
                RegCnstr::Clobber { .. } => unreachable!(),
                RegCnstr::Input {
                    regs,
                    in_iidx,
                    clobber,
                    ..
                } => {
                    if !clobber {
                        find_alloc(&mut allocs, i, regs, *in_iidx);
                    }
                }
                RegCnstr::InputOutput { regs, in_iidx, .. }
                | RegCnstr::Cast { regs, in_iidx, .. } => {
                    // In the worst case, it doesn't matter which way round we do this, but trying
                    // to put the vale in the output register makes it clearer when reading the
                    // generated assembly as to what's going on.
                    if !find_alloc(&mut allocs, i, regs, iidx) {
                        find_alloc(&mut allocs, i, regs, *in_iidx);
                    }
                }
                RegCnstr::Output { regs, .. } => {
                    find_alloc(&mut allocs, i, regs, iidx);
                }
                RegCnstr::Temp { .. } | RegCnstr::KeepAlive { .. } => (),
            }
        }

        // If we've got a can_be_same_as_input `Output` then we might have assigned it a register
        // above but still have unassigned `Input`s. See if we can find an `Input` to merge with
        // the `Output`.
        if let Some(output_i) = output_can_be_same_as_input
            && let Some(output_reg) = allocs[output_i]
        {
            for (i, cnstr) in cnstrs.iter().enumerate() {
                if allocs[i].is_some() {
                    continue;
                }
                if let RegCnstr::Input { regs, in_iidx, .. } = cnstr
                    && self.is_used[*in_iidx] <= iidx
                    && regs.contains(&output_reg)
                {
                    allocs[i] = Some(output_reg);
                    break;
                }
            }
        }

        // For any remaining unallocated constraints -- except can_be_same_as_input `Output`s --
        // allocate any valid empty register or, if all registers are used, forcibly allocate any
        // register not used for other allocations.
        for (i, cnstr) in cnstrs.iter().enumerate() {
            if allocs[i].is_some() {
                continue;
            }
            match cnstr {
                RegCnstr::Clobber { .. } => unreachable!(),
                RegCnstr::Input { regs, .. }
                | RegCnstr::InputOutput { regs, .. }
                | RegCnstr::Output {
                    regs,
                    can_be_same_as_input: false,
                    ..
                }
                | RegCnstr::Temp { regs }
                | RegCnstr::Cast { regs, .. } => {
                    allocs[i] = Some(self.find_force_empty_reg(iidx, &allocs, regs));
                }
                RegCnstr::Output {
                    can_be_same_as_input: true,
                    ..
                } => (), // Dealt with a little below.
                RegCnstr::KeepAlive { .. } => allocs[i] = Some(AB::Reg::undefined()),
            }
        }

        // If we've still got an unallocated can_be_same_as_input `Output`, deal with it now.
        if let Some(i) = output_can_be_same_as_input
            && allocs[i].is_none()
        {
            // Are there any inputs which are not used after this instruction? If so, reuse
            // their register.
            for (j, in_cnstr) in cnstrs.iter().enumerate() {
                if let RegCnstr::Input { regs, in_iidx, .. } = in_cnstr
                    && self.is_used[*in_iidx] <= iidx
                    && regs.contains(&allocs[j].unwrap())
                {
                    allocs[i] = allocs[j];
                    break;
                }
            }

            // If we can't reuse an input register, find an empty register.
            if allocs[i].is_none() {
                let RegCnstr::Output { regs, .. } = cnstrs[i] else {
                    panic!()
                };
                allocs[i] = Some(self.find_force_empty_reg(iidx, &allocs, regs));
            }
        }

        Ok(allocs.map(|x| x.unwrap()))
    }

    /// Given a new [RStates], produce a [RegActions] which is a diff telling us how to get from
    /// the current [self.rstates] to the new `src_rstates` (bearing in mind "src" and "dst" are
    /// relative to reverse code generation).
    fn rstate_diff_to_action(
        &mut self,
        src_rstates: &RStates<AB::Reg>,
        ractions: &mut RegActions<AB::Reg>,
    ) {
        'a: for (dst_reg, dst_rstate) in self.rstates.iter().filter(|(_, x)| !x.iidxs.is_empty()) {
            if src_rstates.iidxs(dst_reg) == &dst_rstate.iidxs {
                // It's preferable not to copy values between registers, so first check if we can keep
                // value(s) in the same registers they're already in.
                let max_bitw = iidxs_maxbitw(self.m, self.b, &dst_rstate.iidxs);
                // If the same value(s) can end up in the same registers, we insert a "self copy"
                // so that future parts of the algorithm know we want to use this, but we
                ractions.self_copies.push(RegCopy {
                    bitw: max_bitw,
                    src_reg: dst_reg,
                    src_fill: src_rstates.fill(dst_reg),
                    dst_reg,
                    dst_fill: dst_rstate.fill,
                });
            } else {
                // Try and find cases where we can copy values between different registers, unspilling
                // where that isn't possible.

                for (src_reg, src_rstate) in src_rstates.iter() {
                    if src_reg != dst_reg
                        && !src_rstate.iidxs.is_empty()
                        && src_rstate.iidxs == dst_rstate.iidxs
                    {
                        let max_bitw = iidxs_maxbitw(self.m, self.b, &dst_rstate.iidxs);
                        ractions.distinct_copies.push(RegCopy {
                            bitw: max_bitw,
                            src_reg,
                            src_fill: src_rstate.fill,
                            dst_reg,
                            dst_fill: dst_rstate.fill,
                        });
                        continue 'a;
                    }
                }

                // If the value(s) isn't an existing register, we will need to ensure it is spilt...
                ractions.spills.push(RegSpill {
                    iidxs: dst_rstate.iidxs.clone(),
                });
                ractions.unspills.push(RegUnspill {
                    iidxs: dst_rstate.iidxs.clone(),
                    reg: dst_reg,
                    fill: dst_rstate.fill,
                });
            }
        }
    }

    /// For use in the instruction `iidx`, return a register from `regs` into which a value can be
    /// placed. Ideally this will find a register with no value in it, but it may have to pick a
    /// register which has a value in it, but which isn't going to be used for `allocs`: the caller
    /// will then have to decide whether to spill (etc).
    fn find_force_empty_reg(
        &self,
        iidx: InstIdx,
        allocs: &[Option<AB::Reg>],
        regs: &[AB::Reg],
    ) -> AB::Reg {
        // We continually iterate over `regs` looking for gradually less desirable registers to
        // use.

        // The best case: is there a register with no value in it?
        if let Some(reg) = regs
            .iter()
            .find(|reg| !allocs.contains(&Some(**reg)) && self.rstates.iidxs(**reg).is_empty())
        {
            return *reg;
        }

        // Is there a value which has already been spilt, so at least we won't cause it to be spilt
        // twice?
        if let Some(reg) = regs.iter().find(|reg| {
            !allocs.contains(&Some(**reg))
                && self.rstates.iidxs(**reg).iter().all(|iidx| {
                    matches!(self.istates[*iidx], IState::Stack(_) | IState::StackOff(_))
                })
        }) {
            return *reg;
        }

        // Is there something which isn't needed in `iidx` so we can force it to be spilt?
        if let Some(reg) = regs.iter().find(|reg| {
            !allocs.contains(&Some(**reg)) && self.rstates.iidxs(**reg).iter().all(|x| *x < iidx)
        }) {
            return *reg;
        }

        panic!("Cannot satisfy register constraints {:?}", self.rstates);
    }
}

#[derive(Clone, Debug, PartialEq)]
enum IState {
    /// This value need not be spilt.
    None,
    /// The variable's value will be stored on the stack at `off` bytes from the base pointer.
    /// See [VarLoc::Stack] for more details.
    Stack(u32),
    /// The variable's value is a pointer into the stack. See [VarLoc::StackOff] for more
    /// details.
    StackOff(u32),
}

#[derive(Clone, Debug)]
struct RStates<Reg: RegT> {
    rstate: IndexVec<Reg::RegIdx, RState>,
}

impl<Reg: RegT> RStates<Reg> {
    fn new() -> Self {
        Self {
            rstate: index_vec![RState::default(); Reg::MAX_REGIDX.index()],
        }
    }

    fn iter(&self) -> impl Iterator<Item = (Reg, &RState)> {
        self.rstate
            .iter_enumerated()
            .map(|(regidx, rstate)| (Reg::from_regidx(regidx), rstate))
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = (Reg, &mut RState)> {
        self.rstate
            .iter_mut_enumerated()
            .map(|(regidx, rstate)| (Reg::from_regidx(regidx), rstate))
    }

    fn fill(&self, reg: Reg) -> RegFill {
        self.rstate[reg.regidx()].fill
    }

    fn set_fill(&mut self, reg: Reg, fill: RegFill) {
        self.rstate[reg.regidx()].fill = fill;
    }

    fn iidxs(&self, reg: Reg) -> &SmallVec<[InstIdx; 2]> {
        &self.rstate[reg.regidx()].iidxs
    }

    fn iidxs_mut(&mut self, reg: Reg) -> &mut SmallVec<[InstIdx; 2]> {
        &mut self.rstate[reg.regidx()].iidxs
    }

    fn set_fill_iidxs(&mut self, reg: Reg, fill: RegFill, iidxs: SmallVec<[InstIdx; 2]>) {
        self.rstate[reg.regidx()] = RState { fill, iidxs };
    }
}

#[derive(Clone, Debug)]
struct RState {
    fill: RegFill,
    iidxs: SmallVec<[InstIdx; 2]>,
}

impl Default for RState {
    fn default() -> Self {
        Self {
            fill: RegFill::Undefined,
            iidxs: smallvec![],
        }
    }
}

#[derive(Debug)]
pub(super) struct Snapshot<AB: HirToAsmBackend + ?Sized> {
    istates: IndexVec<InstIdx, IState>,
    rstates: RStates<AB::Reg>,
}

index_vec::define_index_type! {
    pub(super) struct SnapshotIdx = u32;
}

/// An abstraction of a register.
///
/// The register allocator knows almost nothing about registers except the following:
///
///   * Every register can be converted into a `RegIdx`. Registers must be numbered `0..n` where
///     `n` is the maximum number of registers in the system. As this suggests, the allocator needs
///     to consider registers as (sensible!) indexes.
pub(super) trait RegT: Clone + Copy + Debug + Display + PartialEq + Send + Sync {
    /// A register's index. Every register must be convertible to/from this type.
    type RegIdx: Idx;
    /// How many registers are available in this system?
    const MAX_REGIDX: Self::RegIdx;
    /// Return the undefined register for this backend: this will be "allocated" by constraints
    /// such as [RegCnstr::KeepAlive]. This can be any register the backend wants, including a
    /// normal register, but making it a special value makes it impossible for the backend to
    /// accidentally use the resulting register.
    fn undefined() -> Self;
    /// Make a `Reg` from a `RegIdx`.
    fn from_regidx(idx: Self::RegIdx) -> Self;
    /// What is this register's index?
    fn regidx(&self) -> Self::RegIdx;
    /// Is this a caller saved register in this system's standard ABI?
    ///
    /// Note: a system might use multiple ABIs in different places. Currently this function is only
    /// called for the "standard" ABI (e.g. the SysV x64 ABI on Linux/x64). If and when we need to
    /// call it in other contexts, we may have to provide context to this function to allow it to
    /// differentiate which ABI is in use.
    fn is_caller_saved(&self) -> bool;

    /// For testing purposes, return an iterator-like object that can successively produce
    /// valid registers for this backend.
    #[cfg(test)]
    fn iter_test_regs() -> impl TestRegIter<Self>;

    /// For testing purposes (e.g. `arg reg "x"`), return a register corresponding to the
    /// name `s`.
    #[cfg(test)]
    fn from_str(s: &str) -> Option<Self>;
}

#[cfg(test)]
pub(super) trait TestRegIter<Reg: RegT> {
    /// Return a register which is suitable to hold instances of `ty`. The backend has freedom
    /// to interpret this: it might place all types in the same kinds of registers, or it may
    /// do things like differentiate general-purpose and floating-point registers.
    fn next_reg(&mut self, ty: &Ty) -> Option<Reg>;
}

/// An unordered set of [VarLoc]s.
///
/// Note: this happens to be stored as a [SmallVec] because that is an efficient use of space, but
/// nothing should be inferred about the order of [VarLoc]s from that.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct VarLocs<Reg: RegT> {
    raw: SmallVec<[VarLoc<Reg>; 1]>,
}

impl<Reg: RegT> VarLocs<Reg> {
    pub(super) fn new(vlocs: SmallVec<[VarLoc<Reg>; 1]>) -> Self {
        Self { raw: vlocs }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.raw.len()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &VarLoc<Reg>> {
        self.raw.iter()
    }
}

/// Where an AOT-relevant variable is stored. Note: in general, a given variable may be stored in
/// more than one place so [VarLocs] should be used for such cases.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum VarLoc<Reg> {
    /// The variable's value is stored on the stack at `off` bytes from the base pointer. Whether
    /// `off` is "above" or "below" the base pointer is system dependent.
    Stack(u32),
    /// The variable's value is a pointer into the stack. This is an optimisation (what LLVM
    /// stackmaps call a `Direct`) that allows us to turn a seemingly arbitrary value into a
    /// semi-constant. The pointer is `off` bytes from the base pointer. Whether `off` is "above"
    /// or "below" the base pointer is system dependent.
    StackOff(u32),
    /// The variable's value is stored in a register.
    Reg(Reg),
    /// The variable's value is a constant.
    Const(ConstKind),
}

impl<Reg: Display> Display for VarLoc<Reg> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VarLoc::Stack(x) => write!(f, "Stack(0x{x:X})"),
            VarLoc::StackOff(x) => write!(f, "StackOff(0x{x:X})"),
            VarLoc::Reg(x) => write!(f, "Reg(\"{x}\")"),
            VarLoc::Const(x) => match x {
                ConstKind::Double(x) => write!(f, "{x}"),
                ConstKind::Float(x) => write!(f, "{x}"),
                ConstKind::Int(x) => write!(f, "{x}"),
                ConstKind::Ptr(x) => write!(f, "0x{x:X}"),
            },
        }
    }
}

/// A register constraint. Each constraint leads to a single register being returned. Note: in some
/// situations (see the individual constraints), multiple constraints might return the same
/// register.
#[derive(Debug, PartialEq)]
pub(super) enum RegCnstr<'a, Reg: RegT> {
    /// This instruction clobbers `reg`.
    Clobber { reg: Reg },
    /// Make sure that `in_iidx` is loaded into a register drawn from `regs`, with its upper bits
    /// matching fill `in_fill`. If `clobber` is true, then the value in the register will be
    /// treated as clobbered on exit.
    Input {
        in_iidx: InstIdx,
        in_fill: RegCnstrFill,
        regs: &'a [Reg],
        clobber: bool,
    },
    /// Make sure that `in_iidx` is loaded into a register drawn from `regs`, with its upper bits
    /// matching fill `in_fill`; the result of the instruction will be in the same register with
    /// its upper bits matching fill `out_fill`.
    InputOutput {
        in_iidx: InstIdx,
        in_fill: RegCnstrFill,
        out_fill: RegCnstrFill,
        regs: &'a [Reg],
    },
    /// The result of the instruction will be in a register drawn from `regs` with its upper bits
    /// matching fill `out_fill`. If `can_be_same_as_input` is true, then the allocator may
    /// optionally return a register that is also used for an input (in such a case, the input will
    /// implicitly be considered clobbered).
    Output {
        out_fill: RegCnstrFill,
        regs: &'a [Reg],
        can_be_same_as_input: bool,
    },
    /// Cast the value in `in_iidx` to have the fill `out_fill`, ensuring the value is in `regs`.
    /// This is similar in effect to [Self::InputOutput] with the significant difference that the
    /// register allocator can sometimes merge the results of casts into the same register as
    /// existing values. Note that setting `out_fill` to [RegCnstrFill::AnyOf] will lead to
    /// undefined behaviour.
    Cast {
        in_iidx: InstIdx,
        out_fill: RegCnstrFill,
        regs: &'a [Reg],
    },
    /// A temporary register drawn from `regs` that the instruction will clobber.
    Temp { regs: &'a [Reg] },
    /// Keep alive the values in `InstIdx` but do not allocate a register for them. Returns
    /// `Reg::Undefined`. Intended only for guards.
    KeepAlive { iidxs: &'a [InstIdx] },
}

/// What should the *fill bits* of a register be set to?
///
/// See the description of [RegFill] for the definition of fill bits.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum RegCnstrFill {
    /// We can accept any of the fill bits set in [AnyOfFill].
    ///
    /// Note: currently `AnyOf` can only be used in the `out_fill` of [RegCnstr::InputOutput] and
    /// [RegCnstr::Output]. Using it elsewhere will result in undefined behaviour.
    AnyOf(AnyOfFill),
    /// We do not care what the fill bits are set to.
    Undefined,
    /// We want the fill bits to zero extend the value.
    Zeroed,
    /// We want the fill bits to sign extend the value.
    Signed,
}

impl RegCnstrFill {
    /// Create a [RegCnstrFill] from a [RegFill].
    fn from_regfill(rf: RegFill) -> Self {
        match rf {
            RegFill::Undefined => RegCnstrFill::Undefined,
            RegFill::Zeroed => RegCnstrFill::Zeroed,
            RegFill::Signed => RegCnstrFill::Signed,
        }
    }
}

// The three constants below define the bits in the [AnyOfFill] bitfield.
const ANYOFFILL_UNDEFINED: u8 = 1;
const ANYOFFILL_SIGNED: u8 = 2;
const ANYOFFILL_ZEROED: u8 = 4;

/// A bitfield representing the valid fills an instruction can accept for
/// [RegAlloc::alloc_with_fills].
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AnyOfFill(u8);

impl AnyOfFill {
    /// Create a blank [AnyOfFill] i.e. one that does not accept any fills.
    pub(super) const fn new() -> Self {
        Self(0)
    }

    /// Construct an [AnyOfFill] from a [RegFill].
    const fn from_regfill(fill: RegFill) -> Self {
        match fill {
            RegFill::Undefined => Self::new().with_undefined(),
            RegFill::Zeroed => Self::new().with_zeroed(),
            RegFill::Signed => Self::new().with_signed(),
        }
    }

    /// Return the current [AnyOfFill] extended with the accepted of [RegFill::Undefined].
    pub(super) const fn with_undefined(self) -> Self {
        Self(self.0 | ANYOFFILL_UNDEFINED)
    }

    /// Return the current [AnyOfFill] extended with the accepted of [RegFill::Signed].
    pub(super) const fn with_signed(self) -> Self {
        Self(self.0 | ANYOFFILL_SIGNED)
    }

    /// Return the current [AnyOfFill] extended with the accepted of [RegFill::Zeroed].
    pub(super) const fn with_zeroed(self) -> Self {
        Self(self.0 | ANYOFFILL_ZEROED)
    }

    /// Does this [AnyOfFill] intersect with `other` i.e. do they share at least one set bit in
    /// common?
    const fn intersects_with(&self, other: &Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Can `self` accept [RegFill::Undefined]?
    const fn has_undefined(&self) -> bool {
        (self.0 & ANYOFFILL_UNDEFINED) != 0
    }
}

/// What are the *fill bits* of a register be set to?
///
/// Fill bits are defined as follows:
///
///   * For normal values, we assume they may end up in a `n`-bit register: any bits between the
///     `bitw` of the type and `n`-bits are fill bits. For max-bit values, the fill bits are
///     ignored, and can be set to any value.
///
///   * For floating point values, we assume that 32 bit floats and 64 bit doubles are not
///     intermixed. Fill bits are thus irrelevant in this case.
///
///   * We do not currently support "non-normal / non-float" values (e.g. vector values) and will
///     have to think about those at a later point.
///
/// For example, if a 16 bit value is stored in a 64 bit value, we may know for sure that the upper
/// 48 bits are set to zero, or they sign extend the 16 bit value --- or we may have no idea!
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum RegFill {
    /// We do not know what the fill bits are set to / we do not care what the fill bits are set
    /// to.
    Undefined,
    /// The fill bits zero extend the value / we want the fill bits to zero extend the value.
    Zeroed,
    /// The fill bits sign extend the value / we want the fill bits to sign extend the value.
    Signed,
}

impl RegFill {
    /// Create a [RegFill] from a [RegCnstrFill].
    fn from_regcnstrfill(rcf: &RegCnstrFill) -> Self {
        match rcf {
            RegCnstrFill::AnyOf(_) => panic!(),
            RegCnstrFill::Undefined => RegFill::Undefined,
            RegCnstrFill::Zeroed => RegFill::Zeroed,
            RegCnstrFill::Signed => RegFill::Signed,
        }
    }
}

#[derive(Debug)]
struct RegActions<Reg: RegT> {
    /// An unordered set of values that need to be unspilt.
    unspills: Vec<RegUnspill<Reg>>,
    /// An unordered set of self-copies (i.e. where `src_reg` and `dst_reg` are the same).
    self_copies: Vec<RegCopy<Reg>>,
    /// Before `break_regactions_cycles` this will be an unordered set of register copies between
    /// distinct registers (i.e. where `src_reg` and `dst_reg` are different).
    ///
    /// After `break_regactions_cycles` this will be an ordered set of register copies stored in
    /// reverse-order.
    distinct_copies: Vec<RegCopy<Reg>>,
    /// An unordered set of instructions .
    spills: Vec<RegSpill>,
}

impl<Reg: RegT> RegActions<Reg> {
    fn new() -> Self {
        Self {
            unspills: Vec::new(),
            self_copies: Vec::new(),
            distinct_copies: Vec::new(),
            spills: Vec::new(),
        }
    }
}

/// Unspill `InstIdx` into `Reg` with fill `RegExt`. Note: by definition, spilled values are stored
/// zero extended.
#[derive(Debug)]
struct RegUnspill<Reg: RegT> {
    iidxs: SmallVec<[InstIdx; 2]>,
    reg: Reg,
    fill: RegFill,
}

#[derive(Clone, Debug)]
struct RegCopy<Reg: RegT> {
    bitw: u32,
    src_reg: Reg,
    src_fill: RegFill,
    dst_reg: Reg,
    dst_fill: RegFill,
}

/// Specify that instruction values in `iidx`s currently in `reg` will need to be marked as spilt.
#[derive(Debug)]
struct RegSpill {
    iidxs: SmallVec<[InstIdx; 2]>,
}

/// Return the bit width of the widest instruction in `iidxs`.
fn iidxs_maxbitw<Reg: RegT>(m: &Mod<Reg>, b: &Block, iidxs: &[InstIdx]) -> u32 {
    iidxs
        .iter()
        .map(|iidx| b.inst_bitw(m, *iidx))
        .max()
        .unwrap()
}

/// Iterate over the maximum bitwidth members of `iidxs`.
///
/// # Panics
///
/// If `iidxs` is empty.
fn iter_maxbitw_iidxs<Reg: RegT>(m: &Mod<Reg>, b: &Block, iidxs: &[InstIdx]) -> (u32, MaxBitIter) {
    assert!(!iidxs.is_empty());
    let max_bitw = iidxs_maxbitw(m, b, iidxs);
    let iidxs = if iidxs.len() == 1 {
        // By far the most common case is that there's a single `iidx`, at which point it is by
        // definition the complete set of maximum bit width `iidx`s.
        smallvec![iidxs[0]]
    } else {
        // In the uncommon case, we need to iterate over everything again.
        iidxs
            .iter()
            .filter(move |iidx| b.inst_bitw(m, **iidx) == max_bitw)
            .cloned()
            .collect::<SmallVec<_>>()
    };
    (max_bitw, MaxBitIter { iidxs, i: 0 })
}

#[derive(Clone)]
struct MaxBitIter {
    iidxs: SmallVec<[InstIdx; 1]>,
    i: usize,
}

impl Iterator for MaxBitIter {
    type Item = InstIdx;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.iidxs.len() {
            let v = self.iidxs[self.i];
            self.i += 1;
            Some(v)
        } else {
            None
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::{
        compile::{
            j2::{
                codebuf::ExeCodeBuf, compiled_trace::J2CompiledTrace, hir::Mod, hir::*,
                hir_parser::str_to_mod, hir_to_asm::*,
            },
            jitc_yk::aot_ir,
        },
        location::{HotLocation, HotLocationKind},
        mt::TraceId,
    };
    use fm::{FMBuilder, FMatcher};

    use lazy_static::lazy_static;
    use parking_lot::Mutex;
    use regex::Regex;
    use std::sync::Arc;
    use strum::{Display, EnumCount, FromRepr};

    #[test]
    fn any_of_fill() {
        assert_eq!(AnyOfFill::new().0, 0);
        assert_ne!(AnyOfFill::new().with_undefined().0, 0);
        assert_ne!(AnyOfFill::new().with_signed().0, 0);
        assert_ne!(AnyOfFill::new().with_zeroed().0, 0);
        assert_ne!(
            AnyOfFill::new().with_undefined(),
            AnyOfFill::new().with_signed()
        );
        assert_ne!(
            AnyOfFill::new().with_undefined(),
            AnyOfFill::new().with_zeroed()
        );
        assert_ne!(
            AnyOfFill::new().with_signed(),
            AnyOfFill::new().with_zeroed()
        );

        assert!(
            AnyOfFill::new()
                .with_undefined()
                .intersects_with(&AnyOfFill::new().with_undefined())
        );
        assert!(
            AnyOfFill::new()
                .with_signed()
                .intersects_with(&AnyOfFill::new().with_signed())
        );
        assert!(
            AnyOfFill::new()
                .with_zeroed()
                .intersects_with(&AnyOfFill::new().with_zeroed())
        );
        assert!(
            !AnyOfFill::new()
                .with_undefined()
                .intersects_with(&AnyOfFill::new().with_signed())
        );
        assert!(
            !AnyOfFill::new()
                .with_undefined()
                .intersects_with(&AnyOfFill::new().with_zeroed())
        );
        assert!(
            !AnyOfFill::new()
                .with_signed()
                .intersects_with(&AnyOfFill::new().with_zeroed())
        );

        assert!(AnyOfFill::new().with_undefined().has_undefined());
        assert!(!AnyOfFill::new().with_signed().has_undefined());
        assert!(!AnyOfFill::new().with_zeroed().has_undefined());
    }

    #[derive(Copy, Clone, Debug, Display, EnumCount, FromRepr, PartialEq)]
    #[repr(u8)]
    pub(crate) enum TestReg {
        GPR0,
        GPR1,
        GPR2,
        GPR3,
        FP0,
        FP1,
        FP2,
        FP3,
        Undefined,
    }

    const FP_REGS: [TestReg; 4] = [TestReg::FP0, TestReg::FP1, TestReg::FP2, TestReg::FP3];
    const GP_REGS: [TestReg; 4] = [TestReg::GPR0, TestReg::GPR1, TestReg::GPR2, TestReg::GPR3];

    impl RegT for TestReg {
        type RegIdx = TestRegIdx;
        const MAX_REGIDX: TestRegIdx = TestRegIdx::from_usize_unchecked(TestReg::COUNT);

        fn undefined() -> Self {
            TestReg::Undefined
        }

        fn from_regidx(idx: Self::RegIdx) -> Self {
            TestReg::from_repr(idx.raw()).unwrap()
        }

        fn regidx(&self) -> Self::RegIdx {
            TestRegIdx::from(*self as usize)
        }

        fn is_caller_saved(&self) -> bool {
            todo!()
        }

        fn iter_test_regs() -> impl TestRegIter<Self> {
            TestRegTestIter::new()
        }

        fn from_str(s: &str) -> Option<Self> {
            match s {
                "GPR0" => Some(Self::GPR0),
                "GPR1" => Some(Self::GPR1),
                "GPR2" => Some(Self::GPR2),
                "GPR3" => Some(Self::GPR3),
                _ => None,
            }
        }
    }

    index_vec::define_index_type! {
        pub(crate) struct TestRegIdx = u8;
    }

    struct TestRegTestIter<Reg> {
        fp_regs: Box<dyn Iterator<Item = Reg>>,
        gp_regs: Box<dyn Iterator<Item = Reg>>,
    }

    impl TestRegTestIter<TestReg> {
        fn new() -> Self {
            Self {
                fp_regs: Box::new(
                    [TestReg::FP0, TestReg::FP1, TestReg::FP2, TestReg::FP3]
                        .iter()
                        .cloned(),
                ),
                gp_regs: Box::new(
                    [TestReg::GPR0, TestReg::GPR1, TestReg::GPR2, TestReg::GPR3]
                        .iter()
                        .cloned(),
                ),
            }
        }
    }

    impl TestRegIter<TestReg> for TestRegTestIter<TestReg> {
        fn next_reg(&mut self, ty: &Ty) -> Option<TestReg> {
            match ty {
                Ty::Double | Ty::Float => self.fp_regs.next(),
                Ty::Func(_func_ty) => todo!(),
                Ty::Int(bitw) => {
                    if *bitw <= 64 {
                        self.gp_regs.next()
                    } else {
                        todo!()
                    }
                }
                Ty::Ptr(addrspace) => {
                    assert_eq!(*addrspace, 0);
                    self.gp_regs.next()
                }
                Ty::Void => todo!(),
            }
        }
    }

    struct TestHirToAsm<'a> {
        m: &'a Mod<TestReg>,
        ra_log: Vec<String>,
    }

    impl<'a> TestHirToAsm<'a> {
        fn new(m: &'a Mod<TestReg>) -> Self {
            Self {
                m,
                ra_log: Vec::new(),
            }
        }
    }

    impl<'a> HirToAsmBackend for TestHirToAsm<'a> {
        type Label = TestLabelIdx;
        type Reg = TestReg;
        type BuildTest = String;

        fn smp_to_vloc(_smp_locs: &SmallVec<[yksmp::Location; 1]>) -> VarLocs<Self::Reg> {
            todo!()
        }

        fn thread_local_off(_addr: *const std::ffi::c_void) -> u32 {
            todo!()
        }

        fn build_exe(
            self,
            _log: bool,
            _labels: &[Self::Label],
        ) -> Result<
            (
                ExeCodeBuf,
                IndexVec<
                    GuardRestoreIdx,
                    crate::compile::j2::compiled_trace::J2CompiledGuard<Self::Reg>,
                >,
                Option<String>,
                Vec<usize>,
            ),
            CompilationError,
        > {
            todo!()
        }

        fn build_test(self, _labels: &[Self::Label]) -> Self::BuildTest {
            self.ra_log.join("\n")
        }

        fn iter_possible_regs(&self, b: &Block, iidx: InstIdx) -> impl Iterator<Item = Self::Reg> {
            match b.inst_ty(self.m, iidx) {
                Ty::Double | Ty::Float => FP_REGS.iter().cloned(),
                Ty::Func(_func_ty) => todo!(),
                Ty::Int(_) | Ty::Ptr(_) => GP_REGS.iter().cloned(),
                Ty::Void => todo!(),
            }
        }

        fn log(&mut self, _s: String) {}

        fn const_needs_tmp_reg(
            &self,
            _reg: Self::Reg,
            c: &ConstKind,
        ) -> Option<impl Iterator<Item = Self::Reg>> {
            if let ConstKind::Double(_) | ConstKind::Float(_) = c {
                Some(GP_REGS.iter().cloned())
            } else {
                None
            }
        }

        fn move_const(
            &mut self,
            reg: Self::Reg,
            tmp_reg: Option<Self::Reg>,
            tgt_bitw: u32,
            fill: RegFill,
            c: &ConstKind,
        ) -> Result<(), CompilationError> {
            self.ra_log.push(format!(
                "const {reg:?} tmp_reg={tmp_reg:?} tgt_bitw={tgt_bitw} fill={fill:?} {c:?}"
            ));
            Ok(())
        }

        fn move_stackoff(
            &mut self,
            _reg: Self::Reg,
            _stack_off: u32,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn arrange_fill(
            &mut self,
            reg: Self::Reg,
            src_fill: RegFill,
            dst_bitw: u32,
            dst_fill: RegFill,
        ) {
            self.ra_log.push(format!(
                "arrange_fill {reg:?} from={src_fill:?} dst_bitw={dst_bitw} to={dst_fill:?}"
            ));
        }

        fn copy_reg(
            &mut self,
            from_reg: Self::Reg,
            to_reg: Self::Reg,
        ) -> Result<(), CompilationError> {
            self.ra_log
                .push(format!("copy_reg from={from_reg:?} to={to_reg:?}"));
            Ok(())
        }

        fn align_spill(&self, stack_off: u32, bitw: u32) -> u32 {
            stack_off + (bitw / 8).next_multiple_of(8)
        }

        fn spill(
            &mut self,
            reg: Self::Reg,
            in_fill: RegFill,
            stack_off: u32,
            bitw: u32,
        ) -> Result<(), CompilationError> {
            self.ra_log.push(format!(
                "spill {reg:?} {in_fill:?} stack_off={stack_off} bitw={bitw}"
            ));
            Ok(())
        }

        fn unspill(
            &mut self,
            stack_off: u32,
            reg: Self::Reg,
            out_fill: RegFill,
            bitw: u32,
        ) -> Result<(), CompilationError> {
            self.ra_log.push(format!(
                "unspill stack_off={stack_off} {reg:?} {out_fill:?} bitw={bitw}"
            ));
            Ok(())
        }

        fn move_stack_val(
            &mut self,
            _bitw: u32,
            _src_stack_off: u32,
            _dst_stack_off: u32,
            _reg: Self::Reg,
        ) {
            todo!()
        }

        fn coupler_trace_end(
            &mut self,
            _tgt_ctr: &Arc<J2CompiledTrace<Self::Reg>>,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn coupler_trace_start(
            &mut self,
            _stack_off: u32,
        ) -> Result<Self::Label, CompilationError> {
            todo!();
        }

        fn loop_trace_end(&mut self) -> Result<Self::Label, CompilationError> {
            todo!()
        }

        fn loop_trace_start(&mut self, _iter0_label: Self::Label, _stack_off: u32) {}

        fn side_trace_end(
            &mut self,
            _ctr: &std::sync::Arc<J2CompiledTrace<Self::Reg>>,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn side_trace_start(&mut self, _stack_off: u32) {}

        fn guard_end(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _trid: crate::mt::TraceId,
            _gridx: GuardRestoreIdx,
        ) -> Result<Self::Label, CompilationError> {
            Ok(TestLabelIdx::new(0))
        }

        fn guard_completed(
            &mut self,
            _start_label: Self::Label,
            _patch_label: Self::Label,
            _stack_off: u32,
            _bid: aot_ir::BBlockId,
            _deopt_frames: SmallVec<[crate::compile::j2::compiled_trace::DeoptFrame<Self::Reg>; 1]>,
            _switch: Option<crate::compile::j2::hir::Switch>,
        ) {
        }

        fn i_abs(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Abs,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_add(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            Add {
                tyidx: _, lhs, rhs, ..
            }: &Add,
        ) -> Result<(), CompilationError> {
            let [lhsr, rhsr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::InputOutput {
                        in_iidx: *lhs,
                        in_fill: RegCnstrFill::Zeroed,
                        out_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                    },
                    RegCnstr::Input {
                        in_iidx: *rhs,
                        in_fill: RegCnstrFill::Zeroed,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                ],
            )?;
            self.ra_log
                .push(format!("alloc %{iidx:?} {lhsr:?} {rhsr:?}"));
            Ok(())
        }

        fn i_and(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::And,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_ashr(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::AShr,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_call(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Call,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_ctpop(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::CtPop,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_dynptradd(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::DynPtrAdd,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_fadd(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            FAdd { lhs, rhs, .. }: &FAdd,
        ) -> Result<(), CompilationError> {
            let [lhsr, rhsr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::InputOutput {
                        in_iidx: *lhs,
                        in_fill: RegCnstrFill::Zeroed,
                        out_fill: RegCnstrFill::Undefined,
                        regs: &FP_REGS,
                    },
                    RegCnstr::Input {
                        in_iidx: *rhs,
                        in_fill: RegCnstrFill::Zeroed,
                        regs: &FP_REGS,
                        clobber: false,
                    },
                ],
            )?;
            self.ra_log
                .push(format!("alloc %{iidx:?} {lhsr:?} {rhsr:?}"));
            Ok(())
        }

        fn i_fcmp(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _fcmp: &FCmp,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_fdiv(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _fadd: &FDiv,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_floor(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &Floor,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_fneg(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &FNeg,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_fmul(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _fmul: &FMul,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_fsub(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _fadd: &FSub,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_fpext(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &FPExt,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_fptosi(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &FPToSI,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_guard(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            Guard {
                expect: _,
                cond,
                entry_vars,
                ..
            }: &Guard,
        ) -> Result<Self::Label, CompilationError> {
            let [cndr, _] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *cond,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::KeepAlive { iidxs: entry_vars },
                ],
            )?;

            self.ra_log.push(format!("alloc %{iidx:?} {cndr:?}"));
            Ok(TestLabelIdx::new(0))
        }

        fn i_icmp(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            ICmp {
                pred: _,
                lhs,
                rhs,
                samesign: _,
            }: &ICmp,
        ) -> Result<(), CompilationError> {
            let [lhsr, rhsr, outr] = ra.alloc(
                self,
                iidx,
                [
                    RegCnstr::Input {
                        in_iidx: *lhs,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Input {
                        in_iidx: *rhs,
                        in_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        clobber: false,
                    },
                    RegCnstr::Output {
                        out_fill: RegCnstrFill::Undefined,
                        regs: &GP_REGS,
                        can_be_same_as_input: true,
                    },
                ],
            )?;
            self.ra_log
                .push(format!("alloc %{iidx:?} {lhsr:?} {rhsr:?} {outr:?}"));
            Ok(())
        }

        fn i_inttoptr(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::IntToPtr,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_load(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Load,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_lshr(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::LShr,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_memcpy(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &MemCpy,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_memset(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &MemSet,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_mul(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Mul,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_or(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Or,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_ptradd(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::PtrAdd,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_ptrtoint(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::PtrToInt,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_return(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &Return,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_sdiv(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &SDiv,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_select(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Select,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_sext(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::SExt,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_shl(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Shl,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_sitofp(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &SIToFP,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_smax(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &SMax,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_smin(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &SMin,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_srem(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &SRem,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_store(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Store,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_sub(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::Sub,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_threadlocal(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _tl_off: u32,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_trunc(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            Trunc { val, .. }: &crate::compile::j2::hir::Trunc,
        ) -> Result<(), CompilationError> {
            let [reg] = ra.alloc(
                self,
                iidx,
                [RegCnstr::Cast {
                    in_iidx: *val,
                    out_fill: RegCnstrFill::Undefined,
                    regs: &GP_REGS,
                }],
            )?;
            self.ra_log.push(format!("trunc %{iidx:?} {reg:?}"));
            Ok(())
        }

        fn i_udiv(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &crate::compile::j2::hir::UDiv,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_uitofp(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &UIToFP,
        ) -> Result<(), CompilationError> {
            todo!();
        }

        fn i_xor(
            &mut self,
            _ra: &mut RegAlloc<Self>,
            _b: &Block,
            _iidx: InstIdx,
            _inst: &Xor,
        ) -> Result<(), CompilationError> {
            todo!()
        }

        fn i_zext(
            &mut self,
            ra: &mut RegAlloc<Self>,
            _b: &Block,
            iidx: InstIdx,
            ZExt { val, .. }: &crate::compile::j2::hir::ZExt,
        ) -> Result<(), CompilationError> {
            let [reg] = ra.alloc(
                self,
                iidx,
                [RegCnstr::Cast {
                    in_iidx: *val,
                    out_fill: RegCnstrFill::Zeroed,
                    regs: &GP_REGS,
                }],
            )?;
            self.ra_log.push(format!("zext %{iidx:?} {reg:?}"));
            Ok(())
        }
    }

    index_vec::define_index_type! {
        struct TestLabelIdx = u32;
        IMPL_RAW_CONVERSIONS = true;
    }

    lazy_static! {
        /// Use `{{name}}` to match non-literal strings in tests.
        static ref PTN_RE: Regex = {
            Regex::new(r"\{\{.+?\}\}").unwrap()
        };

        static ref PTN_RE_IGNORE: Regex = {
            Regex::new(r"\{\{_}\}").unwrap()
        };

        static ref TEXT_RE: Regex = {
            Regex::new(r"[a-zA-Z0-9\._]+").unwrap()
        };
    }

    fn fmatcher(ptn: &str) -> FMatcher<'_> {
        FMBuilder::new(ptn)
            .unwrap()
            .name_matcher(PTN_RE.clone(), TEXT_RE.clone())
            .name_matcher_ignore(PTN_RE_IGNORE.clone(), TEXT_RE.clone())
            .build()
            .unwrap()
    }

    /// Enable simple tests of the register allocator.
    ///
    /// This function takes a module `s` in and runs the register allocator on it with our "test"
    /// backend above. It then runs each line of the log through `log_filter`, only keeping lines
    /// where `log_filter` returns `true`. It recombines the log and then matches it against the
    /// [fm] pattern `ptn`.
    fn build_and_test<F>(s: &str, log_filter: F, ptns: &[&str])
    where
        F: Fn(&str) -> bool,
    {
        let m = str_to_mod::<TestReg>(s);

        let hl = Arc::new(Mutex::new(HotLocation {
            kind: HotLocationKind::Tracing(TraceId::testing()),
            tracecompilation_errors: 0,
            #[cfg(feature = "ykd")]
            debug_str: None,
        }));

        let be = TestHirToAsm::new(&m);
        let log = HirToAsm::new(&m, hl, be).build_test().unwrap();
        let log = log
            .lines()
            .filter(|s| log_filter(s))
            .collect::<Vec<_>>()
            .join("\n");
        let mut failures = Vec::with_capacity(ptns.len());
        for ptn in ptns {
            match fmatcher(ptn).matches(&log) {
                Ok(_) => return,
                Err(e) => failures.push(format!("{e}")),
            }
        }

        panic!("{}", failures.join("\n\n"));
    }

    #[test]
    fn simple() {
        build_and_test(
            r#"
          %0: i8 = arg [reg "GPR0"]
          %1: i8 = arg [reg "GPR1"]
          %2: i8 = add %0, %1
          blackbox %2
        "#,
            |_| true,
            &["
          alloc %2 GPR0 GPR1
          arrange_fill GPR0 from=Undefined dst_bitw=8 to=Zeroed
          arrange_fill GPR1 from=Undefined dst_bitw=8 to=Zeroed
        "],
        );
    }

    #[test]
    fn cycles() {
        build_and_test(
            r#"
          %0: i8 = arg [reg "GPR1"]
          %1: i8 = arg [reg "GPR0"]
          %2: i8 = add %0, %1
          blackbox %2
        "#,
            |s| !s.starts_with("arrange_fill"),
            &[
                "
          alloc %2 GPR0 GPR1
          unspill stack_off=8 GPR1 Undefined bitw=8
          copy_reg from=GPR1 to=GPR0
          spill GPR0 Undefined stack_off=8 bitw=8
        ",
                "
          alloc %2 GPR0 GPR1
          unspill stack_off=8 GPR0 Undefined bitw=8
          copy_reg from=GPR0 to=GPR1
          spill GPR1 Undefined stack_off=8 bitw=8
        ",
            ],
        );

        build_and_test(
            r#"
          %0: i8 = arg [reg "GPR0"]
          %1: i8 = arg [reg "GPR1"]
          %2: i8 = arg [reg "GPR2"]
          %3: i8 = add %0, %1
          %4: i8 = add %2, %3
          blackbox %4
        "#,
            |s| !s.starts_with("arrange_fill"),
            &[
                "
          alloc %4 GPR0 GPR1
          alloc %3 GPR1 GPR2
          unspill stack_off=8 GPR0 Undefined bitw=8
          copy_reg from=GPR0 to=GPR1
          copy_reg from=GPR1 to=GPR2
          spill GPR2 Undefined stack_off=8 bitw=8
        ",
                "
          alloc %4 GPR0 GPR1
          alloc %3 GPR1 GPR2
          unspill stack_off=8 GPR1 Undefined bitw=8
          copy_reg from=GPR1 to=GPR2
          copy_reg from=GPR2 to=GPR0
          spill GPR0 Undefined stack_off=8 bitw=8
        ",
            ],
        );

        build_and_test(
            r#"
          %0: i8 = arg [reg "GPR3"]
          %1: i8 = arg [reg "GPR2"]
          %2: i8 = arg [reg "GPR1"]
          %3: i8 = arg [reg "GPR0"]
          %4: i8 = add %0, %1
          %5: i8 = add %4, %2
          %6: i8 = add %5, %3
          %7: i8 = add %5, %3
          blackbox %6
          blackbox %7
        "#,
            |s| !s.starts_with("arrange_fill"),
            &[
                "
          alloc %7 GPR0 GPR1
          unspill stack_off=8 GPR0 Zeroed bitw=8
          alloc %6 GPR0 GPR1
          spill GPR0 Undefined stack_off=8 bitw=8
          alloc %5 GPR0 GPR2
          alloc %4 GPR0 GPR3
          unspill stack_off=16 GPR0 Undefined bitw=8
          copy_reg from=GPR0 to=GPR1
          copy_reg from=GPR1 to=GPR2
          copy_reg from=GPR2 to=GPR3
          spill GPR3 Undefined stack_off=16 bitw=8
        ",
                "
          alloc %7 GPR0 GPR1
          unspill stack_off=8 GPR0 Zeroed bitw=8
          alloc %6 GPR0 GPR1
          spill add GPR0 Undefined stack_off=8 bitw=8
          alloc %5 GPR0 GPR2
          alloc %4 GPR0 GPR3
          unspill stack_off=16 GPR1 Undefined bitw=8
          copy_reg from=GPR1 to=GPR2
          copy_reg from=GPR2 to=GPR3
          copy_reg from=GPR3 to=GPR0
          spill GPR0 Undefined stack_off=16 bitw=8
        ",
            ],
        );
    }

    #[test]
    fn toposort() {
        let m = str_to_mod::<TestReg>("");
        let b = match &m.kind {
            ModKind::Test { block, .. } => block,
            _ => panic!(),
        };

        // Encoding all the possible outcomes below is annoying, particularly if the current code
        // can't generate them. Some of these tests therefore over-specialise on the current
        // output: they may need to be adjusted if the algorithm produces different output.

        // A direct cycle which must be broken.
        let ra = RegAlloc::<TestHirToAsm>::new(&m, b, 0);
        let mut ractions = RegActions {
            unspills: Vec::new(),
            self_copies: Vec::new(),
            distinct_copies: vec![
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR0,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR1,
                    dst_fill: RegFill::Undefined,
                },
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR1,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR0,
                    dst_fill: RegFill::Undefined,
                },
            ],
            spills: Vec::new(),
        };
        ra.toposort_distinct_copies(&mut ractions).unwrap();
        assert_eq!(ractions.spills.len(), 1);
        assert_matches!(
            ractions.distinct_copies.as_slice(),
            &[RegCopy {
                src_reg: TestReg::GPR0,
                dst_reg: TestReg::GPR1,
                ..
            }]
        );

        // A direct cycle which must be broken
        let ra = RegAlloc::<TestHirToAsm>::new(&m, b, 0);
        let mut ractions = RegActions {
            unspills: Vec::new(),
            self_copies: Vec::new(),
            distinct_copies: vec![
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR0,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR1,
                    dst_fill: RegFill::Undefined,
                },
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR1,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR2,
                    dst_fill: RegFill::Undefined,
                },
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR2,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR0,
                    dst_fill: RegFill::Undefined,
                },
            ],
            spills: Vec::new(),
        };
        ra.toposort_distinct_copies(&mut ractions).unwrap();
        assert_eq!(ractions.spills.len(), 1);
        assert_matches!(
            ractions.distinct_copies.as_slice(),
            &[
                RegCopy {
                    src_reg: TestReg::GPR0,
                    dst_reg: TestReg::GPR1,
                    ..
                },
                RegCopy {
                    src_reg: TestReg::GPR1,
                    dst_reg: TestReg::GPR2,
                    ..
                }
            ]
        );

        // A non-cycle, but which must be carefully ordered to avoid overwritings.
        let ra = RegAlloc::<TestHirToAsm>::new(&m, b, 0);
        let mut ractions = RegActions {
            unspills: Vec::new(),
            self_copies: Vec::new(),
            distinct_copies: vec![
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR0,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR1,
                    dst_fill: RegFill::Undefined,
                },
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR2,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR0,
                    dst_fill: RegFill::Undefined,
                },
                RegCopy {
                    bitw: 8,
                    src_reg: TestReg::GPR0,
                    src_fill: RegFill::Undefined,
                    dst_reg: TestReg::GPR3,
                    dst_fill: RegFill::Undefined,
                },
            ],
            spills: Vec::new(),
        };
        ra.toposort_distinct_copies(&mut ractions).unwrap();
        assert_eq!(ractions.spills.len(), 0);
        assert_matches!(
            ractions.distinct_copies.as_slice(),
            &[
                RegCopy {
                    src_reg: TestReg::GPR2,
                    dst_reg: TestReg::GPR0,
                    ..
                },
                RegCopy {
                    src_reg: TestReg::GPR0,
                    dst_reg: TestReg::GPR3,
                    ..
                },
                RegCopy {
                    src_reg: TestReg::GPR0,
                    dst_reg: TestReg::GPR1,
                    ..
                }
            ]
        );
    }

    #[test]
    fn arrange_fills_once() {
        build_and_test(
            r#"
          %0: i8 = arg [reg "GPR0"]
          %1: i8 = add %0, %0
          blackbox %1
          exit [%0]
        "#,
            |_| true,
            &["
          arrange_fill GPR0 from=Zeroed dst_bitw=8 to=Undefined
          copy_reg from=GPR1 to=GPR0
          alloc %1 GPR0 GPR1
          arrange_fill GPR1 from=Undefined dst_bitw=8 to=Zeroed
          copy_reg from=GPR0 to=GPR1
          arrange_fill GPR0 from=Undefined dst_bitw=8 to=Zeroed
        "],
        );
    }

    #[test]
    fn constants() {
        build_and_test(
            r#"
          %0: i8 = 2
          %1: i8 = add %0, %0
          blackbox %1
          exit []
        "#,
            |_| true,
            &["
          alloc %1 GPR0 GPR1
          const GPR1 tmp_reg=None tgt_bitw=8 fill=Zeroed Int(ArbBitInt { bitw: 8, val: 2 })
          const GPR0 tmp_reg=None tgt_bitw=8 fill=Zeroed Int(ArbBitInt { bitw: 8, val: 2 })
        "],
        );
    }

    #[test]
    fn constants_as_late_as_possible() {
        build_and_test(
            r#"
          %0: i8 = 2
          %1: i8 = add %0, %0
          blackbox %1
          %3: i8 = add %0, %0
          blackbox %3
          exit []
        "#,
            |_| true,
            &["
          alloc %3 GPR0 GPR1
          arrange_fill GPR0 from=Zeroed dst_bitw=8 to=Zeroed
          copy_reg from=GPR1 to=GPR0
          arrange_fill GPR1 from=Zeroed dst_bitw=8 to=Zeroed
          alloc %1 GPR0 GPR1
          const GPR1 tmp_reg=None tgt_bitw=8 fill=Zeroed Int(ArbBitInt { bitw: 8, val: 2 })
          const GPR0 tmp_reg=None tgt_bitw=8 fill=Zeroed Int(ArbBitInt { bitw: 8, val: 2 })
        "],
        );
    }

    #[test]
    fn constant_tmp_reg() {
        build_and_test(
            r#"
          %0: double = 0.0double
          %1: double = 1.0double
          %2: double = fadd %0, %1
          blackbox %2
          exit []
        "#,
            |_| true,
            &["
          alloc %2 FP0 FP1
          const FP1 tmp_reg=Some(GPR0) tgt_bitw=64 fill=Zeroed Double(1.0)
          const FP0 tmp_reg=Some(GPR0) tgt_bitw=64 fill=Zeroed Double(0.0)
        "],
        );
    }

    #[test]
    fn only_spill_a_register_once() {
        build_and_test(
            r#"
          %0: i64 = arg [reg]
          %1: i64 = arg [reg]
          %2: i64 = add %0, %1
          blackbox %2
          exit [%1, %0]
        "#,
            |_| true,
            &["
          unspill stack_off=8 GPR1 Undefined bitw=64
          arrange_fill GPR0 from=Zeroed dst_bitw=64 to=Undefined
          alloc %2 GPR1 GPR0
          unspill stack_off=16 GPR0 Undefined bitw=64
          arrange_fill GPR1 from=Undefined dst_bitw=64 to=Zeroed
          copy_reg from=GPR0 to=GPR1
          spill GPR0 Undefined stack_off=8 bitw=64
          spill GPR1 Undefined stack_off=16 bitw=64
        "],
        );
    }

    #[test]
    fn casts() {
        // The main thing we're testing is whether we merge casts when we can, and don't when we
        // shouldn't.

        build_and_test(
            r#"
          %0: i64 = arg [reg]
          %1: i32 = trunc %0
          blackbox %1
          exit [%0]
        "#,
            |_| true,
            &["
          unspill stack_off=8 GPR0 Undefined bitw=64
          trunc %1 GPR0
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
          spill GPR0 Undefined stack_off=8 bitw=64
        "],
        );

        build_and_test(
            r#"
          %0: i64 = arg [reg]
          %1: i32 = trunc %0
          %2: i64 = zext %1
          exit [%2]
        "#,
            |_| true,
            &["
          arrange_fill GPR0 from=Zeroed dst_bitw=64 to=Undefined
          zext %2 GPR0
          arrange_fill GPR0 from=Undefined dst_bitw=32 to=Zeroed
          trunc %1 GPR0
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
        "],
        );

        build_and_test(
            r#"
          %0: i64 = arg [reg]
          %1: i32 = trunc %0
          %2: i16 = trunc %1
          blackbox %2
          exit [%0]
        "#,
            |_| true,
            &["
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
          trunc %2 GPR1
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
          arrange_fill GPR1 from=Undefined dst_bitw=32 to=Undefined
          trunc %1 GPR1
          arrange_fill GPR1 from=Undefined dst_bitw=64 to=Undefined
          copy_reg from=GPR0 to=GPR1
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
        "],
        );

        build_and_test(
            r#"
          %0: i8 = arg [reg]
          %1: i16 = zext %0
          %2: i32 = zext %1
          %3: i8 = trunc %2
          exit [%3]
        "#,
            |_| true,
            &["
          arrange_fill GPR0 from=Undefined dst_bitw=8 to=Undefined
          trunc %3 GPR0
          arrange_fill GPR0 from=Zeroed dst_bitw=32 to=Undefined
          zext %2 GPR0
          arrange_fill GPR0 from=Zeroed dst_bitw=16 to=Zeroed
          zext %1 GPR0
          arrange_fill GPR0 from=Undefined dst_bitw=8 to=Zeroed
        "],
        );

        build_and_test(
            r#"
          %0: i64 = arg [reg]
          %1: i32 = trunc %0
          %2: i64 = zext %1
          blackbox %2
          exit [%0]
        "#,
            |_| true,
            &["
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
          zext %2 GPR1
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
          arrange_fill GPR1 from=Undefined dst_bitw=32 to=Zeroed
          trunc %1 GPR1
          arrange_fill GPR1 from=Undefined dst_bitw=64 to=Undefined
          copy_reg from=GPR0 to=GPR1
          arrange_fill GPR0 from=Undefined dst_bitw=64 to=Undefined
        "],
        );
    }
}
