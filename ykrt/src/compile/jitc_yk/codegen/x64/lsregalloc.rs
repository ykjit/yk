//! A simple linear scan register allocator.
//!
//! The "main" interface from the code generator to the register allocator is `assign_gp_regs` (and/or
//! `assign_fp_regs`) and [RegConstraint]. For example:
//!
//! ```rust,ignore
//! match binop {
//!   BinOp::Add => {
//!     let size = lhs.byte_size(self.m);
//!     let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
//!       &mut self.asm,
//!       iidx,
//!       [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
//!     );
//!     match size {
//!       1 => dynasm!(self.asm; add Rb(lhs_reg.code()), Rb(rhs_reg.code())),
//!       ...
//!     }
//! ```
//!
//! This says "assign two x64 registers: `lhs_reg` will take a value as input and later contain an
//! output value (clobbering the input value, which will be spilled if necessary); `rhs_reg` will
//! take a value as input (and mustn't clobber it)". Those registers can then be used with dynasmrt
//! as one expects.
//!
//! The allocator keeps track of which registers have which trace instruction's values in and of
//! where it has spilled an instruction's value: it guarantees to spill an instruction to at most
//! one place on the stack.

use super::{rev_analyse::RevAnalyse, Register, VarLocation};
use crate::compile::jitc_yk::{
    codegen::{abs_stack::AbstractStack, x64::REG64_BYTESIZE},
    jit_ir::{Const, ConstIdx, FloatTy, Inst, InstIdx, Module, Operand, PtrAddInst, Ty},
};
use dynasmrt::{
    dynasm,
    x64::{
        Assembler, {Rq, Rx},
    },
    DynasmApi, Register as dynasmrtRegister,
};
use std::{marker::PhantomData, mem};

/// The complete set of general purpose x64 registers, in the order that dynasmrt defines them.
/// Note that large portions of the code rely on these registers mapping to the integers 0..15
/// (both inc.) in order.
pub(crate) static GP_REGS: [Rq; 16] = [
    Rq::RAX,
    Rq::RCX,
    Rq::RDX,
    Rq::RBX,
    Rq::RSP,
    Rq::RBP,
    Rq::RSI,
    Rq::RDI,
    Rq::R8,
    Rq::R9,
    Rq::R10,
    Rq::R11,
    Rq::R12,
    Rq::R13,
    Rq::R14,
    Rq::R15,
];

/// How many general purpose registers are there? Only needed because `GP_REGS.len()` doesn't const
/// eval yet.
const GP_REGS_LEN: usize = 16;

/// The complete set of floating point x64 registers, in the order that dynasmrt defines them.
/// Note that large portions of the code rely on these registers mapping to the integers 0..15
/// (both inc.) in order.
pub(crate) static FP_REGS: [Rx; 16] = [
    Rx::XMM0,
    Rx::XMM1,
    Rx::XMM2,
    Rx::XMM3,
    Rx::XMM4,
    Rx::XMM5,
    Rx::XMM6,
    Rx::XMM7,
    Rx::XMM8,
    Rx::XMM9,
    Rx::XMM10,
    Rx::XMM11,
    Rx::XMM12,
    Rx::XMM13,
    Rx::XMM14,
    Rx::XMM15,
];

/// How many floating point registers are there? Only needed because `FP_REGS.len()` doesn't const
/// eval yet.
const FP_REGS_LEN: usize = 16;

/// The set of general registers which we will never assign value to. RSP & RBP are reserved by
/// SysV.
static RESERVED_GP_REGS: [Rq; 2] = [Rq::RSP, Rq::RBP];

/// The set of floating point registers which we will never assign value to.
static RESERVED_FP_REGS: [Rx; 0] = [];

/// A linear scan register allocator.
pub(crate) struct LSRegAlloc<'a> {
    m: &'a Module,
    pub(super) rev_an: RevAnalyse<'a>,
    /// Which general purpose registers are active?
    gp_regset: RegSet<Rq>,
    /// In what state are the general purpose registers?
    gp_reg_states: [RegState; GP_REGS_LEN],
    /// Which floating point registers are active?
    fp_regset: RegSet<Rx>,
    /// In what state are the floating point registers?
    fp_reg_states: [RegState; FP_REGS_LEN],
    /// Where on the stack is an instruction's value spilled? Set to `usize::MAX` if that offset is
    /// currently unknown. Note: multiple instructions can alias to the same [SpillState].
    spills: Vec<SpillState>,
    /// The abstract stack: shared between general purpose and floating point registers.
    stack: AbstractStack,
}

impl<'a> LSRegAlloc<'a> {
    /// Create a new register allocator, with the existing interpreter frame spanning
    /// `interp_stack_len` bytes.
    pub(crate) fn new(m: &'a Module, interp_stack_len: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            // We rely on the registers in GP_REGS being numbered 0..15 (inc.) for correctness.
            for (i, reg) in GP_REGS.iter().enumerate() {
                assert_eq!(reg.code(), u8::try_from(i).unwrap());
            }

            // We rely on the registers in FP_REGS being numbered 0..15 (inc.) for correctness.
            for (i, reg) in FP_REGS.iter().enumerate() {
                assert_eq!(reg.code(), u8::try_from(i).unwrap());
            }
        }

        let mut gp_reg_states = std::array::from_fn(|_| RegState::Empty);
        for reg in RESERVED_GP_REGS {
            gp_reg_states[usize::from(reg.code())] = RegState::Reserved;
        }

        let mut fp_reg_states = std::array::from_fn(|_| RegState::Empty);
        for reg in RESERVED_FP_REGS {
            fp_reg_states[usize::from(reg.code())] = RegState::Reserved;
        }

        let mut stack = AbstractStack::default();
        stack.grow(interp_stack_len);

        let mut rev_an = RevAnalyse::new(m);
        rev_an.analyse_header();
        LSRegAlloc {
            m,
            rev_an,
            gp_regset: RegSet::with_gp_reserved(),
            gp_reg_states,
            fp_regset: RegSet::with_fp_reserved(),
            fp_reg_states,
            spills: vec![SpillState::Empty; m.insts_len()],
            stack,
        }
    }

    /// Reset the register allocator. We use this when moving from the trace header into the trace
    /// body.
    pub(crate) fn reset(&mut self, header_end_vlocs: &[VarLocation]) {
        for rs in self.gp_reg_states.iter_mut() {
            *rs = RegState::Empty;
        }
        for reg in RESERVED_GP_REGS {
            self.gp_reg_states[usize::from(reg.code())] = RegState::Reserved;
        }
        self.gp_regset = RegSet::with_gp_reserved();

        for rs in self.fp_reg_states.iter_mut() {
            *rs = RegState::Empty;
        }
        for reg in RESERVED_FP_REGS {
            self.fp_reg_states[usize::from(reg.code())] = RegState::Reserved;
        }
        self.fp_regset = RegSet::with_fp_reserved();

        self.rev_an.analyse_body(header_end_vlocs);
    }

    /// Before generating code for the instruction at `iidx`, see which registers are no longer
    /// needed and mark them as [RegState::Empty]. Calling this allows the register allocator to
    /// use the set of available registers more efficiently.
    pub(crate) fn expire_regs(&mut self, iidx: InstIdx) {
        for reg in GP_REGS {
            match self.gp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => {
                    debug_assert!(self.gp_regset.is_set(reg));
                }
                RegState::Empty => {
                    debug_assert!(!self.gp_regset.is_set(reg));
                }
                RegState::FromConst(_) => {
                    debug_assert!(self.gp_regset.is_set(reg));
                    // FIXME: Don't immediately expire constants
                    self.gp_regset.unset(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegState::FromInst(reg_iidx) => {
                    debug_assert!(self.gp_regset.is_set(reg));
                    if !self.rev_an.is_inst_var_still_used_at(iidx, reg_iidx) {
                        self.gp_regset.unset(reg);
                        *self.gp_reg_states.get_mut(usize::from(reg.code())).unwrap() =
                            RegState::Empty;
                    }
                }
            }
        }

        for reg in FP_REGS {
            match self.fp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => {
                    debug_assert!(self.fp_regset.is_set(reg));
                }
                RegState::Empty => {
                    debug_assert!(!self.fp_regset.is_set(reg));
                }
                RegState::FromConst(_) => {
                    debug_assert!(self.fp_regset.is_set(reg));
                    // FIXME: Don't immediately expire constants
                    self.fp_regset.unset(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegState::FromInst(reg_iidx) => {
                    debug_assert!(self.fp_regset.is_set(reg));
                    if !self.rev_an.is_inst_var_still_used_at(iidx, reg_iidx) {
                        self.fp_regset.unset(reg);
                        *self.fp_reg_states.get_mut(usize::from(reg.code())).unwrap() =
                            RegState::Empty;
                    }
                }
            }
        }
    }

    /// Align the stack to `align` bytes and return the size of the stack after alignment.
    pub(crate) fn align_stack(&mut self, align: usize) -> usize {
        self.stack.align(align);
        self.stack.size()
    }

    /// The total stack size in bytes of this trace and all it's predecessors (or more accurately
    /// the stack pointer offset from the base pointer of the interpreter loop frame).
    pub(crate) fn stack_size(&mut self) -> usize {
        self.stack.size()
    }

    /// Return the inline [PtrAddInst] for a load/store, if there is one.
    pub(crate) fn ptradd(&self, iidx: InstIdx) -> Option<PtrAddInst> {
        self.rev_an.ptradds[usize::from(iidx)]
    }

    /// Assign registers for the instruction at position `iidx`.
    pub(crate) fn assign_regs<const NG: usize, const NF: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        gp_constraints: [RegConstraint<Rq>; NG],
        fp_constraints: [RegConstraint<Rx>; NF],
    ) -> ([Rq; NG], [Rx; NF]) {
        (
            self.assign_gp_regs(asm, iidx, gp_constraints),
            self.assign_fp_regs(asm, iidx, fp_constraints),
        )
    }
}

/// The parts of the register allocator needed for general purpose registers.
impl LSRegAlloc<'_> {
    /// Forcibly assign the machine register `reg` to the value produced by instruction `iidx`.
    /// Note that if this register is already used, a spill will be generated instead.
    pub(crate) fn force_assign_inst_gp_reg(&mut self, asm: &mut Assembler, iidx: InstIdx, reg: Rq) {
        if self.gp_regset.is_set(reg) {
            // Input values alias to a single register. To avoid the rest of the register allocator
            // having to think about this, we "dealias" the values by spilling.
            self.force_assign_and_spill_inst_gp_reg(asm, iidx, reg);
        } else {
            self.gp_regset.set(reg);
            self.gp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
        }
    }

    /// Forcibly spill the machine register `reg` and assign the spilled value as being produced by
    /// instruction `iidx`.
    pub(crate) fn force_assign_and_spill_inst_gp_reg(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        reg: Rq,
    ) {
        debug_assert_eq!(self.spills[usize::from(iidx)], SpillState::Empty);
        let inst = self.m.inst(iidx);
        let size = inst.def_byte_size(self.m);
        self.stack.align(size); // FIXME
        let frame_off = self.stack.grow(size);
        let off = i32::try_from(frame_off).unwrap();
        match size {
            1 => dynasm!(asm; mov BYTE [rbp - off], Rb(reg.code())),
            2 => dynasm!(asm; mov WORD [rbp - off], Rw(reg.code())),
            4 => dynasm!(asm; mov DWORD [rbp - off], Rd(reg.code())),
            8 => dynasm!(asm; mov QWORD [rbp - off], Rq(reg.code())),
            _ => unreachable!(),
        }
        self.spills[usize::from(iidx)] = SpillState::Stack(off);
    }

    /// Forcibly assign the floating point register `reg`, which must be in the [RegState::Empty] state,
    /// to the value produced by instruction `iidx`.
    pub(crate) fn force_assign_inst_fp_reg(&mut self, iidx: InstIdx, reg: Rx) {
        debug_assert!(!self.fp_regset.is_set(reg));
        self.fp_regset.set(reg);
        self.fp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
    }

    /// Forcibly assign the value produced by instruction `iidx` to `Direct` `frame_off`.
    pub(crate) fn force_assign_inst_direct(&mut self, iidx: InstIdx, frame_off: i32) {
        debug_assert_eq!(self.spills[usize::from(iidx)], SpillState::Empty);
        self.spills[usize::from(iidx)] = SpillState::Direct(frame_off);
    }

    /// Forcibly assign the value produced by instruction `iidx` to `Indirect` `frame_off`.
    pub(crate) fn force_assign_inst_indirect(&mut self, iidx: InstIdx, frame_off: i32) {
        debug_assert_eq!(self.spills[usize::from(iidx)], SpillState::Empty);
        self.spills[usize::from(iidx)] = SpillState::Stack(frame_off);
    }

    /// Forcibly assign a constant to an instruction. This typically only happens when traces pass
    /// live variables that have been optimised to constants into side-traces.
    pub(crate) fn assign_const(&mut self, iidx: InstIdx, bits: u32, v: u64) {
        self.spills[usize::from(iidx)] = SpillState::ConstInt { bits, v };
    }

    /// Assign registers for the instruction `iidx`, which consumes the [Operand] `op` but does not
    /// change its value. In many cases, the register allocator can avoid generating any code for
    /// this case at all.
    pub(crate) fn assign_gp_pass_through(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        op: Operand,
    ) {
        match op {
            Operand::Var(op_iidx) => {
                match self.gp_reg_states.iter().position(|x| {
                    if let RegState::FromInst(y) = x {
                        *y == op_iidx
                    } else {
                        false
                    }
                }) {
                    Some(reg_i) => {
                        if self.rev_an.is_inst_var_still_used_after(iidx, op_iidx) {
                            let mut avoid = RegSet::with_gp_reserved();
                            self.move_or_spill_gp(asm, iidx, &mut avoid, GP_REGS[reg_i]);
                        }
                        self.gp_reg_states[reg_i] = RegState::FromInst(iidx);
                    }
                    None => {
                        self.spills[usize::from(iidx)] = self.spills[usize::from(op_iidx)];
                    }
                }
            }
            Operand::Const(_cidx) => todo!(),
        }
    }

    /// Assign general purpose registers for the instruction at position `iidx`.
    ///
    /// This is a convenience function for [Self::assign_regs] when there are no FP registers.
    pub(crate) fn assign_gp_regs<const N: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        mut constraints: [RegConstraint<Rq>; N],
    ) -> [Rq; N] {
        // No constraint operands should be float-typed.
        #[cfg(debug_assertions)]
        for c in &constraints {
            if let Some(o) = c.operand() {
                debug_assert!(!matches!(self.m.type_(o.tyidx(self.m)), Ty::Float(_)));
            }
        }

        // There must be at most 1 output register.
        debug_assert!(
            constraints
                .iter()
                .filter(|x| match x {
                    RegConstraint::Input(_)
                    | RegConstraint::InputIntoReg(_, _)
                    | RegConstraint::InputIntoRegAndClobber(_, _) => false,
                    RegConstraint::InputOutputIntoReg(_, _)
                    | RegConstraint::Output
                    | RegConstraint::OutputCanBeSameAsInput(_)
                    | RegConstraint::OutputFromReg(_)
                    | RegConstraint::InputOutput(_) => true,
                    RegConstraint::Clobber(_) | RegConstraint::Temporary | RegConstraint::None =>
                        false,
                })
                .count()
                <= 1
        );

        let mut avoid = RegSet::with_gp_reserved();

        // For each constraint, we will find a register to assign it to.
        let mut asgn = [None; N];

        // Where the caller has told us they want to put things in specific registers, we need to
        // make sure we avoid assigning those in all other circumstances.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::InputIntoReg(_, reg) => {
                    asgn[i] = Some(*reg);
                    avoid.set(*reg);
                }
                RegConstraint::InputIntoRegAndClobber(_, reg)
                | RegConstraint::InputOutputIntoReg(_, reg)
                | RegConstraint::OutputFromReg(reg)
                | RegConstraint::Clobber(reg) => {
                    asgn[i] = Some(*reg);
                    avoid.set(*reg);
                }
                RegConstraint::InputOutput(_)
                | RegConstraint::Output
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::Input(_)
                | RegConstraint::Temporary
                | RegConstraint::None => {}
            }
        }

        // Deal with `OutputCanBeSameAsInput`.
        for i in 0..constraints.len() {
            if let RegConstraint::OutputCanBeSameAsInput(search_op) = constraints[i].clone() {
                if let Some(Register::GP(reg)) = self.rev_an.reg_hints[usize::from(iidx)] {
                    if avoid.is_set(reg) {
                        continue;
                    }
                    if let Operand::Var(search_op_iidx) = search_op {
                        if !self
                            .rev_an
                            .is_inst_var_still_used_after(iidx, search_op_iidx)
                        {
                            for j in 0..constraints.len() {
                                if let RegConstraint::Input(in_op) = constraints[j].clone() {
                                    if search_op == in_op {
                                        constraints[i] = RegConstraint::OutputFromReg(reg);
                                        constraints[j] = RegConstraint::InputIntoReg(in_op, reg);
                                        asgn[i] = Some(reg);
                                        asgn[j] = Some(reg);
                                        avoid.set(reg);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // If we have a hint for a constraint, use it.
        for (i, cnstr) in constraints.iter_mut().enumerate() {
            match cnstr {
                RegConstraint::Output
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::InputOutput(_) => {
                    if let Some(Register::GP(reg)) = self.rev_an.reg_hints[usize::from(iidx)] {
                        if !avoid.is_set(reg) {
                            *cnstr = match cnstr {
                                RegConstraint::Output => RegConstraint::OutputFromReg(reg),
                                RegConstraint::OutputCanBeSameAsInput(_) => {
                                    RegConstraint::OutputFromReg(reg)
                                }
                                RegConstraint::InputOutput(op) => {
                                    RegConstraint::InputOutputIntoReg(op.clone(), reg)
                                }
                                _ => unreachable!(),
                            };
                            asgn[i] = Some(reg);
                            avoid.set(reg);
                        }
                    }
                }
                _ => (),
            }
        }

        // If we already have the value in a register, don't assign a new register.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::Input(op) | RegConstraint::InputOutput(op) => match op {
                    Operand::Var(op_iidx) => {
                        if let Some(reg_i) = self.gp_reg_states.iter().position(|x| {
                            if let RegState::FromInst(y) = x {
                                y == op_iidx
                            } else {
                                false
                            }
                        }) {
                            let reg = GP_REGS[reg_i];
                            if !avoid.is_set(reg) {
                                debug_assert!(self.gp_regset.is_set(reg));
                                match cnstr {
                                    RegConstraint::Input(_) => asgn[i] = Some(reg),
                                    RegConstraint::InputOutput(_) => asgn[i] = Some(reg),
                                    _ => unreachable!(),
                                }
                                avoid.set(reg);
                            }
                        }
                    }
                    Operand::Const(_cidx) => (),
                },
                RegConstraint::InputIntoReg(_, _)
                | RegConstraint::InputOutputIntoReg(_, _)
                | RegConstraint::InputIntoRegAndClobber(_, _)
                | RegConstraint::Clobber(_) => {
                    // These were all handled in the first for loop.
                }
                RegConstraint::Output
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::OutputFromReg(_)
                | RegConstraint::Temporary => (),
                RegConstraint::None => {
                    asgn[i] = Some(GP_REGS[0]);
                }
            }
        }

        // Assign a register for all unassigned constraints.
        for (i, _) in constraints.iter().enumerate() {
            if asgn[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            let reg = match self.gp_regset.find_empty_avoiding(avoid) {
                Some(reg) => reg,
                None => {
                    // We need to find a register to spill. Our heuristic is (in order):
                    //   1. If a register's value contains a constant, use that.
                    //   2. If a register's value is already spilt use that.
                    //   3. Spill the register whose value is used furthest away in the trace based
                    //      on the reverse analyser's (def, use) analysis.
                    //   4. If (1) or (2) leads to a tie, spill the register whose values is next
                    //      used furthest away from the current instruction.
                    let mut cnd_const = None;
                    let mut cnd_spill = None;
                    let mut cnd_furthest = None;
                    for reg in GP_REGS {
                        if avoid.is_set(reg) {
                            continue;
                        }
                        match self.gp_reg_states[usize::from(reg.code())] {
                            RegState::Reserved => (),
                            RegState::Empty => unreachable!(),
                            RegState::FromConst(_) => todo!(),
                            RegState::FromInst(from_iidx) => {
                                match self.spills[usize::from(from_iidx)] {
                                    SpillState::Empty => match cnd_furthest {
                                        None => cnd_furthest = Some((reg, from_iidx)),
                                        Some((_, furthest_iidx)) => {
                                            if let Some(next_iidx) =
                                                self.rev_an.next_use(iidx, from_iidx)
                                            {
                                                if next_iidx > furthest_iidx {
                                                    cnd_furthest = Some((reg, from_iidx))
                                                }
                                            }
                                        }
                                    },
                                    SpillState::Stack(_) | SpillState::Direct(_) => match cnd_spill
                                    {
                                        None => cnd_spill = Some((reg, from_iidx)),
                                        Some((_, spill_iidx)) => {
                                            if let Some(next_iidx) =
                                                self.rev_an.next_use(iidx, from_iidx)
                                            {
                                                if next_iidx > spill_iidx {
                                                    cnd_spill = Some((reg, from_iidx))
                                                }
                                            }
                                        }
                                    },
                                    SpillState::ConstInt { .. } => {
                                        // Should we encounter multiple constants in registers
                                        // (which isn't very likely...), we want to spill the one
                                        // in the lowest register, since that's more likely to be
                                        // clobbered by a CALL.
                                        if cnd_const.is_none() {
                                            cnd_const = Some(reg);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(reg) = cnd_const {
                        reg
                    } else if let Some((reg, _)) = cnd_spill {
                        reg
                    } else if let Some((reg, _)) = cnd_furthest {
                        reg
                    } else {
                        panic!("Cannot satisfy register constraints: no registers left");
                    }
                }
            };
            asgn[i] = Some(reg);
            avoid.set(reg);
        }

        // At this point, we've found a register for every constraint. We now need to decide if we
        // need to move/spill any existing values in those registers.

        // Try to move / swap existing registers, if possible.
        debug_assert_eq!(constraints.len(), asgn.len());
        for (cnstr, new_reg) in constraints.iter().zip(asgn.into_iter()) {
            let new_reg = new_reg.unwrap();
            match cnstr {
                RegConstraint::Input(ref op)
                | RegConstraint::InputIntoReg(ref op, _)
                | RegConstraint::InputOutput(ref op)
                | RegConstraint::InputOutputIntoReg(ref op, _)
                | RegConstraint::InputIntoRegAndClobber(ref op, _) => {
                    if let Some(old_reg) = self.find_op_in_gp_reg(op) {
                        if old_reg != new_reg {
                            match self.gp_reg_states[usize::from(new_reg.code())] {
                                RegState::Reserved => unreachable!(),
                                RegState::Empty => {
                                    self.move_gp_reg(asm, old_reg, new_reg);
                                }
                                RegState::FromConst(_) => todo!(),
                                RegState::FromInst(query_iidx) => {
                                    if self.rev_an.is_inst_var_still_used_at(iidx, query_iidx) {
                                        self.swap_gp_reg(asm, old_reg, new_reg);
                                    } else {
                                        self.move_gp_reg(asm, old_reg, new_reg);
                                    }
                                }
                            }
                        }
                    }
                }
                RegConstraint::Output
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::OutputFromReg(_)
                | RegConstraint::Clobber(_)
                | RegConstraint::Temporary
                | RegConstraint::None => (),
            }
        }

        // Spill / unspill what we couldn't move.
        for (cnstr, reg) in constraints.into_iter().zip(asgn.into_iter()) {
            let reg = reg.unwrap();
            match cnstr {
                RegConstraint::Input(ref op) | RegConstraint::InputIntoReg(ref op, _) => {
                    if !self.is_input_in_gp_reg(op, reg) {
                        self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                        self.put_input_in_gp_reg(asm, op, reg);
                    }
                }
                RegConstraint::InputIntoRegAndClobber(ref op, _) => {
                    if !self.is_input_in_gp_reg(op, reg) {
                        self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                        self.put_input_in_gp_reg(asm, op, reg);
                    } else {
                        self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                    }
                    self.gp_regset.unset(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegConstraint::InputOutput(ref op)
                | RegConstraint::InputOutputIntoReg(ref op, _) => {
                    if !self.is_input_in_gp_reg(op, reg) {
                        self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                        self.put_input_in_gp_reg(asm, op, reg);
                    } else {
                        self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                    }
                    self.gp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                }
                RegConstraint::Output
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::OutputFromReg(_) => {
                    self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                    self.gp_regset.set(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                }
                RegConstraint::Clobber(_) | RegConstraint::Temporary => {
                    self.move_or_spill_gp(asm, iidx, &mut avoid, reg);
                    self.gp_regset.unset(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegConstraint::None => (),
            }
        }
        asgn.map(|x| x.unwrap())
    }

    /// Return the GP register containing the value for `op` or `None` if that value is not in any
    /// register.
    fn find_op_in_gp_reg(&self, op: &Operand) -> Option<Rq> {
        self.gp_reg_states
            .iter()
            .enumerate()
            .find(|(_, x)| match (op, x) {
                (Operand::Const(op_cidx), RegState::FromConst(reg_cidx)) => *op_cidx == *reg_cidx,
                (Operand::Var(op_iidx), RegState::FromInst(reg_iidx)) => *op_iidx == *reg_iidx,
                _ => false,
            })
            .map(|(i, _)| GP_REGS[i])
    }

    /// Is the value produced by `op` already in register `reg`?
    fn is_input_in_gp_reg(&self, op: &Operand, reg: Rq) -> bool {
        match self.gp_reg_states[usize::from(reg.code())] {
            RegState::Empty => false,
            RegState::FromConst(reg_cidx) => match op {
                Operand::Const(op_cidx) => reg_cidx == *op_cidx,
                Operand::Var(_) => false,
            },
            RegState::FromInst(reg_iidx) => match op {
                Operand::Const(_) => false,
                Operand::Var(op_iidx) => reg_iidx == *op_iidx,
            },
            RegState::Reserved => unreachable!(),
        }
    }

    /// Put the value for `op` into `reg`. It is assumed that the caller has already checked that
    /// the value for `op` is not already in `reg`.
    fn put_input_in_gp_reg(&mut self, asm: &mut Assembler, op: &Operand, reg: Rq) {
        debug_assert!(!self.is_input_in_gp_reg(op, reg));
        let st = match op {
            Operand::Const(cidx) => {
                self.load_const_into_gp_reg(asm, *cidx, reg);
                RegState::FromConst(*cidx)
            }
            Operand::Var(iidx) => {
                self.force_gp_unspill(asm, *iidx, reg);
                RegState::FromInst(*iidx)
            }
        };
        self.gp_regset.set(reg);
        self.gp_reg_states[usize::from(reg.code())] = st;
    }

    /// Move the value in `old_reg` to `new_reg`, setting `old_reg` to [RegState::Empty].
    fn move_gp_reg(&mut self, asm: &mut Assembler, old_reg: Rq, new_reg: Rq) {
        dynasm!(asm; mov Rq(new_reg.code()), Rq(old_reg.code()));
        self.gp_regset.set(new_reg);
        self.gp_reg_states[usize::from(new_reg.code())] = mem::replace(
            &mut self.gp_reg_states[usize::from(old_reg.code())],
            RegState::Empty,
        );
        self.gp_regset.unset(old_reg);
    }

    /// Swap the values, and register states, for `old_reg` and `new_reg`.
    fn swap_gp_reg(&mut self, asm: &mut Assembler, old_reg: Rq, new_reg: Rq) {
        dynasm!(asm; xchg Rq(new_reg.code()), Rq(old_reg.code()));
        self.gp_reg_states
            .swap(usize::from(old_reg.code()), usize::from(new_reg.code()));
    }

    /// We are about to clobber `old_reg`, so if its value is needed later (1) move it to another
    /// register if there's a spare available or (2) ensure it is already spilled or (2) spill it.
    fn move_or_spill_gp(
        &mut self,
        asm: &mut Assembler,
        cur_iidx: InstIdx,
        avoid: &mut RegSet<Rq>,
        old_reg: Rq,
    ) {
        match self.gp_reg_states[usize::from(old_reg.code())] {
            RegState::Empty => (),
            RegState::FromConst(_) => (),
            RegState::FromInst(query_iidx) => {
                if self
                    .rev_an
                    .is_inst_var_still_used_after(cur_iidx, query_iidx)
                {
                    let mut new_reg = None;
                    // Try to use `query_iidx`s hint, if there is one, and it's not in use...
                    if let Some(Register::GP(reg)) = self.rev_an.reg_hints[usize::from(query_iidx)]
                    {
                        if !self.gp_regset.is_set(reg) && !avoid.is_set(reg) {
                            new_reg = Some(reg);
                        }
                    }
                    // ...otherwise try to find an arbitrary empty register.
                    if new_reg.is_none() {
                        if let Some(reg) = self.gp_regset.find_empty_avoiding(*avoid) {
                            new_reg = Some(reg);
                        }
                    }
                    if let Some(new_reg) = new_reg {
                        // We found a register to move to.
                        dynasm!(asm; mov Rq(new_reg.code()), Rq(old_reg.code()));
                        avoid.set(new_reg);
                        self.gp_regset.set(new_reg);
                        self.gp_reg_states[usize::from(new_reg.code())] =
                            self.gp_reg_states[usize::from(old_reg.code())].clone();
                    } else {
                        // We have no choice but to spill.
                        self.spill_gp_if_not_already(asm, old_reg);
                    }
                }
            }
            RegState::Reserved => unreachable!(),
        }
    }

    /// If the value stored in `reg` is not already spilled to the heap, then spill it. Note that
    /// this function neither writes to the register or changes the register's [RegState].
    fn spill_gp_if_not_already(&mut self, asm: &mut Assembler, reg: Rq) {
        match self.gp_reg_states[usize::from(reg.code())] {
            RegState::Reserved | RegState::Empty | RegState::FromConst(_) => (),
            RegState::FromInst(iidx) => {
                if self.spills[usize::from(iidx)] == SpillState::Empty {
                    let inst = self.m.inst(iidx);
                    let bitw = inst.def_bitw(self.m);
                    let bytew = inst.def_byte_size(self.m);
                    debug_assert!(bitw >= bytew);
                    self.stack.align(bytew);
                    let frame_off = self.stack.grow(bytew);
                    let off = i32::try_from(frame_off).unwrap();
                    match bitw {
                        1 => dynasm!(asm
                            ; and Rq(reg.code()), 1
                            ; mov BYTE [rbp - off], Rb(reg.code())
                        ),
                        8 => dynasm!(asm; mov BYTE [rbp - off], Rb(reg.code())),
                        16 => dynasm!(asm; mov WORD [rbp - off], Rw(reg.code())),
                        32 => dynasm!(asm; mov DWORD [rbp - off], Rd(reg.code())),
                        64 => dynasm!(asm; mov QWORD [rbp - off], Rq(reg.code())),
                        _ => unreachable!(),
                    }
                    self.spills[usize::from(iidx)] = SpillState::Stack(off);
                }
            }
        }
    }

    /// Load the value for `iidx` from the stack into `reg`.
    ///
    /// If the register is larger than the type being loaded the unused high-order bits are
    /// undefined.
    ///
    /// # Panics
    ///
    /// If `iidx` has not previously been spilled.
    fn force_gp_unspill(&mut self, asm: &mut Assembler, iidx: InstIdx, reg: Rq) {
        let inst = self.m.inst(iidx);
        let size = inst.def_byte_size(self.m);

        if let Inst::Const(cidx) = inst {
            self.load_const_into_gp_reg(asm, cidx, reg);
            return;
        }

        match self.spills[usize::from(iidx)] {
            SpillState::Empty => {
                let reg_i = self
                    .gp_reg_states
                    .iter()
                    .position(|x| {
                        if let RegState::FromInst(y) = x {
                            *y == iidx
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let cur_reg = GP_REGS[reg_i];
                if cur_reg != reg {
                    dynasm!(asm; mov Rq(reg.code()), Rq(cur_reg.code()));
                }
            }
            SpillState::Stack(off) => {
                match size {
                    1 => dynasm!(asm ; movzx Rq(reg.code()), BYTE [rbp - off]),
                    2 => dynasm!(asm ; movzx Rq(reg.code()), WORD [rbp - off]),
                    4 => dynasm!(asm ; mov Rd(reg.code()), [rbp - off]),
                    8 => dynasm!(asm ; mov Rq(reg.code()), [rbp - off]),
                    _ => todo!("{}", size),
                }
                self.gp_regset.set(reg);
            }
            SpillState::Direct(off) => match size {
                8 => dynasm!(asm
                    ; lea Rq(reg.code()), [rbp + off]
                ),
                x => todo!("{x}"),
            },
            SpillState::ConstInt { bits, v } => match bits {
                32 => {
                    dynasm!(asm; mov Rd(reg.code()), v as i32)
                }
                8 => {
                    dynasm!(asm; mov Rd(reg.code()), v as i32)
                }
                _ => todo!("{bits}"),
            },
        }
    }

    /// Load the constant from `cidx` into `reg`.
    ///
    /// If the register is larger than the constant, the unused high-order bits are undefined.
    fn load_const_into_gp_reg(&mut self, asm: &mut Assembler, cidx: ConstIdx, reg: Rq) {
        match self.m.const_(cidx) {
            Const::Float(_tyidx, _x) => todo!(),
            Const::Int(tyidx, x) => {
                // `unwrap` cannot fail, integers are sized.
                if self.m.type_(*tyidx).byte_size().unwrap() <= REG64_BYTESIZE {
                    dynasm!(asm; mov Rq(reg.code()), QWORD *x as i64);
                } else {
                    todo!();
                }
            }
            Const::Ptr(x) => {
                dynasm!(asm; mov Rq(reg.code()), QWORD *x as i64)
            }
        }
    }

    /// Return the location of the value at `iidx`. If that instruction's value is available in a
    /// register and is spilled to the stack, the former will always be preferred.
    ///
    /// Note that it is undefined behaviour to ask for the location of an instruction which has not
    /// yet produced a value.
    pub(crate) fn var_location(&self, iidx: InstIdx) -> VarLocation {
        if let Some(reg_i) = self.gp_reg_states.iter().position(|x| {
            if let RegState::FromInst(y) = x {
                *y == iidx
            } else {
                false
            }
        }) {
            VarLocation::Register(Register::GP(GP_REGS[reg_i]))
        } else if let Some(reg_i) = self.fp_reg_states.iter().position(|x| {
            if let RegState::FromInst(y) = x {
                *y == iidx
            } else {
                false
            }
        }) {
            VarLocation::Register(Register::FP(FP_REGS[reg_i]))
        } else {
            let inst = self.m.inst(iidx);
            let size = inst.def_byte_size(self.m);
            match inst {
                Inst::Copy(_) => panic!(),
                Inst::Const(cidx) => match self.m.const_(cidx) {
                    Const::Float(_, v) => VarLocation::ConstFloat(*v),
                    Const::Int(tyidx, v) => {
                        let Ty::Integer(bits) = self.m.type_(*tyidx) else {
                            panic!()
                        };
                        VarLocation::ConstInt { bits: *bits, v: *v }
                    }
                    Const::Ptr(p) => VarLocation::ConstInt {
                        bits: 64,
                        v: u64::try_from(*p).unwrap(),
                    },
                },
                _ => match self.spills[usize::from(iidx)] {
                    SpillState::Empty => panic!(),
                    SpillState::Stack(off) => VarLocation::Stack {
                        frame_off: u32::try_from(off).unwrap(),
                        size,
                    },
                    SpillState::Direct(off) => VarLocation::Direct {
                        frame_off: off,
                        size,
                    },
                    SpillState::ConstInt { bits, v } => VarLocation::ConstInt { bits, v },
                },
            }
        }
    }
}

impl LSRegAlloc<'_> {
    /// Assign floating point registers for the instruction at position `iidx`.
    ///
    /// This is a convenience function for [Self::assign_regs] when there are no GP registers.
    pub(crate) fn assign_fp_regs<const N: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        mut constraints: [RegConstraint<Rx>; N],
    ) -> [Rx; N] {
        // All constraint operands should be float-typed.
        #[cfg(debug_assertions)]
        for c in &constraints {
            if let Some(o) = c.operand() {
                debug_assert!(matches!(self.m.type_(o.tyidx(self.m)), Ty::Float(_)));
            }
        }

        // There must be at most 1 output register.
        debug_assert!(
            constraints
                .iter()
                .filter(|x| match x {
                    RegConstraint::Input(_)
                    | RegConstraint::InputIntoReg(_, _)
                    | RegConstraint::InputIntoRegAndClobber(_, _) => false,
                    RegConstraint::InputOutputIntoReg(_, _)
                    | RegConstraint::Output
                    | RegConstraint::OutputCanBeSameAsInput(_)
                    | RegConstraint::OutputFromReg(_)
                    | RegConstraint::InputOutput(_) => true,
                    RegConstraint::Clobber(_) | RegConstraint::Temporary | RegConstraint::None =>
                        false,
                })
                .count()
                <= 1
        );

        let mut avoid = RegSet::with_fp_reserved();

        // For each constraint, we will find a register to assign it to.
        let mut asgn = [None; N];

        // Where the caller has told us they want to put things in specific registers, we need to
        // make sure we avoid assigning those in all other circumstances.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::InputIntoReg(_, reg) => {
                    asgn[i] = Some(*reg);
                    avoid.set(*reg);
                }
                RegConstraint::InputIntoRegAndClobber(_, reg)
                | RegConstraint::InputOutputIntoReg(_, reg)
                | RegConstraint::OutputFromReg(reg)
                | RegConstraint::Clobber(reg) => {
                    asgn[i] = Some(*reg);
                    avoid.set(*reg);
                }
                RegConstraint::Input(_)
                | RegConstraint::InputOutput(_)
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::Output
                | RegConstraint::Temporary => {}
                RegConstraint::None => {
                    asgn[i] = Some(FP_REGS[0]);
                }
            }
        }

        // Deal with `OutputCanBeSameAsInput`.
        for cnstr in &constraints {
            if let RegConstraint::OutputCanBeSameAsInput(_) = cnstr {
                todo!();
            }
        }

        // If we have a hint for a constraint, use it.
        for (i, cnstr) in constraints.iter_mut().enumerate() {
            match cnstr {
                RegConstraint::Output
                | RegConstraint::OutputCanBeSameAsInput(_)
                | RegConstraint::InputOutput(_) => {
                    if let Some(Register::FP(reg)) = self.rev_an.reg_hints[usize::from(iidx)] {
                        if !avoid.is_set(reg) {
                            *cnstr = match cnstr {
                                RegConstraint::Output => RegConstraint::OutputFromReg(reg),
                                RegConstraint::OutputCanBeSameAsInput(_) => {
                                    RegConstraint::OutputFromReg(reg)
                                }
                                RegConstraint::InputOutput(op) => {
                                    RegConstraint::InputOutputIntoReg(op.clone(), reg)
                                }
                                _ => unreachable!(),
                            };
                            asgn[i] = Some(reg);
                            avoid.set(reg);
                        }
                    }
                }
                _ => (),
            }
        }

        // If we already have the value in a register, don't assign a new register.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::Input(op) | RegConstraint::InputOutput(op) => match op {
                    Operand::Var(op_iidx) => {
                        if let Some(reg_i) = self.fp_reg_states.iter().position(|x| {
                            if let RegState::FromInst(y) = x {
                                y == op_iidx
                            } else {
                                false
                            }
                        }) {
                            let reg = FP_REGS[reg_i];
                            if !avoid.is_set(reg) {
                                debug_assert!(self.fp_regset.is_set(reg));
                                match cnstr {
                                    RegConstraint::Input(_) => asgn[i] = Some(reg),
                                    RegConstraint::InputOutput(_) => asgn[i] = Some(reg),
                                    _ => unreachable!(),
                                }
                                avoid.set(reg);
                            }
                        }
                    }
                    Operand::Const(_cidx) => (),
                },
                RegConstraint::InputIntoReg(_, _)
                | RegConstraint::InputOutputIntoReg(_, _)
                | RegConstraint::InputIntoRegAndClobber(_, _)
                | RegConstraint::Clobber(_) => {
                    // These were all handled in the first for loop.
                }
                RegConstraint::Output
                | RegConstraint::OutputFromReg(_)
                | RegConstraint::Temporary => (),
                RegConstraint::OutputCanBeSameAsInput(_) => todo!(),
                RegConstraint::None => (),
            }
        }

        // Assign a register for all unassigned constraints.
        for (i, _) in constraints.iter().enumerate() {
            if asgn[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            let reg = match self.fp_regset.find_empty_avoiding(avoid) {
                Some(reg) => reg,
                None => {
                    // We need to find a register to spill. Our heuristic is two-fold:
                    //   1. Spill the register whose value is used furthest away in the trace based
                    //      on the reverse analyser's (def, use) analysis.
                    //   2. If (1) leads to a tie, spill the "highest" register (e.g. prefer to
                    //      spill XMM15 over XMM0) because "lower" registers are more likely to be
                    //      clobbered by CALLS, and we assume that the more recently we've put a
                    //      value into a register, the more likely it is to be used again soon.
                    let mut furthest = None;
                    for reg in FP_REGS {
                        if avoid.is_set(reg) {
                            continue;
                        }
                        match self.fp_reg_states[usize::from(reg.code())] {
                            RegState::Reserved => (),
                            RegState::Empty => unreachable!(),
                            RegState::FromConst(_) => todo!(),
                            RegState::FromInst(from_iidx) => {
                                debug_assert!(self
                                    .rev_an
                                    .is_inst_var_still_used_at(iidx, from_iidx));
                                if furthest.is_none() {
                                    furthest = Some((reg, from_iidx));
                                } else if let Some((_, furthest_iidx)) = furthest {
                                    if let Some(next_iidx) = self.rev_an.next_use(iidx, from_iidx) {
                                        if next_iidx > furthest_iidx {
                                            furthest = Some((reg, from_iidx))
                                        }
                                    }
                                }
                            }
                        }
                    }

                    match furthest {
                        Some((reg, _)) => reg,
                        None => panic!("Cannot satisfy register constraints: no registers left"),
                    }
                }
            };
            asgn[i] = Some(reg);
            avoid.set(reg);
        }

        // At this point, we've found a register for every constraint. We now need to decide if we
        // need to move/spill any existing values in those registers.

        // Try to move / swap existing registers, if possible.
        debug_assert_eq!(constraints.len(), asgn.len());
        for (cnstr, new_reg) in constraints.iter().zip(asgn.into_iter()) {
            let new_reg = new_reg.unwrap();
            match cnstr {
                RegConstraint::Input(ref op)
                | RegConstraint::InputIntoReg(ref op, _)
                | RegConstraint::InputOutput(ref op)
                | RegConstraint::InputOutputIntoReg(ref op, _)
                | RegConstraint::InputIntoRegAndClobber(ref op, _) => {
                    if let Some(old_reg) = self.find_op_in_fp_reg(op) {
                        if old_reg != new_reg {
                            match self.fp_reg_states[usize::from(new_reg.code())] {
                                RegState::Reserved => unreachable!(),
                                RegState::Empty => {
                                    self.move_fp_reg(asm, old_reg, new_reg);
                                }
                                RegState::FromConst(_) => todo!(),
                                RegState::FromInst(query_iidx) => {
                                    if self.rev_an.is_inst_var_still_used_at(iidx, query_iidx) {
                                        self.swap_fp_reg(asm, old_reg, new_reg);
                                    } else {
                                        self.move_fp_reg(asm, old_reg, new_reg);
                                    }
                                }
                            }
                        }
                    }
                }
                RegConstraint::Output
                | RegConstraint::OutputFromReg(_)
                | RegConstraint::Clobber(_)
                | RegConstraint::Temporary => (),
                RegConstraint::OutputCanBeSameAsInput(_) => todo!(),
                RegConstraint::None => (),
            }
        }

        // Spill / unspill what we couldn't move.
        for (cnstr, reg) in constraints.into_iter().zip(asgn.into_iter()) {
            let reg = reg.unwrap();
            match cnstr {
                RegConstraint::Input(ref op) | RegConstraint::InputIntoReg(ref op, _) => {
                    if !self.is_input_in_fp_reg(op, reg) {
                        self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                        self.put_input_in_fp_reg(asm, op, reg);
                    }
                }
                RegConstraint::InputIntoRegAndClobber(ref op, _) => {
                    if !self.is_input_in_fp_reg(op, reg) {
                        self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                        self.put_input_in_fp_reg(asm, op, reg);
                    } else {
                        self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                    }
                    self.fp_regset.unset(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegConstraint::InputOutput(ref op)
                | RegConstraint::InputOutputIntoReg(ref op, _) => {
                    if !self.is_input_in_fp_reg(op, reg) {
                        self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                        self.put_input_in_fp_reg(asm, op, reg);
                    } else {
                        self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                    }
                    self.fp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                }
                RegConstraint::Output | RegConstraint::OutputFromReg(_) => {
                    self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                    self.fp_regset.set(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                }
                RegConstraint::Clobber(_) | RegConstraint::Temporary => {
                    self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                    self.fp_regset.unset(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegConstraint::OutputCanBeSameAsInput(_) => todo!(),
                RegConstraint::None => (),
            }
        }
        asgn.map(|x| x.unwrap())
    }

    /// Return the FP register containing the value for `op` or `None` if that value is not in any
    /// register.
    fn find_op_in_fp_reg(&self, op: &Operand) -> Option<Rx> {
        self.fp_reg_states
            .iter()
            .enumerate()
            .find(|(_, x)| match (op, x) {
                (Operand::Const(op_cidx), RegState::FromConst(reg_cidx)) => *op_cidx == *reg_cidx,
                (Operand::Var(op_iidx), RegState::FromInst(reg_iidx)) => *op_iidx == *reg_iidx,
                _ => false,
            })
            .map(|(i, _)| FP_REGS[i])
    }

    /// Is the value produced by `op` already in register `reg`?
    fn is_input_in_fp_reg(&self, op: &Operand, reg: Rx) -> bool {
        match self.fp_reg_states[usize::from(reg.code())] {
            RegState::Empty => false,
            RegState::FromConst(reg_cidx) => match op {
                Operand::Const(op_cidx) => reg_cidx == *op_cidx,
                Operand::Var(_) => false,
            },
            RegState::FromInst(reg_iidx) => match op {
                Operand::Const(_) => false,
                Operand::Var(op_iidx) => reg_iidx == *op_iidx,
            },
            RegState::Reserved => unreachable!(),
        }
    }

    /// Put the value for `op` into `reg`. It is assumed that the caller has already checked that
    /// the value for `op` is not already in `reg`.
    fn put_input_in_fp_reg(&mut self, asm: &mut Assembler, op: &Operand, reg: Rx) {
        debug_assert!(!self.is_input_in_fp_reg(op, reg));
        let st = match op {
            Operand::Const(cidx) => {
                self.load_const_into_fp_reg(asm, *cidx, reg);
                RegState::FromConst(*cidx)
            }
            Operand::Var(iidx) => {
                self.force_fp_unspill(asm, *iidx, reg);
                RegState::FromInst(*iidx)
            }
        };
        self.fp_regset.set(reg);
        self.fp_reg_states[usize::from(reg.code())] = st;
    }

    /// Move the value in `old_reg` to `new_reg`, setting `old_reg` to [RegState::Empty].
    fn move_fp_reg(&mut self, asm: &mut Assembler, old_reg: Rx, new_reg: Rx) {
        dynasm!(asm; movsd Rx(new_reg.code()), Rx(old_reg.code()));
        self.fp_regset.set(new_reg);
        self.fp_reg_states[usize::from(new_reg.code())] = mem::replace(
            &mut self.fp_reg_states[usize::from(old_reg.code())],
            RegState::Empty,
        );
        self.fp_regset.unset(old_reg);
    }

    /// Swap the values, and register states, for `old_reg` and `new_reg`.
    fn swap_fp_reg(&mut self, asm: &mut Assembler, old_reg: Rx, new_reg: Rx) {
        dynasm!(asm
            ; pxor Rx(old_reg.code()), Rx(new_reg.code())
            ; pxor Rx(new_reg.code()), Rx(old_reg.code())
            ; pxor Rx(old_reg.code()), Rx(new_reg.code())
        );
        self.fp_reg_states
            .swap(usize::from(old_reg.code()), usize::from(new_reg.code()));
    }

    /// We are about to clobber `old_reg`, so if its value is needed later (1) move it to another
    /// register if there's a spare available or (2) ensure it is already spilled or (2) spill it.
    fn move_or_spill_fp(
        &mut self,
        asm: &mut Assembler,
        cur_iidx: InstIdx,
        avoid: &mut RegSet<Rx>,
        old_reg: Rx,
    ) {
        match self.fp_reg_states[usize::from(old_reg.code())] {
            RegState::Empty => (),
            RegState::FromConst(_) => (),
            RegState::FromInst(query_iidx) => {
                if self
                    .rev_an
                    .is_inst_var_still_used_after(cur_iidx, query_iidx)
                {
                    match self.fp_regset.find_empty_avoiding(*avoid) {
                        Some(new_reg) => {
                            dynasm!(asm; movsd Rx(new_reg.code()), Rx(old_reg.code()));
                            avoid.set(new_reg);
                            self.fp_regset.set(new_reg);
                            self.fp_reg_states[usize::from(new_reg.code())] =
                                self.fp_reg_states[usize::from(old_reg.code())].clone();
                        }
                        None => self.spill_fp_if_not_already(asm, old_reg),
                    }
                }
            }
            RegState::Reserved => unreachable!(),
        }
    }

    /// If the value stored in `reg` is not already spilled to the heap, then spill it. Note that
    /// this function neither writes to the register or changes the register's [RegState].
    fn spill_fp_if_not_already(&mut self, asm: &mut Assembler, reg: Rx) {
        match self.fp_reg_states[usize::from(reg.code())] {
            RegState::Reserved | RegState::Empty | RegState::FromConst(_) => (),
            RegState::FromInst(iidx) => {
                if self.spills[usize::from(iidx)] == SpillState::Empty {
                    let inst = self.m.inst(iidx);
                    let bitw = inst.def_bitw(self.m);
                    let bytew = inst.def_byte_size(self.m);
                    debug_assert!(bitw >= bytew);
                    self.stack.align(bytew);
                    let frame_off = self.stack.grow(bytew);
                    let off = i32::try_from(frame_off).unwrap();
                    match bitw {
                        32 => dynasm!(asm ; movss [rbp - off], Rx(reg.code())),
                        64 => dynasm!(asm ; movsd [rbp - off], Rx(reg.code())),
                        _ => unreachable!(),
                    }
                    self.spills[usize::from(iidx)] = SpillState::Stack(off);
                }
            }
        }
    }

    /// Load the value for `iidx` from the stack into `reg`.
    ///
    /// # Panics
    ///
    /// If `iidx` has not previously been spilled.
    fn force_fp_unspill(&mut self, asm: &mut Assembler, iidx: InstIdx, reg: Rx) {
        let inst = self.m.inst(iidx);
        let size = inst.def_byte_size(self.m);

        match self.spills[usize::from(iidx)] {
            SpillState::Empty => {
                let reg_i = self
                    .fp_reg_states
                    .iter()
                    .position(|x| {
                        if let RegState::FromInst(y) = x {
                            *y == iidx
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let cur_reg = FP_REGS[reg_i];
                if cur_reg != reg {
                    dynasm!(asm; movsd Rx(reg.code()), Rx(cur_reg.code()));
                }
            }
            SpillState::Stack(off) => {
                match size {
                    4 => dynasm!(asm; movss Rx(reg.code()), [rbp - off]),
                    8 => dynasm!(asm; movsd Rx(reg.code()), [rbp - off]),
                    _ => todo!("{}", size),
                };
                self.fp_regset.set(reg);
            }
            SpillState::Direct(_off) => todo!(),
            SpillState::ConstInt { bits: _bits, v: _v } => {
                todo!()
            }
        }
    }

    /// Load the constant from `cidx` into `reg`.
    fn load_const_into_fp_reg(&mut self, asm: &mut Assembler, cidx: ConstIdx, reg: Rx) {
        match self.m.const_(cidx) {
            Const::Float(tyidx, val) => {
                // FIXME: We need to use a temporary GP register to move immediate values into but
                // we don't have a reliable way of expressing this to the register allocator at
                // this point. We pick a random GP register and push/pop to avoid clobbering it.
                // This is not just inefficient but also highlights a weakness in our general
                // register allocator API.
                let tmp_reg = Rq::RAX;
                match self.m.type_(*tyidx) {
                    Ty::Float(fty) => match fty {
                        FloatTy::Float => {
                            dynasm!(asm
                                ; push Rq(tmp_reg.code())
                                ; mov Rd(tmp_reg.code()), DWORD (*val as f32).to_bits() as i64 as i32
                                ; movd Rx(reg.code()), Rd(tmp_reg.code())
                                ; pop Rq(tmp_reg.code())
                            );
                        }
                        FloatTy::Double => {
                            dynasm!(asm
                                ; push Rq(tmp_reg.code())
                                ; mov Rq(tmp_reg.code()), QWORD val.to_bits() as i64
                                ; movq Rx(reg.code()), Rq(tmp_reg.code())
                                ; pop Rq(tmp_reg.code())
                            );
                        }
                    },
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }
}

/// What constraints are there on registers for an instruction?
///
/// In the following `R` is a fixed register specified inside the variant, whereas *x* is an
/// unspecified register determined by the allocator.
#[derive(Clone, Debug)]
pub(crate) enum RegConstraint<R: dynasmrt::Register> {
    /// Make sure `Operand` is loaded into a register *x* on entry; its value must be unchanged
    /// after the instruction is executed.
    Input(Operand),
    /// Make sure `Operand` is loaded into register `R` on entry; its value must be unchanged
    /// after the instruction is executed.
    InputIntoReg(Operand, R),
    /// Make sure `Operand` is loaded into register `R` on entry and considered clobbered on exit.
    InputIntoRegAndClobber(Operand, R),
    /// Make sure `Operand` is loaded into a register *x* on entry and considered clobbered on exit.
    /// The result of this instruction will be stored in register *x*.
    InputOutput(Operand),
    /// Make sure `Operand` is loaded into register `R` on entry and considered clobbered on exit.
    /// The result of this instruction will be placed in `R`.
    InputOutputIntoReg(Operand, R),
    /// The result of this instruction will be stored in register *x*.
    Output,
    /// The result of this instruction will be stored in register *x*. That register can be the
    /// same as the register used for an `Input(Operand)` constraint, or it can be a different
    /// register, depending on what the register allocator considers best.
    ///
    /// Note: the `Operand` in an `OutputCanBeSameAsInput` is used to search for an `Input`
    /// constraint with the same `Operand`. In other words, the `Operand` in an
    /// `OutputCanBeSameAsInput` is not used directly in the allocator.
    OutputCanBeSameAsInput(Operand),
    /// The result of this instruction will be stored in register `R`.
    OutputFromReg(R),
    /// The register `R` will be clobbered.
    Clobber(R),
    /// A temporary register *x* that the instruction will clobber.
    Temporary,
    /// A no-op register constraint. A random register will be assigned to this: using this
    /// register for any purposes leads to undefined behaviour.
    None,
}

#[cfg(debug_assertions)]
impl<R: dynasmrt::Register> RegConstraint<R> {
    /// Return a reference to the inner [Operand], if there is one.
    fn operand(&self) -> Option<&Operand> {
        match self {
            Self::Input(o)
            | Self::InputIntoReg(o, _)
            | Self::InputIntoRegAndClobber(o, _)
            | Self::InputOutput(o)
            | Self::InputOutputIntoReg(o, _) => Some(o),
            Self::Output
            | Self::OutputCanBeSameAsInput(_)
            | Self::OutputFromReg(_)
            | Self::Clobber(_)
            | Self::Temporary
            | Self::None => None,
        }
    }
}

#[derive(Clone, Debug)]
enum RegState {
    Reserved,
    Empty,
    FromConst(ConstIdx),
    FromInst(InstIdx),
}

/// Which registers in a set of 16 registers are currently used? Happily 16 bits is the right size
/// for (separately) both x64's general purpose and floating point registers.
#[derive(Clone, Copy, Debug)]
struct RegSet<R>(u16, PhantomData<R>);

impl<R: dynasmrt::Register> RegSet<R> {
    /// Create a [RegSet] with all registers unused.
    fn blank() -> Self {
        Self(0, PhantomData)
    }

    fn is_set(&self, reg: R) -> bool {
        self.0 & (1 << u16::from(reg.code())) != 0
    }

    fn set(&mut self, reg: R) {
        self.0 |= 1 << u16::from(reg.code());
    }

    fn unset(&mut self, reg: R) {
        self.0 &= !(1 << u16::from(reg.code()));
    }
}

impl RegSet<Rq> {
    /// Create a [RegSet] with the reserved general purpose registers in [RESERVED_GP_REGS] set.
    fn with_gp_reserved() -> Self {
        let mut new = Self::blank();
        for reg in RESERVED_GP_REGS {
            new.set(reg);
        }
        new
    }

    fn find_empty(&self) -> Option<Rq> {
        if self.0 == u16::MAX {
            None
        } else {
            // The lower registers (e.g. RAX) are those most likely to be used by x64 instructions,
            // so we prefer to give out the highest possible registers (e.g. R15).
            Some(GP_REGS[usize::try_from(15 - (!self.0).leading_zeros()).unwrap()])
        }
    }

    fn find_empty_avoiding(&self, avoid: RegSet<Rq>) -> Option<Rq> {
        let x = self.0 | avoid.0;
        if x == u16::MAX {
            None
        } else {
            // The lower registers (e.g. RAX) are those most likely to be used by x64 instructions,
            // so we prefer to give out the highest possible registers (e.g. R15).
            Some(GP_REGS[usize::try_from(15 - (!x).leading_zeros()).unwrap()])
        }
    }

    pub(crate) fn from_vec(regs: &[Rq]) -> Self {
        let mut s = Self::blank();
        for reg in regs {
            s.set(*reg);
        }
        s
    }
}

impl From<Rq> for RegSet<Rq> {
    fn from(reg: Rq) -> Self {
        Self(1 << u16::from(reg.code()), PhantomData)
    }
}

impl RegSet<Rx> {
    /// Create a [RegSet] with the reserved floating point registers in [RESERVED_FP_REGS] set.
    fn with_fp_reserved() -> Self {
        let mut new = Self::blank();
        for reg in RESERVED_FP_REGS {
            new.set(reg);
        }
        new
    }

    fn find_empty(&self) -> Option<Rx> {
        if self.0 == u16::MAX {
            None
        } else {
            // Could start from 0 rather than 15.
            Some(FP_REGS[usize::try_from(15 - (!self.0).leading_zeros()).unwrap()])
        }
    }

    fn find_empty_avoiding(&self, avoid: RegSet<Rx>) -> Option<Rx> {
        let x = self.0 | avoid.0;
        if x == u16::MAX {
            None
        } else {
            // Could start from 0 rather than 15.
            Some(FP_REGS[usize::try_from(15 - (!x).leading_zeros()).unwrap()])
        }
    }
}

impl From<Rx> for RegSet<Rx> {
    fn from(reg: Rx) -> Self {
        Self(1 << u16::from(reg.code()), PhantomData)
    }
}

/// The spill state of an SSA variable: is it spilled? If so, where?
#[derive(Clone, Copy, Debug, PartialEq)]
enum SpillState {
    /// This variable has not yet been spilt, or has been spilt and will not be used again.
    Empty,
    /// This variable is spilt to the stack with the same semantics as [VarLocation::Stack].
    ///
    /// Note: two SSA variables can alias to the same `Stack` location.
    Stack(i32),
    /// This variable is spilt to the stack with the same semantics as [VarLocation::Direct].
    ///
    /// Note: two SSA variables can alias to the same `Direct` location.
    Direct(i32),
    /// This variable is a constant.
    ConstInt { bits: u32, v: u64 },
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn regset() {
        let mut x = RegSet::blank();
        debug_assert!(!x.is_set(Rq::R15));
        debug_assert_eq!(x.find_empty(), Some(Rq::R15));
        x.set(Rq::R15);
        debug_assert!(x.is_set(Rq::R15));
        debug_assert_eq!(x.find_empty(), Some(Rq::R14));
        x.set(Rq::R14);
        debug_assert_eq!(x.find_empty(), Some(Rq::R13));
        x.unset(Rq::R14);
        debug_assert_eq!(x.find_empty(), Some(Rq::R14));
        for reg in GP_REGS {
            x.set(reg);
            debug_assert!(x.is_set(reg));
        }
        debug_assert_eq!(x.find_empty(), None);
        x.unset(Rq::RAX);
        debug_assert!(!x.is_set(Rq::RAX));
        debug_assert_eq!(x.find_empty(), Some(Rq::RAX));

        let x = RegSet::<Rq>::blank();
        debug_assert_eq!(x.find_empty_avoiding(RegSet::from(Rq::R15)), Some(Rq::R14));
    }
}
