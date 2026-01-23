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

use super::{Register, VarLocation, rev_analyse::RevAnalyse};
use crate::compile::jitc_yk::{
    aot_ir,
    codegen::abs_stack::AbstractStack,
    jit_ir::{
        Const, ConstIdx, FloatTy, HasGuardInfo, Inst, InstIdx, Module, Operand, PtrAddInst, Ty,
    },
};
use dynasmrt::{
    DynasmApi, Register as dynasmrtRegister, dynasm,
    x64::{
        Assembler, {Rq, Rx},
    },
};
use std::{assert_matches, cmp::Ordering, marker::PhantomData, mem};

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
pub(super) static RESERVED_GP_REGS: [Rq; 2] = [Rq::RSP, Rq::RBP];

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
            match &mut self.gp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => {
                    debug_assert!(self.gp_regset.is_set(reg));
                }
                RegState::Empty => {
                    debug_assert!(!self.gp_regset.is_set(reg));
                }
                RegState::FromConst(_, _) => {
                    debug_assert!(self.gp_regset.is_set(reg));
                    // FIXME: Don't immediately expire constants
                    self.gp_regset.unset(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegState::FromInst(iidxs, _) => {
                    assert!(self.gp_regset.is_set(reg));
                    iidxs.retain(|x| self.rev_an.is_inst_var_still_used_at(iidx, *x));
                    if iidxs.is_empty() {
                        self.gp_regset.unset(reg);
                        *self.gp_reg_states.get_mut(usize::from(reg.code())).unwrap() =
                            RegState::Empty;
                    }
                }
            }
        }

        for reg in FP_REGS {
            match &mut self.fp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => {
                    debug_assert!(self.fp_regset.is_set(reg));
                }
                RegState::Empty => {
                    debug_assert!(!self.fp_regset.is_set(reg));
                }
                RegState::FromConst(_, _) => {
                    debug_assert!(self.fp_regset.is_set(reg));
                    // FIXME: Don't immediately expire constants
                    self.fp_regset.unset(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegState::FromInst(iidxs, _) => {
                    assert!(self.fp_regset.is_set(reg));
                    iidxs.retain(|x| self.rev_an.is_inst_var_still_used_at(iidx, *x));
                    if iidxs.is_empty() {
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
    ///
    /// Note: this can change CPU flags. It is therefore undefined behaviour to check CPU flags
    /// after this function has been called.
    pub(crate) fn assign_regs<const NG: usize, const NF: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        gp_constraints: [GPConstraint; NG],
        fp_constraints: [RegConstraint<Rx>; NF],
    ) -> ([Rq; NG], [Rx; NF]) {
        (
            self.assign_gp_regs(asm, iidx, gp_constraints),
            self.assign_fp_regs(asm, iidx, fp_constraints),
        )
    }

    /// Return a [GuardSnapshot] for the guard `ginst`. This should be called when generating code
    /// for a guard: it returns the information the register allocator will later need to perform
    /// its part in generating correct code for this guard's failure in
    /// [Self::get_ready_for_deopt].
    pub(super) fn guard_snapshot(&self) -> GuardSnapshot {
        GuardSnapshot {
            gp_regset: self.gp_regset,
            gp_reg_states: self.gp_reg_states.clone(),
            fp_regset: self.fp_regset,
            fp_reg_states: self.fp_reg_states.clone(),
            spills: self.spills.clone(),
            stack: self.stack.clone(),
        }
    }

    pub(super) fn restore_guard_snapshot(&mut self, gsnap: GuardSnapshot) {
        self.gp_regset = gsnap.gp_regset;
        self.gp_reg_states = gsnap.gp_reg_states;
        self.fp_regset = gsnap.fp_regset;
        self.fp_reg_states = gsnap.fp_reg_states;
        self.spills = gsnap.spills;
        self.stack = gsnap.stack;
    }

    /// When generating the code for a guard failure, do the necessary work from the register
    /// allocator's perspective (e.g. ensuring registers have an appropriate [RegExtension]) for
    /// deopt to occur.
    pub(super) fn get_ready_for_deopt(
        &mut self,
        asm: &mut Assembler,
        ginst: HasGuardInfo,
    ) -> (Rq, Vec<(aot_ir::InstId, VarLocation)>) {
        let patch_reg = self.force_tmp_register(asm, RegSet::with_gp_reserved());

        let gi = ginst.guard_info(self.m);
        // `seen_gp_regs` allows us to zero extend a register at most once.
        let mut seen_gp_regs = RegSet::with_gp_reserved();
        let mut lives = Vec::with_capacity(gi.live_vars().len());
        for (iid, pop) in gi.live_vars() {
            let op = pop.unpack(self.m);
            match op {
                Operand::Var(x) => {
                    if let Some(reg) = self.find_op_in_gp_reg(&op)
                        && !seen_gp_regs.is_set(reg)
                    {
                        assert_ne!(patch_reg, reg);
                        let RegState::FromInst(ref insts, ext) =
                            self.gp_reg_states[usize::from(reg.code())]
                        else {
                            panic!()
                        };
                        let bitw = insts
                            .iter()
                            .map(|x| self.m.inst_nocopy(*x).unwrap().def_bitw(self.m))
                            .max()
                            .unwrap();
                        if ext != RegExtension::ZeroExtended {
                            self.force_zero_extend_to_reg64(asm, reg, bitw);
                            seen_gp_regs.set(reg);
                        }
                    }
                    lives.push((iid.clone(), self.var_location(x)));
                }
                Operand::Const(x) => {
                    // The live variable is a constant (e.g. this can happen during inlining), so
                    // it doesn't have an allocation. We can just push the actual value instead
                    // which will be written as is during deoptimisation.
                    match self.m.const_(x) {
                        Const::Int(_, y) => lives.push((
                            iid.clone(),
                            VarLocation::ConstInt {
                                bits: y.bitw(),
                                v: y.to_zero_ext_u64().unwrap(),
                            },
                        )),
                        Const::Ptr(p) => lives.push((
                            iid.clone(),
                            VarLocation::ConstInt {
                                bits: 64,
                                v: u64::try_from(*p).unwrap(),
                            },
                        )),
                        e => todo!("{:?}", e),
                    }
                }
            }
        }

        (patch_reg, lives)
    }
}

/// The parts of the register allocator needed for general purpose registers.
impl LSRegAlloc<'_> {
    /// Forcibly assign the machine register `reg` to the value produced by instruction `iidx`.
    /// Note that if this register is already used, a spill will be generated instead.
    pub(crate) fn force_assign_inst_gp_reg(
        &mut self,
        _asm: &mut Assembler,
        iidx: InstIdx,
        reg: Rq,
    ) {
        if self.gp_regset.is_set(reg) {
            match &mut self.gp_reg_states[usize::from(reg.code())] {
                RegState::Reserved | RegState::Empty => unreachable!(),
                RegState::FromConst(_, _) => todo!(),
                RegState::FromInst(iidxs, _) => {
                    // We have to assume that if LLVM told us that multiple instructions can live
                    // in a single register that they can do safely.
                    iidxs.push(iidx);
                }
            }
        } else {
            self.gp_regset.set(reg);
            self.gp_reg_states[usize::from(reg.code())] =
                RegState::FromInst(vec![iidx], RegExtension::Undefined);
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
        let bytew = inst.def_byte_size(self.m);
        self.stack.align(bytew); // FIXME
        let frame_off = self.stack.grow(bytew);
        let off = i32::try_from(frame_off).unwrap();
        match inst.def_bitw(self.m) {
            1 => {
                assert_matches!(self.gp_reg_states[usize::from(reg.code())], RegState::Empty);
                self.force_zero_extend_to_reg64(asm, reg, 1);
                dynasm!(asm; mov BYTE [rbp - off], Rb(reg.code()));
            }
            8 => dynasm!(asm; mov BYTE [rbp - off], Rb(reg.code())),
            16 => dynasm!(asm; mov WORD [rbp - off], Rw(reg.code())),
            32 => dynasm!(asm; mov DWORD [rbp - off], Rd(reg.code())),
            64 => dynasm!(asm; mov QWORD [rbp - off], Rq(reg.code())),
            x => todo!("{x}"),
        }
        self.spills[usize::from(iidx)] = SpillState::Stack(off);
    }

    /// Forcibly assign the floating point register `reg`, which must be in the [RegState::Empty] state,
    /// to the value produced by instruction `iidx`.
    pub(crate) fn force_assign_inst_fp_reg(&mut self, iidx: InstIdx, reg: Rx) {
        debug_assert!(!self.fp_regset.is_set(reg));
        self.fp_regset.set(reg);
        self.fp_reg_states[usize::from(reg.code())] =
            RegState::FromInst(vec![iidx], RegExtension::Undefined);
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

    /// Forcibly assign a constant integer to an instruction. This typically only happens when
    /// traces pass live variables that have been optimised to constants into side-traces.
    pub(crate) fn assign_const_int(&mut self, iidx: InstIdx, bits: u32, v: u64) {
        self.spills[usize::from(iidx)] = SpillState::ConstInt { bits, v };
    }

    /// Forcibly assign a constant pointer to an instruction. This typically only happens when
    /// traces pass live variables that have been optimised to constants into side-traces.
    pub(crate) fn assign_const_ptr(&mut self, iidx: InstIdx, v: usize) {
        self.spills[usize::from(iidx)] = SpillState::ConstPtr(v);
    }

    /// Return a currently unused general purpose register, if one exists.
    ///
    /// Note: you must be *very* careful in when you use the register that's returned.
    /// Specifically, if you call this after `assign_regs` and use the register before you have
    /// generated outputs, undefined behaviour will occur.
    pub(super) fn find_empty_gp_reg(&self) -> Option<Rq> {
        for (reg_i, rs) in self.gp_reg_states.iter().enumerate() {
            if let RegState::Empty = rs {
                return Some(GP_REGS[reg_i]);
            }
        }
        None
    }

    /// Forcibly obtain a register, spilling whatever's in there, even if it is "used" by the
    /// current instruction. This is suitable for guards / calls / etc. The register returned is
    /// guaranteed not to be in the set `avoid`.
    fn force_tmp_register(&mut self, asm: &mut Assembler, avoid: RegSet<Rq>) -> Rq {
        for (reg_i, rs) in self.gp_reg_states.iter().enumerate() {
            if avoid.is_set(GP_REGS[reg_i]) {
                continue;
            }
            if let RegState::Empty = rs {
                // The happy case: an empty register is already available!
                return GP_REGS[reg_i];
            }
        }

        // The moderately happy case: a register with a constant in it, or whose instructions are
        // all already spilled.
        for (reg_i, rs) in self.gp_reg_states.iter().enumerate() {
            if avoid.is_set(GP_REGS[reg_i]) {
                continue;
            }
            match rs {
                RegState::Reserved => (),
                RegState::Empty => todo!(),
                RegState::FromConst(_, _) => {
                    self.gp_reg_states[reg_i] = RegState::Empty;
                    self.gp_regset.unset(GP_REGS[reg_i]);
                    return GP_REGS[reg_i];
                }
                RegState::FromInst(from_iidxs, _) => {
                    if from_iidxs
                        .iter()
                        .all(|x| self.spills[usize::from(*x)] != SpillState::Empty)
                    {
                        self.gp_reg_states[reg_i] = RegState::Empty;
                        self.gp_regset.unset(GP_REGS[reg_i]);
                        return GP_REGS[reg_i];
                    }
                }
            }
        }

        // The not happy case: we have to pick a random register and spill it.
        let reg = avoid.find_empty().unwrap();
        self.force_spill_gp(asm, false, reg);
        self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
        self.gp_regset.unset(reg);
        reg
    }

    /// Return the register a normal guard instruction needs. This function guarantees not to set
    /// CPU flags. It is not suitable for use outside `cg_guard`.
    pub(super) fn tmp_register_for_guard(
        &mut self,
        asm: &mut Assembler,
        _iidx: InstIdx,
        cond: Operand,
    ) -> Rq {
        match self.find_op_in_gp_reg(&cond) {
            Some(x) => x,
            None => {
                let reg = self.force_tmp_register(asm, RegSet::with_gp_reserved());
                self.put_input_in_gp_reg(asm, &cond, reg, RegExtension::Undefined);
                reg
            }
        }
    }

    /// Return the `patch register` a combined icmp/guard instruction needs. This function
    /// guarantees not to set CPU flags. It is not suitable for use outside `cg_icmp_guard`.
    pub(super) fn tmp_register_for_icmp_guard(
        &mut self,
        asm: &mut Assembler,
        _iidx: InstIdx,
    ) -> Rq {
        self.force_tmp_register(asm, RegSet::with_gp_reserved())
    }

    /// Return a temporary register suitable for `write_vars`. Note: this might cause the value
    /// originally in the returned value to be spilled.
    pub(super) fn tmp_register_for_write_vars(&mut self, asm: &mut Assembler) -> Rq {
        self.find_empty_gp_reg()
            .unwrap_or_else(|| self.force_tmp_register(asm, RegSet::with_gp_reserved()))
    }

    /// Assign general purpose registers for the instruction at position `iidx`.
    ///
    /// This is a convenience function for [Self::assign_regs] when there are no FP registers.
    pub(crate) fn assign_gp_regs<const N: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        cnstrs: [GPConstraint; N],
    ) -> [Rq; N] {
        // Register assignment is split into three stages:
        //   1. Find a register for each `GPConstraint`. No state changes to `self` are made.
        //   2. Decide how we shuffle the existing registers around. No state changes to `self` are
        //      made.
        //   3. Generate code to shuffle the existing registers around, and update the output
        //      state. State changes to `self` are made during this stage.

        // No constraint operands should be float-typed.
        assert_eq!(
            cnstrs
                .iter()
                .filter(|x| match x {
                    GPConstraint::Input { op, .. }
                    | GPConstraint::InputOutput { op, .. }
                    | GPConstraint::AlignExtension { op, .. } =>
                        matches!(self.m.type_(op.tyidx(self.m)), Ty::Float(_)),
                    GPConstraint::Output { .. }
                    | GPConstraint::Clobber { .. }
                    | GPConstraint::Temporary
                    | GPConstraint::None => false,
                })
                .count(),
            0
        );

        // There must be at most 1 output register.
        assert!(
            cnstrs
                .iter()
                .filter(|x| match x {
                    GPConstraint::InputOutput { .. }
                    | GPConstraint::Output { .. }
                    | GPConstraint::AlignExtension { .. } => true,
                    GPConstraint::Input { .. }
                    | GPConstraint::Clobber { .. }
                    | GPConstraint::Temporary
                    | GPConstraint::None => false,
                })
                .count()
                <= 1
        );

        // Stage 1: Find the register we will use for each `GPConstraint`. Note: we may use the
        // same register for multiple constraints.

        let (asgn_regs, cnstr_regs) = self.find_regs_for_constraints(iidx, &cnstrs);
        assert_eq!(
            cnstr_regs
                .iter()
                .filter(|reg| asgn_regs.is_set(**reg))
                .count(),
            cnstr_regs.len()
        );

        // Stage 2: At this point, we've found a register for every constraint. Decide how we will
        // make use of values in existing registers. Note: this stage does not update any state in
        // `self`.

        // Stage 2.1: For every operand that is in a register -- even if not the right register! --
        // generate copies for it.
        let copies = self.input_regs_to_copies(iidx, &cnstrs, &cnstr_regs);
        let actions = reg_copies_to_actions(copies);

        // Stage 2.2: the copies that we must do above may end up clobbering existing registers.
        // If those registers contain values we still need, move them to another register (if there
        // are spare/empty registers) or otherwise ensure those values are spilled.
        let actions = self.move_or_spill_clobbered_regs(iidx, &actions, &cnstrs, &cnstr_regs);

        // Stage 3: Generate code and update `self`.

        // Stage 3.1: Go through the sequence of actions, copying / spilling values.
        for (reg, action) in &actions {
            match action {
                RegAction::Keep => (),
                RegAction::CopyFrom(from_reg) => {
                    assert_ne!(from_reg, reg);
                    dynasm!(asm; mov Rq(reg.code()), Rq(from_reg.code()));
                    self.gp_regset.set(*reg);
                    let st = self.gp_reg_states[usize::from(from_reg.code())].clone();
                    self.gp_reg_states[usize::from(reg.code())] = st;
                }
                RegAction::Spill => {
                    self.spill_gp_if_not_already(asm, iidx, *reg);
                }
            }
        }

        // Stage 3.2: For any input constraints that don't yet contain a value, unspill the value.
        // Note: by definition, if stage 3.1 didn't put a value in a given input register, the only
        // possibility is that the value has either previously been spilled or is a constant.
        for (cnstr, reg) in cnstrs.iter().zip(cnstr_regs.into_iter()) {
            match cnstr {
                GPConstraint::Input { op, in_ext, .. }
                | GPConstraint::InputOutput { op, in_ext, .. } => {
                    if self.is_input_in_gp_reg(op, reg) {
                        self.align_extensions(asm, reg, *in_ext);
                    } else {
                        self.put_input_in_gp_reg(asm, op, reg, *in_ext);
                    }
                }
                GPConstraint::AlignExtension { op, out_ext } => {
                    if self.is_input_in_gp_reg(op, reg) {
                        self.align_extensions(asm, reg, *out_ext);
                    } else {
                        self.put_input_in_gp_reg(asm, op, reg, *out_ext);
                    }
                    match op {
                        Operand::Const(cidx) => {
                            let ss = match self.m.const_(*cidx) {
                                Const::Float(_ty_idx, _) => todo!(),
                                Const::Int(_ty_idx, arb_bit_int) => SpillState::ConstInt {
                                    bits: arb_bit_int.bitw(),
                                    v: arb_bit_int.to_zero_ext_u64().unwrap(),
                                },
                                Const::Ptr(_) => todo!(),
                            };
                            self.spills[usize::from(iidx)] = ss;
                        }
                        Operand::Var(_) => {
                            let RegState::FromInst(iidxs, ext) =
                                &mut self.gp_reg_states[usize::from(reg.code())]
                            else {
                                panic!()
                            };
                            assert_eq!(out_ext, ext);
                            iidxs.push(iidx);
                        }
                    }
                }
                GPConstraint::Output { .. }
                | GPConstraint::Clobber { force_reg: _ }
                | GPConstraint::Temporary
                | GPConstraint::None => (),
            }
        }

        // Set the output state for constraints.
        for (cnstr, reg) in cnstrs.into_iter().zip(cnstr_regs.into_iter()) {
            match cnstr {
                GPConstraint::Input { clobber_reg, .. } => {
                    assert!(self.gp_regset.is_set(reg));
                    if clobber_reg {
                        self.gp_regset.unset(reg);
                        self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                    }
                }
                GPConstraint::InputOutput { out_ext, .. } => {
                    assert!(self.gp_regset.is_set(reg));
                    self.gp_reg_states[usize::from(reg.code())] =
                        RegState::FromInst(vec![iidx], out_ext);
                }
                GPConstraint::Output { out_ext, .. } => {
                    self.gp_regset.set(reg);
                    self.gp_reg_states[usize::from(reg.code())] =
                        RegState::FromInst(vec![iidx], out_ext);
                }
                GPConstraint::AlignExtension { .. } => {
                    assert!(self.gp_regset.is_set(reg));
                }
                GPConstraint::Clobber { .. } | GPConstraint::Temporary => {
                    self.gp_regset.unset(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                GPConstraint::None => (),
            }
        }

        cnstr_regs
    }

    /// Stage 1 in register allocation: find a register for each `GPConstraint`. Return the
    /// `RegSet` of assigned registers and the `Rq` registers: note these two sets are in a sense
    /// equivalent (the former can be derived from the latter), but having both makes other things
    /// more convenient.
    fn find_regs_for_constraints<const N: usize>(
        &self,
        iidx: InstIdx,
        constraints: &[GPConstraint; N],
    ) -> (RegSet<Rq>, [Rq; N]) {
        // This stage is split into sub-stages:
        //   a. Use `force_reg`s, where specified.
        //   b. Deal with register hints and `can_be_same_as_input`

        // The register we are assigning to each constraint.
        let mut cnstr_regs = [None; N];
        // The registers we have assigned so far. This is a strict superset of the registers
        // contained in `cnstr_regs` because there are some registers we cannot assign under any
        // circumstances.
        let mut asgn_regs = RegSet::with_gp_reserved();

        // Where the caller has told us they want to put things in specific registers, we need to
        // make sure we avoid assigning those in all other circumstances. Note: any given register
        // can appear in at most one `force_reg`.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                GPConstraint::Input { force_reg, .. }
                | GPConstraint::InputOutput { force_reg, .. }
                | GPConstraint::Output { force_reg, .. } => {
                    if let Some(reg) = force_reg {
                        assert!(!asgn_regs.is_set(*reg));
                        cnstr_regs[i] = Some(*reg);
                        asgn_regs.set(*reg);
                    }
                }
                GPConstraint::Clobber { force_reg } => {
                    assert!(!asgn_regs.is_set(*force_reg));
                    cnstr_regs[i] = Some(*force_reg);
                    asgn_regs.set(*force_reg);
                }
                GPConstraint::AlignExtension { .. }
                | GPConstraint::Temporary
                | GPConstraint::None => (),
            }
        }

        // If we have a hint for an output constraint, use it.
        for (i, cnstr) in constraints.iter().enumerate() {
            if cnstr_regs[i].is_some() {
                continue;
            }
            match cnstr {
                GPConstraint::Output { .. }
                | GPConstraint::InputOutput { .. }
                | GPConstraint::AlignExtension { .. } => {
                    if let Some(Register::GP(reg)) = self.rev_an.reg_hint(iidx, iidx)
                        && !asgn_regs.is_set(reg)
                    {
                        cnstr_regs[i] = Some(reg);
                        asgn_regs.set(reg);
                    }
                }
                _ => (),
            }
        }

        // If we already have the input operand in a register, don't assign a new register. Note:
        // multiple `Input` constraints might reference the same operand, so we may "reuse" the
        // same register multiple times. Because of that we can't update `avoid` straight away: we
        // do it in one go once we've processed all the input operands.
        let mut input_regs = RegSet::blank();
        for (i, cnstr) in constraints.iter().enumerate() {
            if cnstr_regs[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            match cnstr {
                GPConstraint::Input { op, .. } | GPConstraint::InputOutput { op, .. } => {
                    if let Some(reg) = self.find_op_in_gp_reg(op)
                        && !asgn_regs.is_set(reg)
                    {
                        assert!(self.gp_regset.is_set(reg));
                        input_regs.set(reg);
                        cnstr_regs[i] = Some(reg);
                    }
                }
                GPConstraint::AlignExtension { op, out_ext } => {
                    // Right now, this is only meant for cg_sext and cg_zext, so there's no point
                    // in trying to handle the `out_ext == Regextension::Undefined` case.
                    assert_matches!(
                        out_ext,
                        RegExtension::SignExtended | RegExtension::ZeroExtended
                    );
                    for reg in self.find_op_in_gp_regs(op) {
                        if !asgn_regs.is_set(reg) {
                            assert!(self.gp_regset.is_set(reg));
                            input_regs.set(reg);
                            cnstr_regs[i] = Some(reg);
                        }
                    }
                }
                GPConstraint::Output { .. }
                | GPConstraint::Clobber { .. }
                | GPConstraint::Temporary => (),
                GPConstraint::None => {
                    // By definition it doesn't matter what register we "assign" here: it's
                    // ignored at any point of importance. To make it less likely that someone uses
                    // the value and doesn't notice, we "assign" a register that's likely to cause
                    // the program to explode if it's used.
                    cnstr_regs[i] = Some(RESERVED_GP_REGS[0]);
                }
            }
        }
        asgn_regs.union(input_regs);

        // If we have an `Output { can_be_same_as_input: true }` constraint, we have the option to
        // take advantage of an input constraint whose value we don't need later. Try and find such
        // a constraint.
        let mut reusable_input_cnstr = None;
        for (i, cnstr) in constraints.iter().enumerate() {
            if let GPConstraint::Input { op, .. } = cnstr {
                match op {
                    Operand::Var(query_iidx) => {
                        if !self.rev_an.is_inst_var_still_used_after(iidx, *query_iidx) {
                            reusable_input_cnstr = Some(i);
                        }
                    }
                    Operand::Const(_) => {
                        reusable_input_cnstr = Some(i);
                    }
                }
            }
        }

        // For input values we will need to unspill, put them in a hint register if possible.
        for (i, cnstr) in constraints.iter().enumerate() {
            if cnstr_regs[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            if let GPConstraint::Input { op, .. } | GPConstraint::InputOutput { op, .. } = cnstr {
                match op {
                    Operand::Var(query_iidx) => {
                        if let Some(Register::GP(reg)) = self.rev_an.reg_hint(iidx, *query_iidx)
                            && !asgn_regs.is_set(reg)
                        {
                            cnstr_regs[i] = Some(reg);
                            asgn_regs.set(reg);
                        }
                    }
                    Operand::Const(_) => (),
                }
            }
        }

        // Assign a register for all unassigned constraints (except `Output { can_be_same_as_input:
        // true }`).
        for (i, cnstr) in constraints.iter().enumerate() {
            if cnstr_regs[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            // If we have an `Output { can_be_same_as_input: true }` constraint *and* we have an
            // input constraint we can reuse, we don't need to allocate a register for the `Output`
            // constraint. We do, though, need to wait until the matching input constraint has an
            // input register, which it might not yet, so skip this constraint for now.
            if let GPConstraint::Output {
                can_be_same_as_input: true,
                ..
            } = cnstr
                && reusable_input_cnstr.is_some()
            {
                continue;
            }

            let reg = match self.gp_regset.find_empty_avoiding(asgn_regs) {
                Some(reg) => reg,
                None => {
                    let mut clobber_regs = asgn_regs.iter_unset_bits().collect::<Vec<_>>();
                    self.sort_clobber_regs(iidx, &mut clobber_regs);
                    *clobber_regs
                        .first()
                        .expect("Cannot satisfy register constraints: no registers left")
                }
            };
            cnstr_regs[i] = Some(reg);
            asgn_regs.set(reg);
        }

        // If there is an `Output { can_be_same_as_input: true }` register and a reusable input
        // constraint register, we can now match the two up.
        for (i, cnstr) in constraints.iter().enumerate() {
            if cnstr_regs[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            if let GPConstraint::Output {
                can_be_same_as_input: true,
                ..
            } = cnstr
            {
                cnstr_regs[i] = cnstr_regs[reusable_input_cnstr.unwrap()];
                break;
            }
        }

        (asgn_regs, cnstr_regs.map(|x| x.unwrap()))
    }

    /// For the registers we're willing to clobber `clobber_regs`, sort them so that the registers
    /// we're most willing to clobber are at the start of the list.
    fn sort_clobber_regs(&self, iidx: InstIdx, clobber_regs: &mut [Rq]) {
        // Our heuristic is (in order):
        // 1. Prefer to clobber registers whose values are unused in the future.
        // 2. Prefer to clobber constants.
        // 3. Prefer to clobber a register that is used further away in the trace.
        // 4. Prefer to clobber a register that is already spilled.
        // 5. Prefer to clobber a register whose value(s) are used the fewest subsequent times in
        //    the trace.
        // 6. Prefer to clobber a register that contains fewer variables.
        clobber_regs.sort_unstable_by(|lhs_reg, rhs_reg| {
            match (
                &self.gp_reg_states[usize::from(lhs_reg.code())],
                &self.gp_reg_states[usize::from(rhs_reg.code())],
            ) {
                (RegState::FromInst(lhs_iidxs, _), RegState::FromInst(rhs_iidxs, _)) => {
                    let lhs = lhs_iidxs
                        .iter()
                        .map(|y| {
                            (
                                self.rev_an.iter_uses_after(iidx, *y).count(),
                                self.rev_an.next_use(iidx, *y),
                            )
                        })
                        .collect::<Vec<_>>();
                    let lhs_next = lhs.iter().map(|(_, iidx)| iidx).min().unwrap();
                    let rhs = rhs_iidxs
                        .iter()
                        .map(|y| {
                            (
                                self.rev_an.iter_uses_after(iidx, *y).count(),
                                self.rev_an.next_use(iidx, *y),
                            )
                        })
                        .collect::<Vec<_>>();
                    let rhs_next = rhs.iter().map(|(_, iidx)| iidx).min().unwrap();

                    if lhs_next.is_none() && rhs_next.is_some() {
                        Ordering::Less
                    } else if lhs_next.is_some() && rhs_next.is_none() {
                        Ordering::Greater
                    } else if lhs_next != rhs_next {
                        lhs_next.cmp(rhs_next).reverse()
                    } else {
                        let lhs_spilled = lhs_iidxs
                            .iter()
                            .all(|x| !matches!(self.spills[usize::from(*x)], SpillState::Empty));
                        let rhs_spilled = rhs_iidxs
                            .iter()
                            .all(|x| !matches!(self.spills[usize::from(*x)], SpillState::Empty));

                        if lhs_spilled && !rhs_spilled {
                            Ordering::Less
                        } else if !lhs_spilled && rhs_spilled {
                            Ordering::Greater
                        } else {
                            let lhs_count = lhs.iter().map(|(count, _)| count).max().unwrap();
                            let rhs_count = rhs.iter().map(|(count, _)| count).max().unwrap();
                            lhs_count
                                .cmp(rhs_count)
                                .then(lhs_iidxs.len().cmp(&rhs_iidxs.len()))
                        }
                    }
                }
                (_, RegState::FromInst(_, _)) => Ordering::Less,
                (RegState::FromInst(_, _), _) => Ordering::Greater,
                (_, _) => Ordering::Equal,
            }
        });
    }

    /// For each input constraint in `cnstrs`, generate a register move if that constraint's
    /// operand is already in a register. The output array is suitable for passing to
    /// [reg_copies_to_actions].
    fn input_regs_to_copies<const N: usize>(
        &self,
        _iidx: InstIdx,
        cnstrs: &[GPConstraint; N],
        cnstr_regs: &[Rq; N],
    ) -> [Option<Rq>; 16] {
        let mut moves = [None; 16];
        // Find registers which contain operands we need to satisfy our constraints.
        for (cnstr, cnstr_reg) in cnstrs.iter().zip(cnstr_regs.iter()) {
            match cnstr {
                GPConstraint::Input { op, .. }
                | GPConstraint::InputOutput { op, .. }
                | GPConstraint::AlignExtension { op, .. } => {
                    if self.is_input_in_gp_reg(op, *cnstr_reg) {
                        // The very happy case: the operand is already in the right register.
                        moves[usize::from(cnstr_reg.code())] = Some(*cnstr_reg);
                    } else if let Some(op_reg) = self.find_op_in_gp_reg(op) {
                        // The moderately happy case: the operand is in a register, but not the
                        // right one.
                        moves[usize::from(cnstr_reg.code())] = Some(op_reg);
                    }
                }
                GPConstraint::Output { .. }
                | GPConstraint::Clobber { .. }
                | GPConstraint::Temporary
                | GPConstraint::None => (),
            }
        }
        moves
    }

    /// For a sequence of actions, identify those existing registers whose values will be clobbered
    /// and generate further actions (in dependency order!) that either: move them to an
    /// empty/spare register; spill them.
    fn move_or_spill_clobbered_regs(
        &self,
        iidx: InstIdx,
        actions: &[(Rq, RegAction)],
        cnstrs: &[GPConstraint],
        cnstr_regs: &[Rq],
    ) -> Vec<(Rq, RegAction)> {
        let mut out = Vec::new();
        let mut clobber_regs = RegSet::with_gp_reserved();
        let mut asgn_regs = RegSet::with_gp_reserved();

        for (to_reg, action) in actions {
            match action {
                RegAction::Keep => {
                    asgn_regs.set(*to_reg);
                }
                RegAction::CopyFrom(from_reg) => {
                    clobber_regs.set(*to_reg);
                    asgn_regs.set(*to_reg);
                    asgn_regs.set(*from_reg);
                }
                RegAction::Spill => todo!(),
            }
        }

        assert_eq!(cnstrs.len(), cnstr_regs.len());
        for (cnstr, cnstr_reg) in cnstrs.iter().zip(cnstr_regs) {
            match cnstr {
                GPConstraint::Input {
                    op,
                    clobber_reg: false,
                    ..
                }
                | GPConstraint::AlignExtension { op, out_ext: _ } => {
                    if !self.is_input_in_gp_reg(op, *cnstr_reg) {
                        clobber_regs.set(*cnstr_reg);
                    }
                    asgn_regs.set(*cnstr_reg);
                }
                GPConstraint::Input {
                    clobber_reg: true, ..
                }
                | GPConstraint::InputOutput { .. }
                | GPConstraint::Output { .. }
                | GPConstraint::Clobber { .. }
                | GPConstraint::Temporary => {
                    clobber_regs.set(*cnstr_reg);
                    asgn_regs.set(*cnstr_reg);
                }
                _ => (),
            }
        }

        // We now know which registers we're going to clobber and have to choose which values we'll
        // move and spill. As a simple heuristic we try to first move those registers whose values
        // will be used most often in the remainder of the trace. `clobber_regs` will (after the
        // `sort_by_key` be in ascending order: i.e. the registers whose values we most want to
        // keep will be at the end.
        let mut clobber_regs = clobber_regs.iter_set_bits().collect::<Vec<_>>();
        self.sort_clobber_regs(iidx, &mut clobber_regs);

        'a: for reg in clobber_regs.iter() {
            match &self.gp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => (),
                RegState::Empty => (),
                RegState::FromConst(_, _) => {
                    // We could move these if we really wanted to.
                }
                RegState::FromInst(op_iidxs, _) => {
                    if op_iidxs
                        .iter()
                        .any(|x| self.rev_an.is_inst_var_still_used_after(iidx, *x))
                    {
                        // If a variable has a register hint, and that register is available, it's
                        // a perfect candidate for moving. We could be really clever here, and copy
                        // multiple times if `op_iidx.len() > 1`. For now, we just find the first
                        // variable with a hint that maps to an unused register.
                        for op_iidx in op_iidxs {
                            if let Some(Register::GP(hint_reg)) =
                                self.rev_an.reg_hint(iidx.checked_add(1).unwrap(), *op_iidx)
                                && !asgn_regs.is_set(*reg)
                            {
                                out.push((hint_reg, RegAction::CopyFrom(*reg)));
                                asgn_regs.set(hint_reg);
                                continue 'a;
                            }
                        }

                        // Try and find any empty register that's available.
                        for (empty_reg_i, _) in self
                            .gp_reg_states
                            .iter()
                            .enumerate()
                            .filter(|(_, rs)| matches!(rs, &RegState::Empty))
                        {
                            let empty_reg = GP_REGS[empty_reg_i];
                            if !asgn_regs.is_set(empty_reg) {
                                out.push((empty_reg, RegAction::CopyFrom(*reg)));
                                asgn_regs.set(empty_reg);
                                continue 'a;
                            }
                        }
                        out.push((*reg, RegAction::Spill));
                    }
                }
            }
        }

        out.extend_from_slice(actions);
        out
    }

    /// Align `reg`'s sign/zero extension with `next_ext`. Returns the previous extension state of
    /// the register.
    fn align_extensions(
        &mut self,
        asm: &mut Assembler,
        reg: Rq,
        next_ext: RegExtension,
    ) -> RegExtension {
        let (bitw, mut cur_ext) = match &mut self.gp_reg_states[usize::from(reg.code())] {
            RegState::Reserved | RegState::Empty => unreachable!(),
            RegState::FromConst(cidx, ext) => (
                self.m
                    .type_(self.m.const_(*cidx).tyidx(self.m))
                    .bitw()
                    .unwrap(),
                ext,
            ),
            RegState::FromInst(iidxs, ext) => (
                iidxs
                    .iter()
                    .map(|x| self.m.inst(*x).def_bitw(self.m))
                    .min()
                    .unwrap(),
                ext,
            ),
        };
        let old_ext = *cur_ext;
        match (&mut cur_ext, next_ext) {
            (&mut RegExtension::Undefined, RegExtension::Undefined) => (),
            (&mut RegExtension::Undefined, RegExtension::ZeroExtended)
            | (&mut RegExtension::SignExtended, RegExtension::ZeroExtended) => {
                *cur_ext = next_ext;
                self.force_zero_extend_to_reg64(asm, reg, bitw);
            }
            (&mut RegExtension::ZeroExtended, RegExtension::Undefined) => (),
            (&mut RegExtension::ZeroExtended, RegExtension::ZeroExtended) => (),
            (&mut RegExtension::Undefined, RegExtension::SignExtended)
            | (&mut RegExtension::ZeroExtended, RegExtension::SignExtended) => {
                *cur_ext = next_ext;
                self.force_sign_extend_to_reg64(asm, reg, bitw);
            }
            (&mut RegExtension::SignExtended, RegExtension::Undefined) => (),
            (&mut RegExtension::SignExtended, RegExtension::SignExtended) => (),
        }
        old_ext
    }

    /// Sign extend the `from_bits`-sized integer stored in `reg` up to the full size of the 64-bit
    /// register.
    ///
    /// `from_bits` must be between 1 and 64.
    fn force_sign_extend_to_reg64(&self, asm: &mut Assembler, reg: Rq, from_bits: u32) {
        debug_assert!(from_bits > 0 && from_bits <= 64);
        match from_bits {
            1 => dynasm!(asm
                ; and Rq(reg.code()), 1
                ; neg Rq(reg.code())
            ),
            8 => dynasm!(asm; movsx Rq(reg.code()), Rb(reg.code())),
            16 => dynasm!(asm; movsx Rq(reg.code()), Rw(reg.code())),
            32 => dynasm!(asm; movsx Rq(reg.code()), Rd(reg.code())),
            64 => (), // nothing to do.
            x => todo!("{x}"),
        }
    }

    /// Zero extend the `from_bitw`-sized integer stored in `reg` up to the full size of the 64-bit
    /// register.
    ///
    /// `from_bits` must be between 1 and 64.
    pub(super) fn force_zero_extend_to_reg64(&self, asm: &mut Assembler, reg: Rq, from_bitw: u32) {
        debug_assert!(from_bitw > 0 && from_bitw <= 64);
        match from_bitw {
            1..=31 => dynasm!(asm; and Rd(reg.code()), ((1u64 << from_bitw) - 1) as i32),
            32 => {
                // mov into a 32-bit register zero extends the upper 32 bits.
                dynasm!(asm; mov Rd(reg.code()), Rd(reg.code()));
            }
            64 => (), // There are no additional bits to zero extend
            x => todo!("{x}"),
        }
    }

    /// Return a GP register containing the value for `op` or `None` if that value is not in any
    /// register.
    pub(super) fn find_op_in_gp_reg(&self, op: &Operand) -> Option<Rq> {
        self.find_op_in_gp_regs(op).nth(0)
    }

    /// Return all the GP registers containing the value for `op` or `None` if that value is not in
    /// any register.
    fn find_op_in_gp_regs<'a>(&'a self, op: &'a Operand) -> impl Iterator<Item = Rq> + 'a {
        self.gp_reg_states
            .iter()
            .enumerate()
            .filter(move |(_, x)| match (op, x) {
                (Operand::Const(op_cidx), RegState::FromConst(reg_cidx, _)) => op_cidx == reg_cidx,
                (Operand::Var(op_iidx), RegState::FromInst(reg_iidxs, _)) => {
                    reg_iidxs.contains(op_iidx)
                }
                _ => false,
            })
            .map(|(i, _)| GP_REGS[i])
    }

    /// Return a GP register that is (1) not in `avoid` (2) contains the value for `op`. Returns
    /// `None` if no register meets these two rules.
    fn find_op_in_gp_reg_avoiding(&self, op: &Operand, avoid: RegSet<Rq>) -> Option<Rq> {
        self.gp_reg_states
            .iter()
            .enumerate()
            .filter(|(reg_i, _)| !avoid.is_set(GP_REGS[*reg_i]))
            .find(|(_, x)| match (op, x) {
                (Operand::Const(op_cidx), RegState::FromConst(reg_cidx, _)) => {
                    *op_cidx == *reg_cidx
                }
                (Operand::Var(op_iidx), RegState::FromInst(reg_iidxs, _)) => {
                    reg_iidxs.contains(op_iidx)
                }
                _ => false,
            })
            .map(|(i, _)| GP_REGS[i])
    }

    /// Is the value produced by `op` already in register `reg`?
    fn is_input_in_gp_reg(&self, op: &Operand, reg: Rq) -> bool {
        match &self.gp_reg_states[usize::from(reg.code())] {
            RegState::Empty => false,
            RegState::FromConst(reg_cidx, _) => match op {
                Operand::Const(op_cidx) => *reg_cidx == *op_cidx,
                Operand::Var(_) => false,
            },
            RegState::FromInst(reg_iidxs, _) => match op {
                Operand::Const(_) => false,
                Operand::Var(op_iidx) => reg_iidxs.contains(op_iidx),
            },
            RegState::Reserved => unreachable!(),
        }
    }

    /// Place the value for `op` into `reg` and force its extension appropriately. If necessary,
    /// `op` will be loaded into `reg`.
    fn put_input_in_gp_reg(
        &mut self,
        asm: &mut Assembler,
        op: &Operand,
        reg: Rq,
        ext: RegExtension,
    ) {
        let st = match op {
            Operand::Const(cidx) => {
                self.load_const_into_gp_reg(asm, *cidx, reg);
                self.align_extensions(asm, reg, ext);
                RegState::FromConst(*cidx, ext)
            }
            Operand::Var(op_iidx) => {
                self.force_gp_unspill(asm, *op_iidx, reg);
                self.align_extensions(asm, reg, ext);
                RegState::FromInst(vec![*op_iidx], ext)
            }
        };
        self.gp_regset.set(reg);
        self.gp_reg_states[usize::from(reg.code())] = st;
    }

    /// Copy the value in `old_reg` to `new_reg` leaving `old_reg`'s [RegState] unchanged.
    fn copy_gp_reg(&mut self, asm: &mut Assembler, old_reg: Rq, new_reg: Rq) {
        assert_ne!(old_reg, new_reg);
        dynasm!(asm; mov Rq(new_reg.code()), Rq(old_reg.code()));
        self.gp_regset.set(new_reg);
        self.gp_reg_states[usize::from(new_reg.code())] =
            self.gp_reg_states[usize::from(old_reg.code())].clone();
    }

    /// Move the value in `old_reg` to `new_reg`, setting `old_reg` to [RegState::Empty].
    fn move_gp_reg(&mut self, asm: &mut Assembler, old_reg: Rq, new_reg: Rq) {
        assert_ne!(old_reg, new_reg);
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
        assert!(self.gp_regset.is_set(old_reg) && self.gp_regset.is_set(new_reg));
        dynasm!(asm; xchg Rq(new_reg.code()), Rq(old_reg.code()));
        self.gp_reg_states
            .swap(usize::from(old_reg.code()), usize::from(new_reg.code()));
    }

    /// Spill the value stored in `reg` if it is both (1) used in the future and (2) not already
    /// spilled. This updates `self.spills` (if necessary) but not `self.gp_reg_state` or
    /// `self.gp_regset`.
    ///
    /// Note: this function can change the CPU flags.
    fn spill_gp_if_not_already(&mut self, asm: &mut Assembler, cur_iidx: InstIdx, reg: Rq) {
        match &self.gp_reg_states[usize::from(reg.code())] {
            RegState::Reserved => unreachable!(),
            RegState::Empty => (),
            RegState::FromConst(_, _) => (),
            RegState::FromInst(query_iidxs, _) => {
                if query_iidxs.iter().any(|x| {
                    self.rev_an.is_inst_var_still_used_after(cur_iidx, *x)
                        && self.spills[usize::from(*x)] == SpillState::Empty
                }) {
                    self.force_spill_gp(asm, true, reg);
                }
            }
        }
    }

    /// Spill the value(s) stored in `reg` if it is not already spilled. This updates `self.spills`
    /// (if necessary) but not `self.gp_reg_state` or `self.gp_regset`.
    ///
    /// If `set_cpu_flags` is set to `true`, this function can change the CPU Flags: doing so
    /// allows it to generate more efficient code.
    fn force_spill_gp(&mut self, asm: &mut Assembler, set_cpu_flags: bool, reg: Rq) {
        match &self.gp_reg_states[usize::from(reg.code())] {
            RegState::Reserved => unreachable!(),
            RegState::Empty => (),
            RegState::FromConst(_, _) => (),
            RegState::FromInst(query_iidxs, ext) => {
                // First work out the non-strict subset of `query_iidx`s which are not yet spilled.
                let need_spilling = query_iidxs
                    .iter()
                    .filter(|x| self.spills[usize::from(**x)] == SpillState::Empty)
                    .collect::<Vec<_>>();
                if !need_spilling.is_empty() {
                    // When multiple variables exist in a single register, we only need to spill
                    // the maximum-sized variable: the other values are all contained within it.
                    // Conveniently, because the stack grows downwards, the offset is correct for
                    // each variable too.
                    let bitw = need_spilling
                        .iter()
                        .map(|x| self.m.inst(**x).def_bitw(self.m))
                        .max()
                        .unwrap();
                    let bytew = need_spilling
                        .iter()
                        .map(|x| self.m.inst(**x).def_byte_size(self.m))
                        .max()
                        .unwrap();
                    self.stack.align(bytew);
                    let frame_off = self.stack.grow(bytew);
                    let off = i32::try_from(frame_off).unwrap();
                    match bitw {
                        1 => {
                            if *ext == RegExtension::ZeroExtended {
                                dynasm!(asm; mov BYTE [rbp - off], Rb(reg.code()));
                            } else if set_cpu_flags {
                                dynasm!(asm
                                    ; bt Rd(reg.code()), 0
                                    ; setc BYTE [rbp - off]
                                );
                            } else if *ext == RegExtension::Undefined {
                                dynasm!(asm
                                    ; and Rd(reg.code()), 0x01
                                    ; mov BYTE [rbp - off], Rb(reg.code())
                                );
                            } else {
                                todo!();
                            }
                        }
                        8 => dynasm!(asm; mov BYTE [rbp - off], Rb(reg.code())),
                        16 => dynasm!(asm; mov WORD [rbp - off], Rw(reg.code())),
                        32 => dynasm!(asm; mov DWORD [rbp - off], Rd(reg.code())),
                        64 => dynasm!(asm; mov QWORD [rbp - off], Rq(reg.code())),
                        _ => unreachable!(),
                    }

                    for iidx in need_spilling {
                        self.spills[usize::from(*iidx)] = SpillState::Stack(off);
                    }
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

        assert!(!matches!(inst, Inst::Const(_)));

        match self.spills[usize::from(iidx)] {
            SpillState::Empty => unreachable!("{iidx}"),
            SpillState::Stack(off) => match size {
                1 => dynasm!(asm ; movzx Rq(reg.code()), BYTE [rbp - off]),
                2 => dynasm!(asm ; movzx Rq(reg.code()), WORD [rbp - off]),
                4 => dynasm!(asm ; mov Rd(reg.code()), [rbp - off]),
                8 => dynasm!(asm ; mov Rq(reg.code()), [rbp - off]),
                _ => todo!("{}", size),
            },
            SpillState::Direct(off) => match size {
                8 => dynasm!(asm
                    ; lea Rq(reg.code()), [rbp + off]
                ),
                x => todo!("{x}"),
            },
            SpillState::ConstInt { bits, v } => match bits {
                64 => {
                    dynasm!(asm; mov Rq(reg.code()), QWORD v as i64)
                }
                32 | 16 | 8 | 1 => {
                    dynasm!(asm; mov Rd(reg.code()), v as i32)
                }
                _ => todo!("{bits}"),
            },
            SpillState::ConstPtr(v) => {
                // unwrap cannot fail since pointers are sized.
                let bitw = self.m.type_(self.m.ptr_tyidx()).bitw().unwrap();
                assert_eq!(bitw, 64);
                dynasm!(asm; mov Rq(reg.code()), QWORD v as i64);
            }
        }
        self.gp_regset.set(reg);
        self.gp_reg_states[usize::from(reg.code())] =
            RegState::FromInst(vec![iidx], RegExtension::ZeroExtended);
    }

    /// Load the constant from `cidx` into `reg`.
    ///
    /// If the register is larger than the constant, the unused high-order bits are undefined.
    fn load_const_into_gp_reg(&mut self, asm: &mut Assembler, cidx: ConstIdx, reg: Rq) {
        match self.m.const_(cidx) {
            Const::Float(_tyidx, _x) => todo!(),
            Const::Int(_, x) => match x.bitw() {
                1..=32 => {
                    dynasm!(asm; mov Rd(reg.code()), x.to_zero_ext_u32().unwrap() as i32);
                }
                64 => dynasm!(asm; mov Rq(reg.code()), QWORD x.to_zero_ext_u64().unwrap() as i64),
                x => todo!("{x}"),
            },
            Const::Ptr(x) => {
                dynasm!(asm; mov Rq(reg.code()), QWORD *x as i64)
            }
        }
        self.gp_regset.set(reg);
        self.gp_reg_states[usize::from(reg.code())] =
            RegState::FromConst(cidx, RegExtension::ZeroExtended);
    }

    /// Return the location of the value at `iidx`. If that instruction's value is available in a
    /// register and is spilled to the stack, the former will always be preferred.
    ///
    /// Note that it is undefined behaviour to ask for the location of an instruction which has not
    /// yet produced a value.
    pub(crate) fn var_location(&self, iidx: InstIdx) -> VarLocation {
        if let Some(reg_i) = &self.gp_reg_states.iter().position(|x| {
            if let RegState::FromInst(y, _) = x {
                y.contains(&iidx)
            } else {
                false
            }
        }) {
            VarLocation::Register(Register::GP(GP_REGS[*reg_i]))
        } else if let Some(reg_i) = &self.fp_reg_states.iter().position(|x| {
            if let RegState::FromInst(y, _) = x {
                y.contains(&iidx)
            } else {
                false
            }
        }) {
            VarLocation::Register(Register::FP(FP_REGS[*reg_i]))
        } else {
            let inst = self.m.inst(iidx);
            let size = inst.def_byte_size(self.m);
            match inst {
                Inst::Copy(_) => panic!(),
                Inst::Const(cidx) => match self.m.const_(cidx) {
                    Const::Float(_, v) => VarLocation::ConstFloat(*v),
                    Const::Int(_, x) => VarLocation::ConstInt {
                        bits: x.bitw(),
                        v: x.to_zero_ext_u64().unwrap(),
                    },
                    Const::Ptr(p) => VarLocation::ConstInt {
                        bits: 64,
                        v: u64::try_from(*p).unwrap(),
                    },
                },
                _ => match self.spills[usize::from(iidx)] {
                    SpillState::Empty => panic!("{iidx}"),
                    SpillState::Stack(off) => VarLocation::Stack {
                        frame_off: u32::try_from(off).unwrap(),
                        size,
                    },
                    SpillState::Direct(off) => VarLocation::Direct {
                        frame_off: off,
                        size,
                    },
                    SpillState::ConstInt { bits, v } => VarLocation::ConstInt { bits, v },
                    SpillState::ConstPtr(v) => VarLocation::ConstPtr(v),
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
                    if let Some(Register::FP(reg)) = self.rev_an.reg_hint(iidx, iidx)
                        && !avoid.is_set(reg)
                    {
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
                _ => (),
            }
        }

        // If we already have the value in a register, don't assign a new register.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::Input(op) | RegConstraint::InputOutput(op) => match op {
                    Operand::Var(op_iidx) => {
                        if let Some(reg_i) = &self.fp_reg_states.iter().position(|x| {
                            if let RegState::FromInst(y, _) = x {
                                y.contains(op_iidx)
                            } else {
                                false
                            }
                        }) {
                            let reg = FP_REGS[*reg_i];
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
                    // We need to find a register to spill.
                    todo!();
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
                RegConstraint::Input(op)
                | RegConstraint::InputIntoReg(op, _)
                | RegConstraint::InputOutput(op)
                | RegConstraint::InputOutputIntoReg(op, _)
                | RegConstraint::InputIntoRegAndClobber(op, _) => {
                    if let Some(old_reg) = self.find_op_in_fp_reg(op)
                        && old_reg != new_reg
                    {
                        match self.fp_reg_states[usize::from(new_reg.code())] {
                            RegState::Reserved => unreachable!(),
                            RegState::Empty => {
                                self.move_fp_reg(asm, old_reg, new_reg);
                            }
                            RegState::FromConst(_, _) => todo!(),
                            RegState::FromInst(_, _) => todo!(),
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
                    self.fp_reg_states[usize::from(reg.code())] =
                        RegState::FromInst(vec![iidx], RegExtension::Undefined);
                }
                RegConstraint::Output | RegConstraint::OutputFromReg(_) => {
                    self.move_or_spill_fp(asm, iidx, &mut avoid, reg);
                    self.fp_regset.set(reg);
                    self.fp_reg_states[usize::from(reg.code())] =
                        RegState::FromInst(vec![iidx], RegExtension::Undefined);
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
                (Operand::Const(op_cidx), RegState::FromConst(reg_cidx, _)) => {
                    *op_cidx == *reg_cidx
                }
                (Operand::Var(op_iidx), RegState::FromInst(reg_iidxs, _)) => {
                    reg_iidxs.contains(op_iidx)
                }
                _ => false,
            })
            .map(|(i, _)| FP_REGS[i])
    }

    /// Is the value produced by `op` already in register `reg`?
    fn is_input_in_fp_reg(&self, op: &Operand, reg: Rx) -> bool {
        match &self.fp_reg_states[usize::from(reg.code())] {
            RegState::Empty => false,
            RegState::FromConst(reg_cidx, _) => match op {
                Operand::Const(op_cidx) => *reg_cidx == *op_cidx,
                Operand::Var(_) => false,
            },
            RegState::FromInst(reg_iidxs, _) => match op {
                Operand::Const(_) => false,
                Operand::Var(op_iidx) => reg_iidxs.contains(op_iidx),
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
                RegState::FromConst(*cidx, RegExtension::Undefined)
            }
            Operand::Var(iidx) => {
                self.force_fp_unspill(asm, *iidx, reg);
                RegState::FromInst(vec![*iidx], RegExtension::Undefined)
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
        match &self.fp_reg_states[usize::from(old_reg.code())] {
            RegState::Empty => (),
            RegState::FromConst(_, _) => (),
            RegState::FromInst(reg_iidxs, _) => {
                assert_eq!(reg_iidxs.len(), 1);
                let query_iidx = reg_iidxs[0];
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
        match &self.fp_reg_states[usize::from(reg.code())] {
            RegState::Reserved | RegState::Empty | RegState::FromConst(_, _) => (),
            RegState::FromInst(iidxs, _) => {
                assert_eq!(iidxs.len(), 1);
                let iidx = iidxs[0];
                if self.spills[usize::from(iidx)] == SpillState::Empty {
                    let inst = self.m.inst(iidx);
                    let bitw = inst.def_bitw(self.m);
                    let bytew = inst.def_byte_size(self.m);
                    debug_assert!(usize::try_from(bitw).unwrap() >= bytew);
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
                        if let RegState::FromInst(y, _) = x {
                            y.contains(&iidx)
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
            SpillState::ConstInt { .. } | SpillState::ConstPtr(_) => {
                panic!(); // would indicate some kind of type confusion.
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
    /// A no-op register constraint. No registers will be assigned for this, but we have to put a
    /// value in here for normal Rust reasons. A random register will thus end up being returned
    /// from this, but using that register for any purposes leads to undefined behaviour.
    None,
}

/// A GP register constraint. Each constraint leads to a single register being returned. Note: in
/// some situations (see the individual constraints), multiple constraints might return the same
/// register.
#[derive(Clone, Debug)]
pub(crate) enum GPConstraint {
    /// Make sure that `op` is loaded into a register with its upper bits matching extension
    /// `in_ext`. If `force_reg` is `Some`, that register is guaranteed to be used. If `clobber` is
    /// true, then the value in the register will be treated as clobbered on exit.
    Input {
        op: Operand,
        in_ext: RegExtension,
        force_reg: Option<Rq>,
        clobber_reg: bool,
    },
    /// Make sure that `op` is loaded into a register with its upper bits matching extension
    /// `in_ext`; the result of the instruction will be in the same register with its upper bits
    /// matching extension `out_ext`. If `force_reg` is `Some`, that register is guaranteed to be
    /// used.
    InputOutput {
        op: Operand,
        in_ext: RegExtension,
        out_ext: RegExtension,
        force_reg: Option<Rq>,
    },
    /// The result of the instruction will be in a register with its upper bits matching extension
    /// `out_ext`. If `force_reg` is `Some`, that register is guaranteed to be used. If
    /// `can_be_same_as_input` is true, then the allocator may optionally return a register that is
    /// also used for an input (in such a case, the input will implicitly be considered clobbered).
    Output {
        out_ext: RegExtension,
        force_reg: Option<Rq>,
        can_be_same_as_input: bool,
    },
    /// Align `op`'s upper bits to `out_ext`: the instruction must not use the resulting register
    /// for any purposes.
    AlignExtension { op: Operand, out_ext: RegExtension },
    /// This instruction clobbers `force_reg`.
    Clobber { force_reg: Rq },
    /// A temporary register that the instruction will clobber.
    Temporary,
    /// A no-op register constraint. No registers will be assigned for this, but we have to put a
    /// value in here for normal Rust reasons. A random register will thus end up being returned
    /// from this, but using that register for any purposes leads to undefined behaviour.
    None,
}

/// Takes a map of "register X should copy the value of register Y" and turn it into a sequence of
/// copies respecting dependencies.
///
/// For example, if we say that "R8 should copy R9 and R7 should copy R8" we must do the second
/// copy first to avoid R7 ending up with the value originally in R9. Furthermore, there can be
/// (direct or indirect) cycles in the input such as "R8 should copy R9 and R9 should copy R8":
/// this function will first copy one of those registers into a temporary register, and then
/// generate two copies, one of which will copy from that temporary register.
///
/// The input format is: one copy per `Some`. For example `[None, Some(R15), Some(RDX)...]` means
/// "RAX doesn't copy anything; RCX copies R15's value; and RDX's value remains unchanged". When a
/// register's value remains unchanged a `RegAction::Keep` is generated.
fn reg_copies_to_actions(copies: [Option<Rq>; 16]) -> Vec<(Rq, RegAction)> {
    let mut actions = Vec::new();
    let mut action_regset = RegSet::with_gp_reserved();

    // Step 1: create a sequence of copies in dependency order, without worrying about whether
    // there are cycles.
    for (to_reg_i, from_reg) in copies.into_iter().enumerate() {
        if let Some(from_reg) = from_reg {
            let to_reg = GP_REGS[to_reg_i];
            action_regset.set(to_reg);
            if from_reg == to_reg {
                // The operand is already in the correct register.
                actions.push((to_reg, RegAction::Keep));
                continue;
            }
            match actions
                .iter()
                .position(|(x, action)| *x == from_reg && matches!(action, RegAction::CopyFrom(_)))
            {
                Some(i) => {
                    actions.insert(i, (to_reg, RegAction::CopyFrom(from_reg)));
                }
                None => {
                    actions.push((to_reg, RegAction::CopyFrom(from_reg)));
                }
            }
        }
    }

    // Step 2: Deal with cycles
    let mut i = 0;
    while i < actions.len() {
        match actions[i] {
            (to_reg, RegAction::CopyFrom(_from_reg)) => {
                let mut tmp_reg = None;
                let mut j = i + 1;
                while j < actions.len() {
                    match actions[j] {
                        (fwd_to_reg, RegAction::CopyFrom(fwd_from_reg)) => {
                            if fwd_from_reg == to_reg {
                                // We could be much cleverer here, reusing existing moves, operands
                                // in other registers, and so on.
                                if tmp_reg.is_none() {
                                    match action_regset.find_empty() {
                                        Some(empty_reg) => {
                                            tmp_reg = Some(empty_reg);
                                            assert_ne!(empty_reg, to_reg, "{copies:?}");
                                            actions.insert(
                                                i,
                                                (empty_reg, RegAction::CopyFrom(to_reg)),
                                            );
                                            action_regset.set(empty_reg);
                                            j += 1;
                                        }
                                        None => todo!(),
                                    }
                                }
                                assert_ne!(fwd_to_reg, tmp_reg.unwrap());
                                actions[j] = (fwd_to_reg, RegAction::CopyFrom(tmp_reg.unwrap()));
                            }
                        }
                        (_, RegAction::Keep) => (),
                        _ => todo!(),
                    }
                    j += 1;
                }
            }
            (_, RegAction::Keep) => (),
            _ => todo!(),
        }
        i += 1;
    }

    actions
}

/// This `enum` serves two related purposes: it tells us what we know about the unused upper bits
/// of a value *and* it serves as a specification of what we want those values to be (in a
/// [GPConstraint]). What counts as "upper bits"?
///
///   * For normal values, we assume they may end up in a 64-bit register: any bits between the
///     `bitw` of the type and 64 bits are "upper bits". For 64 bit values, the extension is
///     ignored, and can be set to any value.
///
///   * For floating point values, we assume that 32 bit floats and 64 bit doubles are not
///     intermixed. The extension is thus ignored.
///
///   * We do not currently support "non-normal / non-float" values (e.g. vector values) and will
///     have to think about those at a later point.
///
/// For example, if a 16 bit value is stored in a 64 bit value, we may know for sure that the upper
/// 48 bits are set to zero, or they sign extend the 16 bit value --- or we may have no idea!
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum RegExtension {
    /// We do not know what the upper bits are set to / we do not care what the upper bits are set
    /// to.
    Undefined,
    /// The upper bits zero extend the value / we want the upper bits to zero extend the value.
    ZeroExtended,
    /// The upper bits sign extend the value / we want the upper bits to sign extend the value.
    SignExtended,
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

/// The information the register allocator records at the point of a guard's code generation that
/// it later needs to get a failing guard ready for deopt.
#[derive(Debug)]
pub(super) struct GuardSnapshot {
    gp_regset: RegSet<Rq>,
    gp_reg_states: [RegState; GP_REGS_LEN],
    fp_regset: RegSet<Rx>,
    fp_reg_states: [RegState; FP_REGS_LEN],
    spills: Vec<SpillState>,
    stack: AbstractStack,
}

#[derive(Clone, Debug, PartialEq)]
enum RegState {
    Reserved,
    Empty,
    FromConst(ConstIdx, RegExtension),
    FromInst(Vec<InstIdx>, RegExtension),
}

/// What action should be performed to a register to get it into the right state for an
/// instruction's inputs?
#[derive(Clone, Debug, PartialEq)]
enum RegAction {
    /// Keep the register's value unchanged.
    Keep,
    /// Copy the value from another register.
    CopyFrom(Rq),
    /// Spill this register.
    Spill,
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

    fn union(&mut self, other: Self) {
        self.0 |= other.0;
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

    fn iter_set_bits(&self) -> impl Iterator<Item = Rq> + '_ {
        (0usize..16)
            .filter(|x| self.is_set(GP_REGS[*x]))
            .map(|x| GP_REGS[x])
    }

    fn iter_unset_bits(&self) -> impl Iterator<Item = Rq> + '_ {
        (0usize..16)
            .filter(|x| !self.is_set(GP_REGS[*x]))
            .map(|x| GP_REGS[x])
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
    ConstInt {
        bits: u32,
        v: u64,
    },
    // This variable is a constant pointer.
    ConstPtr(usize),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::jitc_yk::jit_ir::BinOp;
    use rand::Rng;
    use std::collections::HashMap;

    #[test]
    fn regset() {
        let mut x = RegSet::blank();
        assert!(!x.is_set(Rq::R15));
        assert_eq!(x.find_empty(), Some(Rq::R15));
        x.set(Rq::R15);
        assert!(x.is_set(Rq::R15));
        assert_eq!(x.find_empty(), Some(Rq::R14));
        x.set(Rq::R14);
        assert_eq!(x.find_empty(), Some(Rq::R13));
        x.unset(Rq::R14);
        assert_eq!(x.find_empty(), Some(Rq::R14));
        for reg in GP_REGS {
            x.set(reg);
            assert!(x.is_set(reg));
        }
        assert_eq!(x.find_empty(), None);
        x.unset(Rq::RAX);
        assert!(!x.is_set(Rq::RAX));
        assert_eq!(x.find_empty(), Some(Rq::RAX));

        let mut x = RegSet::blank();
        x.set(Rq::R15);
        let mut y = RegSet::blank();
        y.set(Rq::R14);
        x.union(y);
        assert_eq!(
            x.iter_set_bits().collect::<Vec<_>>(),
            vec![Rq::R14, Rq::R15]
        );

        let x = RegSet::<Rq>::blank();
        assert_eq!(x.find_empty_avoiding(RegSet::from(Rq::R15)), Some(Rq::R14));
    }

    fn check_reg_states(
        reg_states: &[[RegState; 16]],
        iidx: InstIdx,
        check: HashMap<&str, RegState>,
    ) {
        for (reg_name, expected) in check.iter() {
            // Note: Intel (not DWARF!) ordering of registers
            let reg_i = [
                "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
                "r12", "r13", "r14", "r15", "rip",
            ]
            .iter()
            .position(|x| x == reg_name)
            .unwrap();
            let actual = &reg_states[usize::from(iidx)][reg_i];
            assert_eq!(
                actual,
                expected,
                "at InstIdx({iidx}), {reg_name} (reg offset {reg_i}) is {actual:?}, not {expected:?}, in {:?}",
                reg_states[usize::from(iidx)]
            );
        }
    }

    #[test]
    fn can_be_same_as_input() {
        let m = Module::from_str(
            "
          entry:
            %0: i64 = param reg
            %1: i64 = param reg
            %2: i64 = param reg
            %3: i64 = param reg
            %4: i64 = param reg
            %5: i64 = param reg
            %6: i64 = param reg
            %7: i64 = param reg
            %8: i64 = param reg
            %9: i64 = param reg
            %10: i64 = param reg
            %11: i64 = add %6, %7
            black_box %11
        ",
        );

        let mut ra = LSRegAlloc::new(&m, 0);
        let mut asm = dynasmrt::x64::Assembler::new().unwrap();
        let mut gp_regsets = Vec::with_capacity(m.insts_len());
        let mut reg_states = Vec::with_capacity(m.insts_len());
        for (iidx, inst) in m.iter_skipping_insts() {
            match inst {
                Inst::BlackBox(_) => (),
                Inst::Param(pinst) => {
                    match VarLocation::from_yksmp_location(&m, iidx, m.param(pinst.paramidx())) {
                        VarLocation::Register(Register::GP(reg)) => {
                            ra.force_assign_inst_gp_reg(&mut asm, iidx, reg);
                        }
                        _ => todo!(),
                    }
                }
                Inst::BinOp(binst) => match binst.binop() {
                    BinOp::Add => {
                        let [_, _, _] = ra.assign_gp_regs(
                            &mut asm,
                            iidx,
                            [
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Output {
                                    out_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    can_be_same_as_input: true,
                                },
                            ],
                        );
                    }
                    _ => panic!(),
                },
                _ => panic!(),
            }
            gp_regsets.push(ra.gp_regset);
            reg_states.push(ra.gp_reg_states.clone());
        }

        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(10),
            HashMap::from([
                (
                    "r8",
                    RegState::FromInst(vec![InstIdx::unchecked_from(6)], RegExtension::Undefined),
                ),
                (
                    "r9",
                    RegState::FromInst(vec![InstIdx::unchecked_from(7)], RegExtension::Undefined),
                ),
            ]),
        );
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(11),
            HashMap::from([
                (
                    "r8",
                    RegState::FromInst(vec![InstIdx::unchecked_from(11)], RegExtension::Undefined),
                ),
                (
                    "r9",
                    RegState::FromInst(vec![InstIdx::unchecked_from(7)], RegExtension::Undefined),
                ),
            ]),
        );
    }

    #[test]
    fn spilling() {
        let m = Module::from_str(
            "
          entry:
            %0: i8 = param reg
            %1: i8 = add %0, %0
            %2: i8 = add %0, %1
            %3: i8 = add %0, %2
            %4: i8 = add %0, %3
            %5: i8 = add %0, %4
            %6: i8 = add %0, %5
            %7: i8 = add %0, %6
            %8: i8 = add %0, %7
            %9: i8 = add %0, %8
            %10: i8 = add %0, %9
            %11: i8 = add %0, %10
            %12: i8 = add %0, %11
            %13: i8 = add %0, %12
            %14: i64 = zext %1
            %15: i8 = add %0, %13
            %16: i8 = add %0, %13
            black_box %1
            black_box %2
            black_box %3
            black_box %4
            black_box %5
            black_box %6
            black_box %7
            black_box %8
            black_box %9
            black_box %10
            black_box %11
            black_box %12
            black_box %13
            black_box %14
            black_box %15
            black_box %16
        ",
        );

        let mut ra = LSRegAlloc::new(&m, 0);
        let mut asm = dynasmrt::x64::Assembler::new().unwrap();
        let mut gp_regsets = Vec::with_capacity(m.insts_len());
        let mut reg_states = Vec::with_capacity(m.insts_len());
        let mut spill_states = Vec::with_capacity(m.insts_len());
        for (iidx, inst) in m.iter_skipping_insts() {
            match inst {
                Inst::BlackBox(_) => (),
                Inst::Param(pinst) => {
                    match VarLocation::from_yksmp_location(&m, iidx, m.param(pinst.paramidx())) {
                        VarLocation::Register(Register::GP(reg)) => {
                            ra.force_assign_inst_gp_reg(&mut asm, iidx, reg);
                        }
                        _ => todo!(),
                    }
                }
                Inst::BinOp(binst) => match binst.binop() {
                    BinOp::Add => {
                        let [_, _, _] = ra.assign_gp_regs(
                            &mut asm,
                            iidx,
                            [
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Output {
                                    out_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    can_be_same_as_input: true,
                                },
                            ],
                        );
                    }
                    _ => panic!(),
                },
                Inst::ZExt(zinst) => {
                    let [_reg] = ra.assign_gp_regs(
                        &mut asm,
                        iidx,
                        [GPConstraint::AlignExtension {
                            op: zinst.val(&m),
                            out_ext: RegExtension::ZeroExtended,
                        }],
                    );
                }
                _ => panic!(),
            }
            gp_regsets.push(ra.gp_regset);
            reg_states.push(ra.gp_reg_states.clone());
            spill_states.push(ra.spills.clone());
        }

        #[allow(clippy::needless_range_loop)]
        for i in 1..14 {
            assert!(
                spill_states[i]
                    .iter()
                    .all(|x| matches!(x, SpillState::Empty))
            );
            check_reg_states(
                &reg_states,
                InstIdx::unchecked_from(i),
                HashMap::from([
                    (
                        "rax",
                        RegState::FromInst(
                            vec![InstIdx::unchecked_from(0)],
                            RegExtension::Undefined,
                        ),
                    ),
                    (
                        "r15",
                        RegState::FromInst(
                            vec![InstIdx::unchecked_from(1)],
                            RegExtension::Undefined,
                        ),
                    ),
                ]),
            );
        }

        assert!(
            spill_states[14]
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != 1)
                .all(|(_, x)| matches!(*x, SpillState::Empty))
        );
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(14),
            HashMap::from([
                (
                    "rax",
                    RegState::FromInst(vec![InstIdx::unchecked_from(0)], RegExtension::Undefined),
                ),
                (
                    "r15",
                    RegState::FromInst(
                        vec![InstIdx::unchecked_from(1), InstIdx::unchecked_from(14)],
                        RegExtension::ZeroExtended,
                    ),
                ),
            ]),
        );

        assert_matches!(spill_states[15][12], SpillState::Stack(_),);
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(15),
            HashMap::from([
                (
                    "rax",
                    RegState::FromInst(vec![InstIdx::unchecked_from(0)], RegExtension::Undefined),
                ),
                (
                    "r15",
                    RegState::FromInst(
                        vec![InstIdx::unchecked_from(1), InstIdx::unchecked_from(14)],
                        RegExtension::ZeroExtended,
                    ),
                ),
            ]),
        );
    }

    #[test]
    fn spilling2() {
        let m = Module::from_str(
            "
          func_decl f(i8, i8, i8, i8, i8, i8)
          entry:
            %0: i8 = param reg
            %1: i8 = add %0, %0
            %2: i8 = add %0, %1
            %3: i8 = add %0, %2
            %4: i8 = add %0, %3
            %5: i8 = add %0, %4
            %6: i8 = add %0, %5
            %7: i8 = add %0, %6
            %8: i8 = add %0, %7
            %9: i8 = add %0, %8
            %10: i8 = add %0, %9
            %11: i8 = add %0, %10
            %12: i8 = add %1, %11
            %13: i8 = add %2, %12
            %14: i8 = add %3, %13
            %15: i8 = add %4, %14
            %16: i8 = add %5, %15
            %17: i8 = add %6, %16
            %18: i8 = add %7, %17
            %19: i8 = add %8, %18
            %20: i8 = add %9, %19
            %21: i8 = add %10, %20
            %22: i8 = add %11, %21
            %23: i8 = add %12, %22
            call @f(%2, %4, %6, %8, %10, %12)
            black_box %1
            black_box %2
            black_box %3
            black_box %4
            black_box %5
            black_box %6
            black_box %7
            black_box %8
            black_box %9
            black_box %10
            black_box %11
            black_box %12
            black_box %13
            black_box %14
            black_box %15
            black_box %16
            black_box %17
            black_box %18
            black_box %19
            black_box %20
            black_box %21
            black_box %22
            black_box %23
        ",
        );

        let mut ra = LSRegAlloc::new(&m, 0);
        let mut asm = dynasmrt::x64::Assembler::new().unwrap();
        let mut gp_regsets = Vec::with_capacity(m.insts_len());
        let mut reg_states = Vec::with_capacity(m.insts_len());
        let mut spill_states = Vec::with_capacity(m.insts_len());
        for (iidx, inst) in m.iter_skipping_insts() {
            match inst {
                Inst::BlackBox(_) => (),
                Inst::Param(pinst) => {
                    match VarLocation::from_yksmp_location(&m, iidx, m.param(pinst.paramidx())) {
                        VarLocation::Register(Register::GP(reg)) => {
                            ra.force_assign_inst_gp_reg(&mut asm, iidx, reg);
                        }
                        _ => todo!(),
                    }
                }
                Inst::BinOp(binst) => match binst.binop() {
                    BinOp::Add => {
                        let [_, _, _] = ra.assign_gp_regs(
                            &mut asm,
                            iidx,
                            [
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Input {
                                    op: binst.rhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Output {
                                    out_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    can_be_same_as_input: true,
                                },
                            ],
                        );
                    }
                    _ => panic!(),
                },
                Inst::Call(_) => {
                    // We don't need to fill this out for our particular test.
                }
                Inst::ZExt(zinst) => {
                    let [_reg] = ra.assign_gp_regs(
                        &mut asm,
                        iidx,
                        [GPConstraint::AlignExtension {
                            op: zinst.val(&m),
                            out_ext: RegExtension::ZeroExtended,
                        }],
                    );
                }
                _ => panic!(),
            }
            gp_regsets.push(ra.gp_regset);
            reg_states.push(ra.gp_reg_states.clone());
            spill_states.push(ra.spills.clone());
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..14 {
            assert_eq!(
                spill_states[i]
                    .iter()
                    .filter(|x| matches!(x, SpillState::Stack(_)))
                    .count(),
                0
            )
        }

        assert_eq!(
            spill_states[14]
                .iter()
                .filter(|x| matches!(x, SpillState::Stack(_)))
                .count(),
            1
        );
        assert_eq!(
            spill_states[15]
                .iter()
                .filter(|x| matches!(x, SpillState::Stack(_)))
                .count(),
            2
        );
        assert_eq!(
            spill_states[16]
                .iter()
                .filter(|x| matches!(x, SpillState::Stack(_)))
                .count(),
            3
        );
        assert_eq!(
            spill_states[17]
                .iter()
                .filter(|x| matches!(x, SpillState::Stack(_)))
                .count(),
            4
        );
        assert_eq!(
            spill_states[18]
                .iter()
                .filter(|x| matches!(x, SpillState::Stack(_)))
                .count(),
            5
        );
    }

    #[test]
    fn multiple_instructions_in_one_reg() {
        // This (long!) sequence tests two things: first that we merge zext/sext instructions when
        // possible and that we spill at the correct point (hence all the `add`s).
        let m = Module::from_str(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i64 = param reg
                %3: i32 = zext %0
                %4: i64 = zext %0
                %5: i64 = zext %3
                %6: i64 = sext %3
                %7: i64 = add %6, %5
                %8: i64 = add %7, %5
                %9: i64 = add %8, %5
                %10: i64 = add %9, %5
                %11: i64 = add %10, %5
                %12: i64 = add %11, %5
                %13: i64 = add %12, %5
                %14: i64 = add %13, %5
                %15: i64 = add %14, %5
                %16: i64 = add %15, %5
                %17: i64 = add %16, %5
                %18: i64 = add %17, %5
                %19: i64 = add %18, %5
                %20: i64 = add %19, %5
                black_box %3
                black_box %6
                black_box %7
                black_box %8
                black_box %9
                black_box %10
                black_box %11
                black_box %12
                black_box %13
                black_box %14
                black_box %15
                black_box %16
                black_box %17
                black_box %18
                black_box %19
                black_box %20
        ",
        );

        let mut ra = LSRegAlloc::new(&m, 0);
        let mut asm = dynasmrt::x64::Assembler::new().unwrap();
        let mut gp_regsets = Vec::with_capacity(m.insts_len());
        let mut reg_states = Vec::with_capacity(m.insts_len());
        let mut spill_states = Vec::with_capacity(m.insts_len());
        for (iidx, inst) in m.iter_skipping_insts() {
            match inst {
                Inst::BlackBox(_) => (),
                Inst::Param(pinst) => {
                    match VarLocation::from_yksmp_location(&m, iidx, m.param(pinst.paramidx())) {
                        VarLocation::Register(Register::GP(reg)) => {
                            ra.force_assign_inst_gp_reg(&mut asm, iidx, reg);
                        }
                        _ => todo!(),
                    }
                }
                Inst::BinOp(binst) => match binst.binop() {
                    BinOp::Add => {
                        let [_, _, _] = ra.assign_gp_regs(
                            &mut asm,
                            iidx,
                            [
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Input {
                                    op: binst.lhs(&m),
                                    in_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    clobber_reg: false,
                                },
                                GPConstraint::Output {
                                    out_ext: RegExtension::Undefined,
                                    force_reg: None,
                                    can_be_same_as_input: true,
                                },
                            ],
                        );
                    }
                    _ => panic!(),
                },
                Inst::SExt(zinst) => {
                    let [_] = ra.assign_gp_regs(
                        &mut asm,
                        iidx,
                        [GPConstraint::AlignExtension {
                            op: zinst.val(&m),
                            out_ext: RegExtension::SignExtended,
                        }],
                    );
                }
                Inst::ZExt(zinst) => {
                    let [_] = ra.assign_gp_regs(
                        &mut asm,
                        iidx,
                        [GPConstraint::AlignExtension {
                            op: zinst.val(&m),
                            out_ext: RegExtension::ZeroExtended,
                        }],
                    );
                }
                _ => panic!(),
            }
            gp_regsets.push(ra.gp_regset);
            reg_states.push(ra.gp_reg_states.clone());
            spill_states.push(ra.spills.clone());
        }

        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(2),
            HashMap::from([(
                "rax",
                RegState::FromInst(vec![InstIdx::unchecked_from(0)], RegExtension::Undefined),
            )]),
        );
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(3),
            HashMap::from([(
                "rax",
                RegState::FromInst(
                    vec![InstIdx::unchecked_from(0), InstIdx::unchecked_from(3)],
                    RegExtension::ZeroExtended,
                ),
            )]),
        );
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(4),
            HashMap::from([(
                "rax",
                RegState::FromInst(
                    vec![
                        InstIdx::unchecked_from(0),
                        InstIdx::unchecked_from(3),
                        InstIdx::unchecked_from(4),
                    ],
                    RegExtension::ZeroExtended,
                ),
            )]),
        );
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(5),
            HashMap::from([(
                "rax",
                RegState::FromInst(
                    vec![
                        InstIdx::unchecked_from(0),
                        InstIdx::unchecked_from(3),
                        InstIdx::unchecked_from(4),
                        InstIdx::unchecked_from(5),
                    ],
                    RegExtension::ZeroExtended,
                ),
            )]),
        );
        check_reg_states(
            &reg_states,
            InstIdx::unchecked_from(6),
            HashMap::from([(
                "rax",
                RegState::FromInst(
                    vec![
                        InstIdx::unchecked_from(0),
                        InstIdx::unchecked_from(3),
                        InstIdx::unchecked_from(4),
                        InstIdx::unchecked_from(5),
                        InstIdx::unchecked_from(6),
                    ],
                    RegExtension::SignExtended,
                ),
            )]),
        );

        #[allow(clippy::needless_range_loop)]
        for i in 7..19 {
            assert!(
                spill_states[i]
                    .iter()
                    .all(|x| matches!(x, SpillState::Empty))
            );
        }

        assert_matches!(spill_states[20][3], SpillState::Stack(_));
        assert_matches!(spill_states[20][6], SpillState::Stack(_));
    }

    #[test]
    fn multiple_operands_to_one_reg() {
        let m = Module::from_str(
            "
          entry:
            %0: i64 = param reg
            %1: i64 = add %0, %0
            black_box %1
        ",
        );

        let mut ra = LSRegAlloc::new(&m, 0);
        let mut asm = dynasmrt::x64::Assembler::new().unwrap();

        let iidx0 = InstIdx::unchecked_from(0);
        let Inst::Param(pinst) = m.inst(iidx0) else {
            panic!()
        };
        match VarLocation::from_yksmp_location(&m, iidx0, m.param(pinst.paramidx())) {
            VarLocation::Register(Register::GP(reg)) => {
                ra.force_assign_inst_gp_reg(&mut asm, iidx0, reg);
            }
            _ => todo!(),
        }

        let iidx1 = InstIdx::unchecked_from(1);
        let Inst::BinOp(binst) = m.inst(iidx1) else {
            panic!()
        };
        let cnstrs = [
            GPConstraint::Input {
                op: binst.lhs(&m),
                in_ext: RegExtension::Undefined,
                force_reg: None,
                clobber_reg: false,
            },
            GPConstraint::Input {
                op: binst.lhs(&m),
                in_ext: RegExtension::Undefined,
                force_reg: None,
                clobber_reg: false,
            },
            GPConstraint::Output {
                out_ext: RegExtension::Undefined,
                force_reg: None,
                can_be_same_as_input: true,
            },
        ];
        let (_asgn_regs, cnstr_regs) = ra.find_regs_for_constraints(iidx0, &cnstrs);
        assert_eq!(cnstr_regs[0], cnstr_regs[1]);
        assert_ne!(cnstr_regs[1], cnstr_regs[2]);

        let copies = ra.input_regs_to_copies(iidx1, &cnstrs, &cnstr_regs);
        assert_eq!(copies.iter().filter(|x| x.is_some()).count(), 1);
        let actions = reg_copies_to_actions(copies);
        let actions = ra.move_or_spill_clobbered_regs(iidx1, &actions, &cnstrs, &cnstr_regs);
        assert_matches!(actions.as_slice(), &[(_, RegAction::Keep)]);
    }

    /// A convenience function mapping pairs of copies to a full array suitable for passing to
    /// [reg_copies_to_actions].
    fn expand_copies(in_moves: &[(Rq, Rq)]) -> [Option<Rq>; 16] {
        let mut out_moves = [None; 16];
        for (to_reg, from_reg) in in_moves {
            assert!(
                out_moves[usize::from(to_reg.code())].is_none(),
                "{to_reg:?}"
            );
            out_moves[usize::from(to_reg.code())] = Some(*from_reg);
        }
        out_moves
    }

    #[test]
    fn reg_copies() {
        assert_eq!(reg_copies_to_actions(expand_copies(&[])), []);
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[(Rq::RAX, Rq::RCX)])),
            [(Rq::RAX, RegAction::CopyFrom(Rq::RCX)),]
        );
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[(Rq::RAX, Rq::RCX), (Rq::R8, Rq::R9)])),
            [
                (Rq::RAX, RegAction::CopyFrom(Rq::RCX)),
                (Rq::R8, RegAction::CopyFrom(Rq::R9)),
            ]
        );
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[
                (Rq::RDX, Rq::RBX),
                (Rq::RCX, Rq::RDX),
                (Rq::RAX, Rq::RCX)
            ])),
            [
                (Rq::RAX, RegAction::CopyFrom(Rq::RCX)),
                (Rq::RCX, RegAction::CopyFrom(Rq::RDX)),
                (Rq::RDX, RegAction::CopyFrom(Rq::RBX))
            ]
        );
        // Duplicates
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[
                (Rq::RAX, Rq::RBX),
                (Rq::RCX, Rq::RBX),
                (Rq::RDX, Rq::RBX)
            ])),
            [
                (Rq::RAX, RegAction::CopyFrom(Rq::RBX)),
                (Rq::RCX, RegAction::CopyFrom(Rq::RBX)),
                (Rq::RDX, RegAction::CopyFrom(Rq::RBX))
            ]
        );
        // Keeps
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[
                (Rq::RAX, Rq::RAX),
                (Rq::RCX, Rq::RCX),
                (Rq::RDX, Rq::RBX)
            ])),
            [
                (Rq::RAX, RegAction::Keep),
                (Rq::RCX, RegAction::Keep),
                (Rq::RDX, RegAction::CopyFrom(Rq::RBX))
            ]
        );
        // Direct cycles
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[(Rq::RAX, Rq::RCX), (Rq::RCX, Rq::RAX)])),
            [
                (Rq::R15, RegAction::CopyFrom(Rq::RCX)),
                (Rq::RCX, RegAction::CopyFrom(Rq::RAX)),
                (Rq::RAX, RegAction::CopyFrom(Rq::R15))
            ]
        );
        // A direct cycle case which checks that we don't accidentally hand out `R15` (which is the
        // first register we hand out if nothing is used!) and create a move from `R15 -> R15`.
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[(Rq::RDI, Rq::R15), (Rq::R15, Rq::RDI)])),
            [
                (Rq::R14, RegAction::CopyFrom(Rq::R15)),
                (Rq::R15, RegAction::CopyFrom(Rq::RDI)),
                (Rq::RDI, RegAction::CopyFrom(Rq::R14))
            ]
        );
        // Direct cycle + duplicates
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[
                (Rq::RAX, Rq::RCX),
                (Rq::RCX, Rq::RAX),
                (Rq::R8, Rq::RAX),
                (Rq::R9, Rq::RCX)
            ])),
            [
                (Rq::R9, RegAction::CopyFrom(Rq::RCX)), // could be optimised away!
                (Rq::R15, RegAction::CopyFrom(Rq::RCX)),
                (Rq::RCX, RegAction::CopyFrom(Rq::RAX)),
                (Rq::R8, RegAction::CopyFrom(Rq::RAX)),
                (Rq::RAX, RegAction::CopyFrom(Rq::R15))
            ]
        );
        // Indirect cycle
        assert_eq!(
            reg_copies_to_actions(expand_copies(&[
                (Rq::RAX, Rq::RBX),
                (Rq::RBX, Rq::RCX),
                (Rq::RCX, Rq::RAX)
            ])),
            [
                (Rq::R15, RegAction::CopyFrom(Rq::RBX)),
                (Rq::RBX, RegAction::CopyFrom(Rq::RCX)),
                (Rq::RCX, RegAction::CopyFrom(Rq::RAX)),
                (Rq::RAX, RegAction::CopyFrom(Rq::R15))
            ]
        );

        // Fuzz `reg_copies_to_actions` with lots of random data.
        let mut rng = rand::rng();
        for _ in 0..10000 {
            let mut moves = Vec::new();
            // Note: if we crank the `10` below any higher, we get into register pressure
            // situations where we hit a `todo` in `reg_copies_to_actions`. Since I haven't yet seen
            // that case in real life, I'm not hugely inclined to spend a lot of time implementing
            // it yet. If/when we do, the `10` should become a `16`.
            for _ in 0..rng.random_range(1..10) {
                let from_reg = loop {
                    let from_reg = GP_REGS[rng.random_range(0..16)];
                    if !moves.iter().any(|(x, _)| *x == from_reg) {
                        break from_reg;
                    }
                };
                let to_reg = loop {
                    let to_reg = GP_REGS[rng.random_range(0..16)];
                    if from_reg != to_reg {
                        break to_reg;
                    }
                };
                moves.push((from_reg, to_reg));
            }
            reg_copies_to_actions(expand_copies(&moves));
        }
    }
}
