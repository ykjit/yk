//! A simple linear scan register allocator.
//!
//! The "main" interface from the code generator to the register allocator is `get_gp_regs` (and/or
//! `get_fp_regs`) and [RegConstraint]. For example:
//!
//! ```rust,ignore
//! match binop {
//!   BinOp::Add => {
//!     let size = lhs.byte_size(self.m);
//!     let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
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
//! This says "return two x64 registers: `lhs_reg` will take a value as input and later contain an
//! output value (clobbering the input value, which will be spilled if necessary); `rhs_reg` will
//! take a value as input (and mustn't clobber it)". Those registers can then be used with dynasmrt
//! as one expects.
//!
//! The allocator keeps track of which registers have which trace instruction's values in and of
//! where it has spilled an instruction's value: it guarantees to spill an instruction to at most
//! one place on the stack.

use crate::compile::jitc_yk::{
    codegen::{
        abs_stack::AbstractStack,
        reg_alloc::{self, VarLocation},
    },
    jit_ir::{Const, ConstIdx, FloatTy, Inst, InstIdx, Module, Operand, Ty},
};
use dynasmrt::{
    dynasm,
    x64::{
        Assembler, {Rq, Rx},
    },
    DynasmApi, Register,
};
use std::marker::PhantomData;

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
    /// Which general purpose registers are active?
    gp_regset: RegSet<Rq>,
    /// In what state are the general purpose registers?
    gp_reg_states: [RegState; GP_REGS_LEN],
    /// Which floating point registers are active?
    fp_regset: RegSet<Rx>,
    /// In what state are the floating point registers?
    fp_reg_states: [RegState; FP_REGS_LEN],
    /// Record the [InstIdx] of the last instruction that the value produced by an instruction is
    /// used. By definition this must either be unused (if an instruction does not produce a value)
    /// or `>=` the offset in this vector.
    inst_vals_alive_until: Vec<InstIdx>,
    /// Where on the stack is an instruction's value spilled? Set to `usize::MAX` if that offset is
    /// currently unknown.
    spills: Vec<SpillState>,
    /// The abstract stack: shared between general purpose and floating point registers.
    stack: AbstractStack,
}

impl<'a> LSRegAlloc<'a> {
    pub(crate) fn new(m: &'a Module) -> Self {
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

        let mut gp_reg_states = [RegState::Empty; GP_REGS_LEN];
        for reg in RESERVED_GP_REGS {
            gp_reg_states[usize::from(reg.code())] = RegState::Reserved;
        }

        let mut fp_reg_states = [RegState::Empty; FP_REGS_LEN];
        for reg in RESERVED_FP_REGS {
            fp_reg_states[usize::from(reg.code())] = RegState::Reserved;
        }

        LSRegAlloc {
            m,
            gp_regset: RegSet::with_gp_reserved(),
            gp_reg_states,
            fp_regset: RegSet::with_fp_reserved(),
            fp_reg_states,
            inst_vals_alive_until: m.inst_vals_alive_until(),
            spills: vec![SpillState::Empty; m.insts_len()],
            stack: Default::default(),
        }
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
                    if !self.is_inst_var_still_used_at(iidx, reg_iidx) {
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
                    if !self.is_inst_var_still_used_at(iidx, reg_iidx) {
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

    // Is the value produced by instruction `query_iidx` used after (but not including!)
    // instruction `cur_idx`?
    fn is_inst_var_still_used_after(&self, cur_iidx: InstIdx, query_iidx: InstIdx) -> bool {
        usize::from(cur_iidx) < usize::from(self.inst_vals_alive_until[usize::from(query_iidx)])
    }

    /// Is the value produced by instruction `query_iidx` used at or after instruction `cur_idx`?
    fn is_inst_var_still_used_at(&self, cur_iidx: InstIdx, query_iidx: InstIdx) -> bool {
        usize::from(cur_iidx) <= usize::from(self.inst_vals_alive_until[usize::from(query_iidx)])
    }
}

/// The parts of the register allocator needed for general purpose registers.
impl<'a> LSRegAlloc<'a> {
    /// Forcibly assign the machine register `reg`, which must be in the [RegState::Empty] state,
    /// to the value produced by instruction `iidx`.
    pub(crate) fn force_assign_inst_gp_reg(&mut self, iidx: InstIdx, reg: Rq) {
        debug_assert!(!self.gp_regset.is_set(reg));
        self.gp_regset.set(reg);
        self.gp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
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
        self.spills[usize::from(iidx)] = SpillState::Indirect(frame_off);
    }

    /// Allocate registers for the instruction at position `iidx`.
    pub(crate) fn get_gp_regs<const N: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        constraints: [RegConstraint<Rq>; N],
    ) -> [Rq; N] {
        self.get_gp_regs_avoiding(asm, iidx, constraints, RegSet::with_gp_reserved())
    }

    /// Allocate registers for the instruction at position `iidx`. Registers which should not be
    /// allocated/touched in any way are specified in `avoid`: note that these registers are not
    /// considered clobbered so will not be spilled.
    pub(crate) fn get_gp_regs_avoiding<const N: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        constraints: [RegConstraint<Rq>; N],
        mut avoid: RegSet<Rq>,
    ) -> [Rq; N] {
        let mut found_output = false; // Check that there aren't multiple output regs
        let mut out = [None; N];

        // Where the caller has told us they want to put things in specific registers, we need to
        // make sure we avoid allocating those in other circumstances.
        for cnstr in &constraints {
            match cnstr {
                RegConstraint::InputIntoReg(_, reg)
                | RegConstraint::InputIntoRegAndClobber(_, reg)
                | RegConstraint::InputOutputIntoReg(_, reg)
                | RegConstraint::OutputFromReg(reg) => {
                    avoid.set(*reg);
                }
                RegConstraint::Input(_)
                | RegConstraint::InputOutput(_)
                | RegConstraint::Output
                | RegConstraint::Temporary => {}
            }
        }

        // If we already have the value in a register, don't allocate a new register.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::Input(op) | RegConstraint::InputOutput(op) => match op {
                    Operand::Local(op_iidx) => {
                        if let Some(reg_i) = self.gp_reg_states.iter().position(|x| {
                            if let RegState::FromInst(y) = x {
                                y == op_iidx
                            } else {
                                false
                            }
                        }) {
                            let reg = GP_REGS[reg_i];
                            if !avoid.is_set(reg) {
                                assert!(self.gp_regset.is_set(reg));
                                avoid.set(reg);
                                out[i] = Some(reg);
                                if let RegConstraint::InputOutput(_) = cnstr {
                                    debug_assert!(!found_output);
                                    found_output = true;
                                    if self.is_inst_var_still_used_after(iidx, *op_iidx) {
                                        self.spill_gp_if_not_already(asm, reg);
                                    }
                                    self.gp_reg_states[usize::from(reg.code())] =
                                        RegState::FromInst(iidx);
                                }
                            }
                        }
                    }
                    Operand::Const(_cidx) => (),
                },
                RegConstraint::InputOutputIntoReg(_op, _reg)
                | RegConstraint::InputIntoReg(_op, _reg)
                | RegConstraint::InputIntoRegAndClobber(_op, _reg) => {
                    // OPT: do the same trick as Input/InputOutput
                }
                RegConstraint::Output
                | RegConstraint::OutputFromReg(_)
                | RegConstraint::Temporary => (),
            }
        }

        for (i, x) in constraints.iter().enumerate() {
            if out[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            match x {
                RegConstraint::Input(op)
                | RegConstraint::InputIntoReg(op, _)
                | RegConstraint::InputIntoRegAndClobber(op, _)
                | RegConstraint::InputOutput(op)
                | RegConstraint::InputOutputIntoReg(op, _) => {
                    let reg = match x {
                        RegConstraint::Input(_) | RegConstraint::InputOutput(_) => {
                            self.get_empty_gp_reg(asm, iidx, avoid)
                        }
                        RegConstraint::InputIntoReg(_, reg)
                        | RegConstraint::InputIntoRegAndClobber(_, reg)
                        | RegConstraint::InputOutputIntoReg(_, reg) => {
                            // OPT: Not everything needs spilling
                            self.spill_gp_if_not_already(asm, *reg);
                            *reg
                        }
                        RegConstraint::Output
                        | RegConstraint::OutputFromReg(_)
                        | RegConstraint::Temporary => {
                            unreachable!()
                        }
                    };

                    // At this point we know the value in `reg` has been spilled if necessary, so
                    // we can overwrite it.
                    match op {
                        Operand::Local(op_iidx) => {
                            self.force_gp_unspill(asm, *op_iidx, reg);
                        }
                        Operand::Const(cidx) => {
                            // FIXME: we could reuse consts in regs
                            self.load_const_into_gp_reg(asm, *cidx, reg);
                        }
                    }

                    self.gp_regset.set(reg);
                    out[i] = Some(reg);
                    avoid.set(reg);
                    let st = match x {
                        RegConstraint::Input(_) | RegConstraint::InputIntoReg(_, _) => match op {
                            Operand::Local(op_iidx) => RegState::FromInst(*op_iidx),
                            Operand::Const(cidx) => RegState::FromConst(*cidx),
                        },
                        RegConstraint::InputIntoRegAndClobber(_, _) => {
                            self.gp_regset.unset(reg);
                            RegState::Empty
                        }
                        RegConstraint::InputOutput(_) | RegConstraint::InputOutputIntoReg(_, _) => {
                            debug_assert!(!found_output);
                            found_output = true;
                            RegState::FromInst(iidx)
                        }
                        RegConstraint::Output
                        | RegConstraint::OutputFromReg(_)
                        | RegConstraint::Temporary => {
                            unreachable!()
                        }
                    };
                    self.gp_reg_states[usize::from(reg.code())] = st;
                }
                RegConstraint::Output => {
                    let reg = self.get_empty_gp_reg(asm, iidx, avoid);
                    self.gp_regset.set(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                    avoid.set(reg);
                    out[i] = Some(reg);
                }
                RegConstraint::OutputFromReg(reg) => {
                    // OPT: Don't have to always spill.
                    self.spill_gp_if_not_already(asm, *reg);
                    self.gp_regset.set(*reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                    avoid.set(*reg);
                    out[i] = Some(*reg);
                }
                RegConstraint::Temporary => {
                    let reg = self.get_empty_gp_reg(asm, iidx, avoid);
                    self.gp_regset.unset(reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                    avoid.set(reg);
                    out[i] = Some(reg);
                }
            }
        }

        out.map(|x| x.unwrap())
    }

    /// If the value stored in `reg` is not already spilled to the heap, then spill it. Note that
    /// this function neither writes to the register or changes the register's [RegState].
    fn spill_gp_if_not_already(&mut self, asm: &mut Assembler, reg: Rq) {
        match self.gp_reg_states[usize::from(reg.code())] {
            RegState::Reserved | RegState::Empty | RegState::FromConst(_) => (),
            RegState::FromInst(iidx) => {
                if self.spills[usize::from(iidx)] == SpillState::Empty {
                    let inst = self.m.inst_no_proxies(iidx);
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
            }
        }
    }

    /// Load the value for `iidx` from the stack into `reg`.
    ///
    /// # Panics
    ///
    /// If `iidx` has not previously been spilled.
    fn force_gp_unspill(&mut self, asm: &mut Assembler, iidx: InstIdx, reg: Rq) {
        let (iidx, inst) = self.m.inst_deproxy(iidx);
        let size = inst.def_byte_size(self.m);

        if let Inst::ProxyConst(cidx) = inst {
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
                    match size {
                        1 => dynasm!(asm ; movzx Rq(reg.code()), Rb(cur_reg.code())),
                        2 => dynasm!(asm ; movzx Rq(reg.code()), Rw(cur_reg.code())),
                        4 => dynasm!(asm ; mov Rd(reg.code()), Rd(cur_reg.code())),
                        8 => dynasm!(asm ; mov Rq(reg.code()), Rq(cur_reg.code())),
                        _ => todo!("{}", size),
                    }
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
                    ; mov Rq(reg.code()), [rbp]
                    ; lea Rq(reg.code()), [Rq(reg.code()) + off]
                ),
                x => todo!("{x}"),
            },
            SpillState::Indirect(off) => {
                match size {
                    8 => {
                        dynasm!(asm
                            ; mov Rq(reg.code()), [rbp]
                            ; mov Rq(reg.code()), [Rq(reg.code()) + off]
                        );
                    }
                    4 => {
                        dynasm!(asm
                            ; mov Rq(reg.code()), [rbp]
                            ; mov Rd(reg.code()), [Rq(reg.code()) + off]
                        );
                    }
                    _ => todo!(),
                }
                self.gp_regset.set(reg);
            }
        }
    }

    /// Load the constant from `cidx` into `reg`.
    fn load_const_into_gp_reg(&mut self, asm: &mut Assembler, cidx: ConstIdx, reg: Rq) {
        match self.m.const_(cidx) {
            Const::Float(_tyidx, _x) => todo!(),
            Const::Int(tyidx, x) => {
                let Ty::Integer(width) = self.m.type_(*tyidx) else {
                    panic!()
                };
                // The `as`s are all safe because the IR guarantees that no more than `width` bits
                // are set in the integer i.e. we are only ever truncating zeros.
                match width {
                    1 | 8 => dynasm!(asm; mov Rb(reg.code()), BYTE *x as i8),
                    16 => dynasm!(asm; mov Rw(reg.code()), WORD *x as i16),
                    32 => dynasm!(asm; mov Rd(reg.code()), DWORD *x as i32),
                    64 => dynasm!(asm; mov Rq(reg.code()), QWORD *x as i64),
                    _ => todo!(),
                }
            }
            Const::Ptr(x) => {
                dynasm!(asm; mov Rq(reg.code()), QWORD *x as i64)
            }
        }
    }

    /// Get an empty general purpose register, freeing one if necessary. Will not touch any
    /// registers set in `avoid`.
    fn get_empty_gp_reg(&mut self, asm: &mut Assembler, iidx: InstIdx, avoid: RegSet<Rq>) -> Rq {
        match self.gp_regset.find_empty_avoiding(avoid) {
            Some(reg) => reg,
            None => {
                // We need to find a register to spill. Our heuristic is two-fold:
                //   1. Spill the register whose value is used furthest away in the trace. This is
                //      a proxy for "the value is less likely to be used soon".
                //   2. If (1) leads to a tie, spill the "highest" register (e.g. prefer to spill
                //      R15 over RAX) because "lower" registers are more likely to be clobbered by
                //      CALLS, and we assume that the more recently we've put a value into a
                //      register, the more likely it is to be used again soon.
                let mut furthest = None;
                for reg in GP_REGS {
                    if avoid.is_set(reg) {
                        continue;
                    }
                    match self.gp_reg_states[usize::from(reg.code())] {
                        RegState::Reserved => (),
                        RegState::Empty => unreachable!(),
                        RegState::FromConst(_) => todo!(),
                        RegState::FromInst(from_iidx) => {
                            debug_assert!(self.is_inst_var_still_used_at(iidx, from_iidx));
                            if furthest.is_none() {
                                furthest = Some((reg, from_iidx));
                            } else if let Some((_, furthest_iidx)) = furthest {
                                if self.inst_vals_alive_until[usize::from(from_iidx)]
                                    >= self.inst_vals_alive_until[usize::from(furthest_iidx)]
                                {
                                    furthest = Some((reg, from_iidx))
                                }
                            }
                        }
                    }
                }

                match furthest {
                    Some((reg, _)) => {
                        self.spill_gp_if_not_already(asm, reg);
                        self.gp_regset.unset(reg);
                        self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                        reg
                    }
                    None => panic!("Cannot satisfy register constraints: no registers left"),
                }
            }
        }
    }

    /// Clobber each register in `regs`, spilling if it is used at or after instruction `iidx`, and
    /// (whether it is used later or not) marking the reg state as [RegState::Empty].
    ///
    /// FIXME: This method has one genuine use (clobbering registers before a CALL) and one hack
    /// use (hence the function name) in x64/mod.rs. What we currently call `avoids` should
    /// really be `clobbers`, but currently there is one case in x64/mod.rs which requires us to
    /// avoid a register without clobbering it, so we have to break this out into its own function.
    pub(crate) fn clobber_gp_regs_hack(&mut self, asm: &mut Assembler, iidx: InstIdx, regs: &[Rq]) {
        for reg in regs {
            match self.gp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => unreachable!(),
                RegState::Empty => (),
                RegState::FromConst(_) => {
                    self.gp_regset.unset(*reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegState::FromInst(from_iidx) => {
                    // OPT: We can MOV some of these rather than just spilling.
                    if self.is_inst_var_still_used_at(iidx, from_iidx) {
                        self.spill_gp_if_not_already(asm, *reg);
                    }
                    self.gp_regset.unset(*reg);
                    self.gp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
            }
        }
    }

    /// Return the location of the value at `iidx`. If that instruction's value is available in a
    /// register and is spilled to the stack, the former will always be preferred.
    ///
    /// Note that it is undefined behaviour to ask for the location of an instruction which has not
    /// yet produced a value.
    pub(crate) fn var_location(&mut self, iidx: InstIdx) -> VarLocation {
        if let Some(reg_i) = self.gp_reg_states.iter().position(|x| {
            if let RegState::FromInst(y) = x {
                *y == iidx
            } else {
                false
            }
        }) {
            VarLocation::Register(reg_alloc::Register::GP(GP_REGS[reg_i]))
        } else if let Some(reg_i) = self.fp_reg_states.iter().position(|x| {
            if let RegState::FromInst(y) = x {
                *y == iidx
            } else {
                false
            }
        }) {
            VarLocation::Register(reg_alloc::Register::FP(FP_REGS[reg_i]))
        } else {
            let (iidx, inst) = self.m.inst_deproxy(iidx);
            let size = inst.def_byte_size(self.m);
            match inst {
                Inst::ProxyInst(_) => panic!(),
                Inst::ProxyConst(cidx) => match self.m.const_(cidx) {
                    Const::Float(_, _) => todo!(),
                    Const::Int(tyidx, v) => {
                        let Ty::Integer(bits) = self.m.type_(*tyidx) else {
                            panic!()
                        };
                        VarLocation::ConstInt { bits: *bits, v: *v }
                    }
                    Const::Ptr(_) => todo!(),
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
                    SpillState::Indirect(off) => VarLocation::Indirect {
                        frame_off: off,
                        size,
                    },
                },
            }
        }
    }
}

/// The parts of the register allocator needed for floating point registers.
impl<'a> LSRegAlloc<'a> {
    /// Allocate registers for the instruction at position `iidx`.
    pub(crate) fn get_fp_regs<const N: usize>(
        &mut self,
        asm: &mut Assembler,
        iidx: InstIdx,
        constraints: [RegConstraint<Rx>; N],
    ) -> [Rx; N] {
        let mut avoid = RegSet::blank();
        let mut found_output = false; // Check that there aren't multiple output regs
        let mut out = [None; N];

        for cnstr in &constraints {
            match cnstr {
                RegConstraint::InputIntoReg(_, reg)
                | RegConstraint::InputIntoRegAndClobber(_, reg)
                | RegConstraint::OutputFromReg(reg) => avoid.set(*reg),
                RegConstraint::Input(_) | RegConstraint::InputOutput(_) | RegConstraint::Output => {
                }
                RegConstraint::InputOutputIntoReg(_, _) | RegConstraint::Temporary => {
                    panic!();
                }
            }
        }

        // If we already have the value in a register, don't allocate a new register.
        for (i, cnstr) in constraints.iter().enumerate() {
            match cnstr {
                RegConstraint::Input(op) | RegConstraint::InputOutput(op) => match op {
                    Operand::Local(op_iidx) => {
                        if let Some(reg_i) = self.fp_reg_states.iter().position(|x| {
                            if let RegState::FromInst(y) = x {
                                y == op_iidx
                            } else {
                                false
                            }
                        }) {
                            let reg = FP_REGS[reg_i];
                            if !avoid.is_set(reg) {
                                assert!(self.fp_regset.is_set(reg));
                                avoid.set(reg);
                                out[i] = Some(reg);
                                if let RegConstraint::InputOutput(_) = cnstr {
                                    debug_assert!(!found_output);
                                    found_output = true;
                                    if self.is_inst_var_still_used_after(iidx, *op_iidx) {
                                        self.spill_fp_if_not_already(asm, reg);
                                    }
                                    self.fp_reg_states[usize::from(reg.code())] =
                                        RegState::FromInst(iidx);
                                }
                            }
                        }
                    }
                    Operand::Const(_cidx) => (),
                },
                RegConstraint::InputIntoReg(_op, _reg)
                | RegConstraint::InputIntoRegAndClobber(_op, _reg) => {
                    // OPT: do the same trick as Input/InputOutput
                }
                RegConstraint::Output | RegConstraint::OutputFromReg(_) => (),
                RegConstraint::InputOutputIntoReg(_op, _reg) => unreachable!(),
                RegConstraint::Temporary => todo!(),
            }
        }

        for (i, x) in constraints.iter().enumerate() {
            if out[i].is_some() {
                // We've already allocated this constraint.
                continue;
            }
            match x {
                RegConstraint::Input(op)
                | RegConstraint::InputIntoReg(op, _)
                | RegConstraint::InputIntoRegAndClobber(op, _)
                | RegConstraint::InputOutput(op) => {
                    let reg = match x {
                        RegConstraint::Input(_) | RegConstraint::InputOutput(_) => {
                            self.get_empty_fp_reg(asm, iidx, avoid)
                        }
                        RegConstraint::InputIntoReg(_, reg)
                        | RegConstraint::InputIntoRegAndClobber(_, reg) => {
                            // OPT: Not everything needs spilling
                            self.spill_fp_if_not_already(asm, *reg);
                            *reg
                        }
                        RegConstraint::InputOutputIntoReg(_, _) => {
                            unreachable!()
                        }
                        RegConstraint::Output
                        | RegConstraint::OutputFromReg(_)
                        | RegConstraint::Temporary => {
                            unreachable!()
                        }
                    };

                    // At this point we know the value in `reg` has been spilled if necessary, so
                    // we can overwrite it.
                    match op {
                        Operand::Local(op_iidx) => {
                            self.force_fp_unspill(asm, *op_iidx, reg);
                        }
                        Operand::Const(cidx) => {
                            // FIXME: we could reuse consts in regs
                            self.load_const_into_fp_reg(asm, *cidx, reg);
                        }
                    }

                    self.fp_regset.set(reg);
                    out[i] = Some(reg);
                    avoid.set(reg);
                    let st = match x {
                        RegConstraint::Input(_) | RegConstraint::InputIntoReg(_, _) => match op {
                            Operand::Local(op_iidx) => RegState::FromInst(*op_iidx),
                            Operand::Const(cidx) => RegState::FromConst(*cidx),
                        },
                        RegConstraint::InputIntoRegAndClobber(_, _) => {
                            self.fp_regset.unset(reg);
                            RegState::Empty
                        }
                        RegConstraint::InputOutput(_) => {
                            debug_assert!(!found_output);
                            found_output = true;
                            RegState::FromInst(iidx)
                        }
                        RegConstraint::InputOutputIntoReg(_, _)
                        | RegConstraint::Output
                        | RegConstraint::OutputFromReg(_) => {
                            unreachable!()
                        }
                        RegConstraint::Temporary => todo!(),
                    };
                    self.fp_reg_states[usize::from(reg.code())] = st;
                }
                RegConstraint::Output => {
                    let reg = self.get_empty_fp_reg(asm, iidx, avoid);
                    self.fp_regset.set(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                    avoid.set(reg);
                    out[i] = Some(reg);
                }
                RegConstraint::OutputFromReg(reg) => {
                    // OPT: Don't have to always spill.
                    self.spill_fp_if_not_already(asm, *reg);
                    self.fp_regset.set(*reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::FromInst(iidx);
                    avoid.set(*reg);
                    out[i] = Some(*reg);
                }
                RegConstraint::InputOutputIntoReg(_, _) => unreachable!(),
                RegConstraint::Temporary => todo!(),
            }
        }

        out.map(|x| x.unwrap())
    }

    /// If the value stored in `reg` is not already spilled to the heap, then spill it. Note that
    /// this function neither writes to the register or changes the register's [RegState].
    fn spill_fp_if_not_already(&mut self, asm: &mut Assembler, reg: Rx) {
        match self.fp_reg_states[usize::from(reg.code())] {
            RegState::Reserved | RegState::Empty | RegState::FromConst(_) => (),
            RegState::FromInst(iidx) => {
                if self.spills[usize::from(iidx)] == SpillState::Empty {
                    let inst = self.m.inst_no_proxies(iidx);
                    let size = inst.def_byte_size(self.m);
                    self.stack.align(size); // FIXME
                    let frame_off = self.stack.grow(size);
                    let off = i32::try_from(frame_off).unwrap();
                    match size {
                        4 => dynasm!(asm ; movss [rbp - off], Rx(reg.code())),
                        8 => dynasm!(asm ; movsd [rbp - off], Rx(reg.code())),
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
        let inst = self.m.inst_no_proxies(iidx);
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
                    todo!();
                    // dynasm!(asm; mov Rq(reg.code()), Rq(cur_reg.code()));
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
            SpillState::Indirect(off) => {
                let tmp_reg = Rq::RAX;
                match size {
                    4 => dynasm!(asm
                        ; push Rq(tmp_reg.code())
                        ; mov Rq(tmp_reg.code()), [rbp]
                        ; movss Rx(reg.code()), [Rq(tmp_reg.code()) + off]
                        ; pop Rq(tmp_reg.code())
                    ),
                    _ => todo!("{}", size),
                };
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

    /// Get an empty general purpose register, freeing one if necessary. Will not touch any
    /// registers set in `avoid`.
    fn get_empty_fp_reg(&mut self, asm: &mut Assembler, iidx: InstIdx, avoid: RegSet<Rx>) -> Rx {
        match self.fp_regset.find_empty_avoiding(avoid) {
            Some(reg) => reg,
            None => {
                // We need to find a register to spill. Our heuristic is two-fold:
                //   1. Spill the register whose value is used furthest away in the trace. This is
                //      a proxy for "the value is less likely to be used soon".
                //   2. If (1) leads to a tie, spill the "highest" register (e.g. prefer to spill
                //      XMM15 over XMM0) because "lower" registers are more likely to be clobbered
                //      by CALLS, and we assume that the more recently we've put a value into a
                //      register, the more likely it is to be used again soon.
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
                            debug_assert!(self.is_inst_var_still_used_at(iidx, from_iidx));
                            if furthest.is_none() {
                                furthest = Some((reg, from_iidx));
                            } else if let Some((_, furthest_iidx)) = furthest {
                                if self.inst_vals_alive_until[usize::from(from_iidx)]
                                    >= self.inst_vals_alive_until[usize::from(furthest_iidx)]
                                {
                                    furthest = Some((reg, from_iidx))
                                }
                            }
                        }
                    }
                }

                match furthest {
                    Some((reg, _)) => {
                        self.spill_fp_if_not_already(asm, reg);
                        self.fp_regset.unset(reg);
                        self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                        reg
                    }
                    None => panic!("Cannot satisfy register constraints: no registers left"),
                }
            }
        }
    }

    /// Clobber all floating point registers. Used before a CALL.
    pub(crate) fn clobber_fp_regs(&mut self, asm: &mut Assembler, iidx: InstIdx) {
        for reg in FP_REGS {
            match self.fp_reg_states[usize::from(reg.code())] {
                RegState::Reserved => unreachable!(),
                RegState::Empty => (),
                RegState::FromConst(_) => {
                    self.fp_regset.unset(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
                RegState::FromInst(from_iidx) => {
                    // OPT: We can MOV some of these rather than just spilling.
                    if self.is_inst_var_still_used_at(iidx, from_iidx) {
                        self.spill_fp_if_not_already(asm, reg);
                    }
                    self.fp_regset.unset(reg);
                    self.fp_reg_states[usize::from(reg.code())] = RegState::Empty;
                }
            }
        }
    }
}

/// What constraints are there on registers for an instruction?
#[derive(Debug)]
pub(crate) enum RegConstraint<R: Register> {
    /// Make sure `Operand` is loaded into a register *R* on entry; its value must be unchanged
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
    /// The result of this instruction will be stored in register `R`.
    OutputFromReg(R),
    /// A temporary register *x*: it will be clobbered by the instruction and left in an
    /// indeterminate state on exit.
    Temporary,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RegState {
    Reserved,
    Empty,
    FromConst(ConstIdx),
    FromInst(InstIdx),
}

/// Which registers in a set of 16 registers are currently used? Happily 16 bits is the right size
/// for (separately) both x64's general purpose and floating point registers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegSet<R>(u16, PhantomData<R>);

impl<R: Register> RegSet<R> {
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
    Stack(i32),
    /// This variable is spilt to the stack with the same semantics as [VarLocation::Direct].
    Direct(i32),
    /// This variable is spilt to the stack with the same semantics as [VarLocation::Indirect].
    Indirect(i32),
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
