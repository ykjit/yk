//! The X64 JIT Code Generator.
//!
//! Conventions used in this module:
//!   * Functions with a `cg_X` prefix generate code for a [jit_ir] construct `X`.
//!   * Helper functions arguments are in order `(<destination>, <source_1>, ... <source_n>)`.
//!
//! FIXME: the code generator clobbers registers willy-nilly because at the time of writing we have
//! a register allocator that doesn't actually use any registers. Later we will have to audit the
//! backend and insert register save/restore for clobbered registers.

use super::{
    super::{
        int_signs::{SignExtend, Truncate},
        jit_ir::{self, BinOp, FloatTy, InstIdx, Module, Operand, Ty},
        CompilationError,
    },
    reg_alloc::{self, StackDirection, VarLocation},
    CodeGen,
};
#[cfg(any(debug_assertions, test))]
use crate::compile::jitc_yk::gdb::{self, GdbCtx};
use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        jitc_yk::{
            aot_ir,
            jit_ir::{Const, IndirectCallIdx, InlinedFrame},
            RootTracePtr, YkSideTraceInfo,
        },
        CompiledTrace, Guard, GuardIdx, SideTraceInfo,
    },
    location::{HotLocation, HotLocationKind},
    mt::MT,
};
use dynasmrt::{
    components::StaticLabel,
    dynasm,
    x64::{Rq, Rx},
    AssemblyOffset, DynamicLabel, DynasmApi, DynasmError, DynasmLabelApi, ExecutableBuffer,
    Register,
};
use indexmap::IndexMap;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};
use std::{cell::Cell, slice};
use std::{collections::HashMap, error::Error};
use ykaddr::addr::symbol_to_ptr;
use yksmp;

mod deopt;
mod lsregalloc;

use deopt::{__yk_deopt, __yk_guardcheck, __yk_reenter_jit};
use lsregalloc::{LSRegAlloc, RegConstraint, RegSet};

/// General purpose argument registers as defined by the x64 SysV ABI.
static ARG_GP_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// The registers clobbered by a function call in the x64 SysV ABI.
static CALLER_CLOBBER_REGS: [Rq; 9] = [
    Rq::RAX,
    Rq::RCX,
    Rq::RDX,
    Rq::RSI,
    Rq::RDI,
    Rq::R8,
    Rq::R9,
    Rq::R10,
    Rq::R11,
];

/// Floating point argument registers as defined by the x64 SysV ABI.
static ARG_FP_REGS: [Rx; 8] = [
    Rx::XMM0,
    Rx::XMM1,
    Rx::XMM2,
    Rx::XMM3,
    Rx::XMM4,
    Rx::XMM5,
    Rx::XMM6,
    Rx::XMM7,
];

/// Registers used by stackmaps to store live variables.
static STACKMAP_GP_REGS: [Rq; 7] = [
    Rq::RBX,
    Rq::R12,
    Rq::R13,
    Rq::R14,
    Rq::R15,
    Rq::RSI,
    Rq::RDI,
];

/// The argument index of the live variables struct argument in the JITted code function.
static JITFUNC_LIVEVARS_ARGIDX: usize = 0;

/// The size of a 64-bit register in bytes.
pub(crate) static REG64_SIZE: usize = 8;
static RBP_DWARF_NUM: u16 = 6;

/// The x64 SysV ABI requires a 16-byte aligned stack prior to any call.
const SYSV_CALL_STACK_ALIGN: usize = 16;

/// On x64 the stack grows down.
const STACK_DIRECTION: StackDirection = StackDirection::GrowsDown;

/// A function that we can put a debugger breakpoint on.
/// FIXME: gross hack.
#[cfg(debug_assertions)]
#[no_mangle]
#[inline(never)]
pub extern "C" fn __yk_break() {}

/// A simple front end for the X64 code generator.
pub(crate) struct X64CodeGen;

impl CodeGen for X64CodeGen {
    fn codegen(
        &self,
        m: Module,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
        sp_offset: Option<usize>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        Assemble::new(&m, sp_offset)?.codegen(mt, hl, sp_offset.is_some())
    }
}

impl X64CodeGen {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self))
    }
}

/// The x64 code generator.
struct Assemble<'a> {
    m: &'a jit_ir::Module,
    ra: LSRegAlloc<'a>,
    /// The locations of the live variables at the beginning of the loop.
    loop_start_locs: Vec<VarLocation>,
    asm: dynasmrt::x64::Assembler,
    /// Deopt info, with one entry per guard, in the order that the guards appear in the trace.
    deoptinfo: HashMap<usize, DeoptInfo>,
    ///
    /// Maps assembly offsets to comments.
    ///
    /// Comments used by the trace printer for debugging and testing only.
    ///
    /// Each assembly offset can have zero or more comment lines.
    comments: Cell<IndexMap<usize, Vec<String>>>,
    /// Stack pointer offset from the base pointer of the interpreter frame. If this is a root
    /// trace it's initialised to the size of the interpreter frame. Otherwise its value is passed
    /// in via [YkSideTraceInfo::sp_offset].
    sp_offset: usize,
}

impl<'a> Assemble<'a> {
    fn new(
        m: &'a jit_ir::Module,
        sp_offset: Option<usize>,
    ) -> Result<Box<Assemble<'a>>, CompilationError> {
        #[cfg(debug_assertions)]
        m.assert_well_formed();

        let asm = dynasmrt::x64::Assembler::new()
            .map_err(|e| CompilationError::ResourceExhausted(Box::new(e)))?;
        // Since we are executing the trace in the main interpreter frame we need this to
        // initialise the trace's register allocator in order to access local variables.
        let sp_offset = if let Some(off) = sp_offset {
            // This is a side-trace. Use the passed in stack size to initialise the register
            // allocator.
            off
        } else {
            // This is a normal trace, so we need to retrieve the stack size of the main
            // interpreter frame.
            // FIXME: For now the control point stackmap id is always 0. Though
            // we likely want to support multiple control points in the future. We can either pass
            // the correct stackmap id in via the control point, or compute the stack size
            // dynamically upon entering the control point (e.g. by subtracting the current RBP
            // from the previous RBP).
            if let Ok(sm) = AOT_STACKMAPS.as_ref() {
                let (rec, pinfo) = sm.get(0);
                let size = if pinfo.hasfp {
                    // The frame size includes the pushed RBP, but since we only care about the size of
                    // the local variables we need to subtract it again.
                    rec.size - u64::try_from(REG64_SIZE).unwrap()
                } else {
                    rec.size
                };
                usize::try_from(size).unwrap()
            } else {
                // The unit tests in this file don't have AOT code. So if we don't find stackmaps here
                // that's ok. In real-world programs and our C-tests this shouldn't happen though.
                #[cfg(not(test))]
                panic!("Couldn't find AOT stackmaps.");
                #[cfg(test)]
                0
            }
        };

        Ok(Box::new(Self {
            m,
            ra: LSRegAlloc::new(m, sp_offset),
            asm,
            loop_start_locs: Vec::new(),
            deoptinfo: HashMap::new(),
            comments: Cell::new(IndexMap::new()),
            sp_offset,
        }))
    }

    fn codegen(
        mut self: Box<Self>,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
        issidetrace: bool,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        if issidetrace {
            // Recover registers.
            for reg in lsregalloc::FP_REGS.iter() {
                dynasm!(self.asm
                    ; pop rcx
                    ; movq Rx(reg.code()), rcx
                );
            }
            for reg in lsregalloc::GP_REGS.iter() {
                dynasm!(self.asm; pop Rq(reg.code()));
            }
            // Re-align stack. This was misaligned when we pushed RSI in the guard failure routine.
            // This isn't really neccessary since we are aligning the stack when we calculate the
            // stack size for this trace. However, this removes the pushed RSI register which
            // serves no further pupose at this point.
            dynasm!(self.asm; add rsp, 8);
        }

        let alloc_off = self.emit_prologue();

        for (iidx, inst) in self.m.iter_skipping_insts() {
            self.ra.expire_regs(iidx);
            self.cg_inst(iidx, inst)?;
        }

        if !self.deoptinfo.is_empty() {
            // We now have to construct the "full" deopt points. Inside the trace itself, are just
            // a pair of instructions: a `cmp` followed by a `jnz` to a `fail_label` that has not
            // yet been defined. We now have to construct a full call to `__yk_deopt` for each of
            // those labels. Since, in general, we'll have multiple guards, we construct a simple
            // stub which puts an ID in a register then JMPs to (shared amongst all guards) code
            // which does the full call to __yk_deopt.
            #[allow(unused_mut)] // `mut` required in debug builds. See below.
            let mut infos = self
                .deoptinfo
                .iter()
                .map(|(id, l)| (*id, l.fail_label))
                .collect::<Vec<_>>();
            // Debugging deopt asm is much easier if the stubs are in order.
            #[cfg(debug_assertions)]
            infos.sort_by(|a, b| a.0.cmp(&b.0));

            let guardcheck_label = self.asm.new_dynamic_label();
            for (deoptid, fail_label) in infos {
                self.comment(
                    self.asm.offset(),
                    format!("Deopt ID for guard {:?}", deoptid),
                );
                // FIXME: Why are `deoptid`s 64 bit? We're not going to have that many guards!
                let deoptid = i32::try_from(deoptid).unwrap();
                dynasm!(self.asm
                    ;=> fail_label
                    ; push rsi // FIXME: We push RSI now so we can fish it back out in
                               // `deopt_label`.
                    ; mov rsi, deoptid
                    ; jmp => guardcheck_label
                );
            }

            let deopt_label = self.asm.new_dynamic_label();
            self.comment(self.asm.offset(), "Call __yk_deopt".to_string());
            // Clippy points out that `__yk_depot as i64` isn't portable, but since this entire module
            // is x86 only, we don't need to worry about portability.
            #[allow(clippy::fn_to_numeric_cast)]
            {
                dynasm!(self.asm; => guardcheck_label);
                // Push all the general purpose registers to the stack.
                for (i, reg) in lsregalloc::GP_REGS.iter().rev().enumerate() {
                    if *reg == Rq::RSI {
                        // RSI is handled differently in `fail_label`: RSI now contains the deopt
                        // ID, so we have to fish the actual value out of the stack. See the FIXME
                        // in `fail_label`.
                        let off = i32::try_from(i * 8).unwrap();
                        dynasm!(self.asm; push QWORD [rsp + off]);
                    } else {
                        dynasm!(self.asm; push Rq(reg.code()));
                    }
                }
                dynasm!(self.asm; mov rdx, rsp);
                for reg in lsregalloc::FP_REGS.iter().rev() {
                    dynasm!(self.asm
                        ; movq rcx, Rx(reg.code())
                        ; push rcx
                    );
                }
                dynasm!(self.asm; mov rcx, rsp);

                // Check whether we need to deoptimise or jump into a side-trace.
                dynasm!(self.asm
                    ; push rsi  // Save `deoptid`.
                    ; push rdx  // Save `gp_regs` pointer.
                    ; push rcx  // Save `fp_regs` pointer.
                    ; mov rdi, rsi
                    ; mov rax, QWORD __yk_guardcheck as i64
                    ; call rax
                    ; pop rcx
                    ; pop rdx
                    ; pop rsi
                    ; cmp rax, 0
                    ; je => deopt_label
                );

                // Jump into side-trace. The side-trace takes care of recovering the saved
                // registers.
                dynasm!(self.asm
                    ; jmp rax
                );

                // Deoptimise.
                dynasm!(self.asm; => deopt_label);
                dynasm!(self.asm
                    ; mov rdi, rbp
                    ; mov rax, QWORD __yk_deopt as i64
                    ; sub rsp, 8 // Align the stack
                    ; call rax
                );
            }
        }

        // Now we know the size of the stack frame (i.e. self.asp), patch the allocation with the
        // correct amount.
        self.patch_frame_allocation(alloc_off);

        // If an error happens here, we've made a mistake in the assembly we generate.
        self.asm
            .commit()
            .map_err(|e| CompilationError::InternalError(format!("When committing: {e}")))?;

        // This unwrap cannot fail if `commit` (above) succeeded.
        let buf = self.asm.finalize().unwrap();

        #[cfg(any(debug_assertions, test))]
        let gdb_ctx = gdb::register_jitted_code(
            self.m.ctr_id(),
            buf.ptr(AssemblyOffset(0)),
            buf.size(),
            self.comments.get_mut(),
        )?;

        Ok(Arc::new(X64CompiledTrace {
            mt,
            buf,
            deoptinfo: self.deoptinfo,
            sp_offset: self.ra.stack_size(),
            entry_vars: self.loop_start_locs.clone(),
            hl: Arc::downgrade(&hl),
            comments: self.comments.take(),
            #[cfg(any(debug_assertions, test))]
            gdb_ctx,
        }))
    }

    /// Codegen an instruction.
    fn cg_inst(
        &mut self,
        iidx: jit_ir::InstIdx,
        inst: &jit_ir::Inst,
    ) -> Result<(), CompilationError> {
        self.comment(self.asm.offset(), inst.display(iidx, self.m).to_string());

        match inst {
            #[cfg(test)]
            jit_ir::Inst::BlackBox(_) => unreachable!(),
            jit_ir::Inst::Const(_) | jit_ir::Inst::Copy(_) | jit_ir::Inst::Tombstone => {
                unreachable!();
            }

            jit_ir::Inst::BinOp(i) => self.cg_binop(iidx, i),
            jit_ir::Inst::LoadTraceInput(i) => self.cg_loadtraceinput(iidx, i),
            jit_ir::Inst::Load(i) => self.cg_load(iidx, i),
            jit_ir::Inst::PtrAdd(i) => self.cg_ptradd(iidx, i),
            jit_ir::Inst::DynPtrAdd(i) => self.cg_dynptradd(iidx, i),
            jit_ir::Inst::Store(i) => self.cg_store(iidx, i),
            jit_ir::Inst::LookupGlobal(i) => self.cg_lookupglobal(iidx, i),
            jit_ir::Inst::Call(i) => self.cg_call(iidx, i)?,
            jit_ir::Inst::IndirectCall(i) => self.cg_indirectcall(iidx, i)?,
            jit_ir::Inst::ICmp(i) => self.cg_icmp(iidx, i),
            jit_ir::Inst::Guard(i) => self.cg_guard(iidx, i),
            jit_ir::Inst::TraceLoopStart => self.cg_traceloopstart(),
            jit_ir::Inst::TraceLoopJump => self.cg_traceloopjump(),
            jit_ir::Inst::RootJump => self.cg_rootjump(self.m.root_jump_addr()),
            jit_ir::Inst::SExt(i) => self.cg_sext(iidx, i),
            jit_ir::Inst::ZeroExtend(i) => self.cg_zeroextend(iidx, i),
            jit_ir::Inst::Trunc(i) => self.cg_trunc(iidx, i),
            jit_ir::Inst::Select(i) => self.cg_select(iidx, i),
            jit_ir::Inst::SIToFP(i) => self.cg_sitofp(iidx, i),
            jit_ir::Inst::FPExt(i) => self.cg_fpext(iidx, i),
            jit_ir::Inst::FCmp(i) => self.cg_fcmp(iidx, i),
            jit_ir::Inst::FPToSI(i) => self.cg_fptosi(iidx, i),
        }
        Ok(())
    }

    /// Add a comment to the trace, for use when disassembling its native code.
    fn comment(&mut self, off: AssemblyOffset, line: String) {
        self.comments.get_mut().entry(off.0).or_default().push(line);
    }

    /// Emit the prologue of the JITted code.
    ///
    /// The JITted code is executed inside the same frame as the main interpreter loop. This allows
    /// us to easily access live variables on that frame's stack. Because of this we don't need to
    /// create a new frame here, though we do need to make space for any extra stack space this
    /// trace needs.
    ///
    /// Note that there is no correspoinding `emit_epilogue()`. This is because the only way out of
    /// JITted code is via deoptimisation, which will rewrite the whole stack anyway.
    ///
    /// Returns the offset at which to patch up the stack allocation later.
    fn emit_prologue(&mut self) -> AssemblyOffset {
        self.comment(self.asm.offset(), "prologue".to_owned());

        // Emit a dummy frame allocation instruction that initially allocates 0 bytes, but will be
        // patched later when we know how big the frame needs to be.
        let alloc_off = self.asm.offset();
        dynasm!(self.asm
            ; sub rsp, DWORD 0
        );

        // In debug mode, add a call to `__yk_break` to make debugging easier. Note that this
        // clobbers r11 (a caller saved register).
        #[cfg(debug_assertions)]
        {
            self.comment(self.asm.offset(), "Breakpoint hack".into());
            // Clippy points out that `__yk_depot as i64` isn't portable, but since this entire
            // module is x86 only, we don't need to worry about portability.
            #[allow(clippy::fn_to_numeric_cast)]
            {
                dynasm!(self.asm
                    ; push r11
                    ; mov r11, QWORD __yk_break as i64
                    ; call r11
                    ; pop r11
                );
            }
        }

        alloc_off
    }

    fn patch_frame_allocation(&mut self, asm_off: AssemblyOffset) {
        // The stack should be 16-byte aligned after allocation. This ensures that calls in the
        // trace also get a 16-byte aligned stack, as per the SysV ABI.
        // Since we initialise the register allocator with interpreter frame and parent trace
        // frames, the actual size we need to substract from RSP is the difference between the
        // current stack size and the base size we inherited.
        let stack_size = self.ra.align_stack(SYSV_CALL_STACK_ALIGN) - self.sp_offset;

        match i32::try_from(stack_size) {
            Ok(asp) => {
                let mut patchup = self.asm.alter_uncommitted();
                patchup.goto(asm_off);
                dynasm!(patchup
                    // The size of this instruction must be the exactly the same as the dummy
                    // allocation instruction that was emitted during `emit_prologue()`.
                    ; sub rsp, DWORD asp
                );
            }
            Err(_) => {
                // If we get here, then the frame was so big that the dummy instruction we had
                // planned to patch isn't big enough to encode the desired allocation size. Cross
                // this bridge if/when we get to it.
                todo!();
            }
        }
    }

    fn cg_binop(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::BinOpInst) {
        let lhs = inst.lhs(self.m);
        let rhs = inst.rhs(self.m);

        match inst.binop() {
            BinOp::Add => {
                let byte_size = lhs.byte_size(self.m);
                match (&lhs, &rhs) {
                    (Operand::Const(cidx), Operand::Var(_))
                    | (Operand::Var(_), Operand::Const(cidx)) => {
                        // Addition involves a constant. We may be able to emit more optimal code.
                        let Const::Int(ctyidx, v) = self.m.const_(*cidx) else {
                            unreachable!()
                        };
                        let Ty::Integer(bit_size) = self.m.type_(*ctyidx) else {
                            unreachable!()
                        };
                        // If it's a 64-bit add and the numeric value of the constant can be
                        // expressed in 32-bits...
                        //
                        // We are only optimising (exactly) 64-bit add operations for now, but
                        // there is certainly oppertunity to optimised other cases later.
                        //
                        // We could have used Rust `as` casts to truncate and sign-extend here,
                        // since we are currently only dealing with i32s and i64s, but if/when we
                        // want to cover operations on other "odd-bit-size" integers we will need
                        // these custom implementations.
                        if *bit_size == 64 && v.truncate(32).sign_extend(32, 64) == *v {
                            let v32 = v.truncate(32);
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [RegConstraint::InputOutput(lhs)],
                            );
                            dynasm!(self.asm; add Rq(lhs_reg.code()), v32 as i32);
                            return;
                        }
                        // Same optimisation, but for 32-bit add.
                        //
                        // This time the constant fits by definition.
                        if *bit_size == 32 {
                            let v32 = v.truncate(32);
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [RegConstraint::InputOutput(lhs)],
                            );
                            dynasm!(self.asm; add Rd(lhs_reg.code()), v32 as i32);
                            return;
                        }
                    }
                    _ => (),
                }

                let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match byte_size {
                    1 => dynasm!(self.asm; add Rb(lhs_reg.code()), Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; add Rw(lhs_reg.code()), Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; add Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; add Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::And => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    1 => dynasm!(self.asm; and Rb(lhs_reg.code()), Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; and Rw(lhs_reg.code()), Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; and Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; and Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::AShr => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, _rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::InputOutput(lhs),
                        RegConstraint::InputIntoReg(rhs, Rq::RCX),
                    ],
                );
                debug_assert_eq!(_rhs_reg, Rq::RCX);
                match size {
                    1 => dynasm!(self.asm; sar Rb(lhs_reg.code()), cl),
                    2 => dynasm!(self.asm; sar Rw(lhs_reg.code()), cl),
                    4 => dynasm!(self.asm; sar Rd(lhs_reg.code()), cl),
                    8 => dynasm!(self.asm; sar Rq(lhs_reg.code()), cl),
                    _ => todo!(),
                }
            }
            BinOp::LShr => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, _rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::InputOutput(lhs),
                        RegConstraint::InputIntoReg(rhs, Rq::RCX),
                    ],
                );
                debug_assert_eq!(_rhs_reg, Rq::RCX);
                match size {
                    1 => dynasm!(self.asm; shr Rb(lhs_reg.code()), cl),
                    2 => dynasm!(self.asm; shr Rw(lhs_reg.code()), cl),
                    4 => dynasm!(self.asm; shr Rd(lhs_reg.code()), cl),
                    8 => dynasm!(self.asm; shr Rq(lhs_reg.code()), cl),
                    _ => todo!(),
                }
            }
            BinOp::Shl => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, _rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::InputOutput(lhs),
                        RegConstraint::InputIntoReg(rhs, Rq::RCX),
                    ],
                );
                debug_assert_eq!(_rhs_reg, Rq::RCX);
                match size {
                    1 => dynasm!(self.asm; shl Rb(lhs_reg.code()), cl),
                    2 => dynasm!(self.asm; shl Rw(lhs_reg.code()), cl),
                    4 => dynasm!(self.asm; shl Rd(lhs_reg.code()), cl),
                    8 => dynasm!(self.asm; shl Rq(lhs_reg.code()), cl),
                    _ => todo!(),
                }
            }
            BinOp::Mul => {
                let size = lhs.byte_size(self.m);
                self.ra
                    .clobber_gp_regs_hack(&mut self.asm, iidx, &[Rq::RDX]);
                let [_lhs_reg, rhs_reg] = self.ra.assign_gp_regs_avoiding(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::InputOutputIntoReg(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                    ],
                    RegSet::from(Rq::RDX),
                );
                debug_assert_eq!(_lhs_reg, Rq::RAX);
                match size {
                    1 => dynasm!(self.asm; mul Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; mul Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; mul Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; mul Rq(rhs_reg.code())),
                    _ => todo!(),
                }
                // Note that because we are code-genning an unchecked multiply, the higher-order part of
                // the result in RDX is entirely ignored.
            }
            BinOp::Or => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    1 => dynasm!(self.asm; or Rb(lhs_reg.code()), Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; or Rw(lhs_reg.code()), Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; or Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; or Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::SDiv => {
                let size = lhs.byte_size(self.m);
                self.ra
                    .clobber_gp_regs_hack(&mut self.asm, iidx, &[Rq::RDX]);
                let [_lhs_reg, rhs_reg] = self.ra.assign_gp_regs_avoiding(
                    &mut self.asm,
                    iidx,
                    [
                        // The quotient is stored in RAX. We don't care about the remainder stored
                        // in RDX.
                        RegConstraint::InputOutputIntoReg(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                    ],
                    RegSet::from(Rq::RDX),
                );
                // The dividend is hard-coded into DX:AX/EDX:EAX/RDX:RAX. However unless we have 128bit
                // values or want to optimise register usage, we won't be needing this, and just zero out
                // RDX.

                // Signed division (idiv) operates on the DX:AX, EDX:EAX, RDX:RAX registers, so we
                // use `cdq`/`cqo` to double the size via sign extension and store the result in
                // DX:AX, EDX:EAX, RDX:RAX.
                match size {
                    // There's no `cwd` equivalent for byte-sized values, so we use `movsx`
                    // (sign-extend) instead.
                    1 => dynasm!(self.asm; movsx ax, al; idiv Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; cwd; idiv Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; cdq; idiv Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; cqo; idiv Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::SRem => {
                // The dividend is hard-coded into DX:AX/EDX:EAX/RDX:RAX. However unless we have 128bit
                // values or want to optimise register usage, we won't be needing this, and just zero out
                // RDX.
                let size = lhs.byte_size(self.m);
                debug_assert!(size == 4 || size == 8);
                let [_lhs_reg, rhs_reg, _rem_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::InputIntoRegAndClobber(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                        RegConstraint::OutputFromReg(Rq::RDX),
                    ],
                );
                debug_assert_eq!(_lhs_reg, Rq::RAX);
                debug_assert_eq!(_rem_reg, Rq::RDX);
                dynasm!(self.asm; xor rdx, rdx);
                match size {
                    1 => dynasm!(self.asm; idiv Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; idiv Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; idiv Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; idiv Rq(rhs_reg.code())),
                    _ => todo!(),
                }
                // The remainder is stored in RDX. We don't care about the quotient stored in RAX.
            }
            BinOp::Sub => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    1 => dynasm!(self.asm; sub Rb(lhs_reg.code()), Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; sub Rw(lhs_reg.code()), Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; sub Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; sub Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::Xor => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    1 => dynasm!(self.asm; xor Rb(lhs_reg.code()), Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; xor Rw(lhs_reg.code()), Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; xor Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; xor Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::UDiv => {
                let size = lhs.byte_size(self.m);
                self.ra
                    .clobber_gp_regs_hack(&mut self.asm, iidx, &[Rq::RDX]);
                let [_lhs_reg, rhs_reg] = self.ra.assign_gp_regs_avoiding(
                    &mut self.asm,
                    iidx,
                    [
                        // The quotient is stored in RAX. We don't care about the remainder stored
                        // in RDX.
                        RegConstraint::InputOutputIntoReg(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                    ],
                    RegSet::from(Rq::RDX),
                );
                debug_assert_eq!(_lhs_reg, Rq::RAX);
                // Like SDiv the dividend goes into AX, DX:AX, EDX:EAX, RDX:RAX. But since the
                // values aren't signed we don't need to sign-extend them and can just zero out
                // `rdx`.
                dynasm!(self.asm; xor rdx, rdx);
                match size {
                    1 => dynasm!(self.asm; div Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; div Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; div Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; div Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FDiv => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    4 => dynasm!(self.asm; divss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; divsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FAdd => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    4 => dynasm!(self.asm; addss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; addsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FMul => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    4 => dynasm!(self.asm; mulss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; mulsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FSub => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    4 => dynasm!(self.asm; subss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; subsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            x => todo!("{x:?}"),
        }
    }

    /// Codegen a [jit_ir::LoadTraceInputInst]. This only informs the register allocator about the
    /// locations of live variables without generating any actual machine code.
    fn cg_loadtraceinput(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadTraceInputInst) {
        let m = match &self.m.tilocs()[usize::try_from(inst.locidx()).unwrap()] {
            yksmp::Location::Register(0, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::RAX))
            }
            yksmp::Location::Register(1, ..) => {
                // Since the control point passes the stackmap ID via RDX this case only happens in
                // side-traces.
                VarLocation::Register(reg_alloc::Register::GP(Rq::RDX))
            }
            yksmp::Location::Register(2, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::RCX))
            }
            yksmp::Location::Register(3, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::RBX))
            }
            yksmp::Location::Register(4, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::RSI))
            }
            yksmp::Location::Register(5, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::RDI))
            }
            yksmp::Location::Register(8, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R8))
            }
            yksmp::Location::Register(9, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R9))
            }
            yksmp::Location::Register(10, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R10))
            }
            yksmp::Location::Register(11, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R11))
            }
            yksmp::Location::Register(12, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R12))
            }
            yksmp::Location::Register(13, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R13))
            }
            yksmp::Location::Register(14, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R14))
            }
            yksmp::Location::Register(15, ..) => {
                VarLocation::Register(reg_alloc::Register::GP(Rq::R15))
            }
            yksmp::Location::Register(x, ..) if *x >= 17 && *x <= 32 => VarLocation::Register(
                reg_alloc::Register::FP(lsregalloc::FP_REGS[usize::from(x - 17)]),
            ),
            yksmp::Location::Direct(6, off, size) => VarLocation::Direct {
                frame_off: *off,
                size: usize::from(*size),
            },
            // Since the trace shares the same stack frame as the main interpreter loop, we can
            // translate indirect locations into normal stack locations. Note that while stackmaps
            // use negative offsets, we use positive offsets for stack locations.
            yksmp::Location::Indirect(6, off, size) => VarLocation::Stack {
                frame_off: u32::try_from(*off * -1).unwrap(),
                size: usize::from(*size),
            },
            yksmp::Location::Constant(v) => {
                // FIXME: This isn't fine-grained enough, as there may be constants of any
                // bit-size.
                let size = self.m.inst_no_copies(iidx).def_byte_size(self.m);
                match size {
                    4 => VarLocation::ConstInt {
                        bits: 32,
                        v: u64::from(*v),
                    },
                    _ => todo!(),
                }
            }
            e => {
                todo!("{:?}", e);
            }
        };
        let size = self.m.inst_no_copies(iidx).def_byte_size(self.m);
        debug_assert!(size <= REG64_SIZE);
        match m {
            VarLocation::Register(reg_alloc::Register::GP(reg)) => {
                self.ra.force_assign_inst_gp_reg(iidx, reg);
            }
            VarLocation::Register(reg_alloc::Register::FP(reg)) => {
                self.ra.force_assign_inst_fp_reg(iidx, reg);
            }
            VarLocation::Direct { frame_off, size: _ } => {
                self.ra.force_assign_inst_direct(iidx, frame_off);
            }
            VarLocation::Stack { frame_off, size: _ } => {
                self.ra
                    .force_assign_inst_indirect(iidx, i32::try_from(frame_off).unwrap());
            }
            VarLocation::ConstInt { bits, v } => {
                self.ra.assign_const(iidx, bits, v);
            }
            e => panic!("{:?}", e),
        }
    }

    fn cg_load(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadInst) {
        match self.m.type_(inst.tyidx()) {
            Ty::Integer(_) | Ty::Ptr => {
                let [reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(inst.operand(self.m))],
                );
                let size = self.m.inst_no_copies(iidx).def_byte_size(self.m);
                debug_assert!(size <= REG64_SIZE);
                match size {
                    1 => dynasm!(self.asm ; movzx Rq(reg.code()), BYTE [Rq(reg.code())]),
                    2 => dynasm!(self.asm ; movzx Rq(reg.code()), WORD [Rq(reg.code())]),
                    4 => dynasm!(self.asm ; mov Rd(reg.code()), [Rq(reg.code())]),
                    8 => dynasm!(self.asm ; mov Rq(reg.code()), [Rq(reg.code())]),
                    _ => todo!("{}", size),
                };
            }
            Ty::Float(fty) => {
                let [src_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(inst.operand(self.m))],
                );
                let [tgt_reg] =
                    self.ra
                        .assign_fp_regs(&mut self.asm, iidx, [RegConstraint::Output]);
                match fty {
                    FloatTy::Float => {
                        dynasm!(self.asm; movss Rx(tgt_reg.code()), [Rq(src_reg.code())])
                    }
                    FloatTy::Double => {
                        dynasm!(self.asm; movsd Rx(tgt_reg.code()), [Rq(src_reg.code())])
                    }
                }
            }
            x => todo!("{x:?}"),
        }
    }

    fn cg_ptradd(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::PtrAddInst) {
        // LLVM semantics dictate that the offset should be sign-extended/truncated up/down to the
        // size of the LLVM pointer index type. For address space zero on x86, truncation can't
        // happen, and when an immediate second operand is used for x86_64 `add`, it is implicitly
        // sign extended.
        let [reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(inst.ptr(self.m))],
        );

        dynasm!(self.asm ; add Rq(reg.code()), inst.off());
    }

    fn cg_dynptradd(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::DynPtrAddInst) {
        let [num_elems_reg, ptr_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                RegConstraint::InputOutput(inst.num_elems(self.m)),
                RegConstraint::Input(inst.ptr(self.m)),
            ],
        );

        // LLVM semantics dictate that the element size and number of elements should be
        // sign-extended/truncated up/down to the size of the LLVM pointer index type. For address
        // space zero on x86_64, truncation can't happen, and when an immediate third operand is
        // used for x86_64 `mul`, it is implicitly sign extended.
        dynasm!(self.asm
            // multiply the element size by the number of elements.
            ; imul Rq(num_elems_reg.code()), Rq(num_elems_reg.code()), i32::from(inst.elem_size())
            // add the result to the pointer. We make use of addition's commutative property to
            // reverse the "obvious" ordering of registers: doing so allows us not to overwrite
            // ptr_reg.
            ; add Rq(num_elems_reg.code()), Rq(ptr_reg.code())
        );
    }

    fn cg_store(&mut self, iidx: InstIdx, inst: &jit_ir::StoreInst) {
        let val = inst.val(self.m);
        match self.m.type_(val.tyidx(self.m)) {
            Ty::Integer(_) | Ty::Ptr => {
                let size = val.byte_size(self.m);
                let [tgt_reg, val_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::Input(inst.tgt(self.m)),
                        RegConstraint::Input(val),
                    ],
                );
                match size {
                    1 => dynasm!(self.asm ; mov [Rq(tgt_reg.code())], Rb(val_reg.code())),
                    2 => dynasm!(self.asm ; mov [Rq(tgt_reg.code())], Rw(val_reg.code())),
                    4 => dynasm!(self.asm ; mov [Rq(tgt_reg.code())], Rd(val_reg.code())),
                    8 => dynasm!(self.asm ; mov [Rq(tgt_reg.code())], Rq(val_reg.code())),
                    _ => todo!(),
                }
            }
            Ty::Float(fty) => {
                let [tgt_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(inst.tgt(self.m))],
                );
                let [val_reg] =
                    self.ra
                        .assign_fp_regs(&mut self.asm, iidx, [RegConstraint::Input(val)]);
                match fty {
                    FloatTy::Float => {
                        dynasm!(self.asm ; movss [Rq(tgt_reg.code())], Rx(val_reg.code()));
                    }
                    FloatTy::Double => {
                        dynasm!(self.asm ; movsd [Rq(tgt_reg.code())], Rx(val_reg.code()));
                    }
                }
            }
            Ty::Void | Ty::Func(_) => todo!(),
            Ty::Unimplemented(_) => todo!(),
        }
    }

    #[cfg(not(test))]
    fn cg_lookupglobal(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LookupGlobalInst) {
        let decl = inst.decl(self.m);
        if decl.is_threadlocal() {
            todo!();
        }
        let [tgt_reg] = self
            .ra
            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Output]);
        let sym_addr = self.m.globalvar_ptr(inst.global_decl_idx()).addr();
        dynasm!(self.asm ; mov Rq(tgt_reg.code()), QWORD i64::try_from(sym_addr).unwrap());
    }

    #[cfg(test)]
    fn cg_lookupglobal(&mut self, _inst_idx: jit_ir::InstIdx, _inst: &jit_ir::LookupGlobalInst) {
        panic!("Cannot lookup globals in cfg(test) as ykllvm will not have compiled this binary");
    }

    /// Codegen a call.
    fn cg_call(
        &mut self,
        iidx: InstIdx,
        inst: &jit_ir::DirectCallInst,
    ) -> Result<(), CompilationError> {
        let func_decl_idx = inst.target();
        let fty = self.m.func_type(func_decl_idx);
        let args = (0..(inst.num_args()))
            .map(|i| inst.operand(self.m, i))
            .collect::<Vec<_>>();

        // unwrap safe on account of linker symbol names not containing internal NULL bytes.
        let va = symbol_to_ptr(self.m.func_decl(func_decl_idx).name())
            .map_err(|e| CompilationError::General(e.to_string()))?;
        self.emit_call(iidx, fty, Some(va), None, &args)
    }

    /// Codegen a indirect call.
    fn cg_indirectcall(
        &mut self,
        iidx: InstIdx,
        indirect_call_idx: &IndirectCallIdx,
    ) -> Result<(), CompilationError> {
        let inst = self.m.indirect_call(*indirect_call_idx);
        let jit_ir::Ty::Func(fty) = self.m.type_(inst.ftyidx()) else {
            panic!()
        };
        let args = (0..(inst.num_args()))
            .map(|i| inst.operand(self.m, i))
            .collect::<Vec<_>>();
        self.emit_call(iidx, fty, None, Some(inst.target(self.m)), &args)
    }

    fn emit_call(
        &mut self,
        iidx: InstIdx,
        fty: &jit_ir::FuncTy,
        callee: Option<*const ()>,
        callee_op: Option<Operand>,
        args: &[Operand],
    ) -> Result<(), CompilationError> {
        // OPT: We clobber more than we need to.
        self.ra
            .clobber_gp_regs_hack(&mut self.asm, iidx, &CALLER_CLOBBER_REGS);
        self.ra.clobber_fp_regs(&mut self.asm, iidx);

        // Arrange arguments according to the ABI.
        let mut gp_regs = ARG_GP_REGS.iter();
        let mut fp_regs = ARG_FP_REGS.iter();
        let mut num_float_args = 0;
        for arg in args.iter() {
            match self.m.type_(arg.tyidx(self.m)) {
                Ty::Float(_) => {
                    let Some(reg) = fp_regs.next() else {
                        todo!("ran out of fp regs");
                    };
                    let [_] = self.ra.assign_fp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputIntoRegAndClobber(arg.clone(), *reg)],
                    );
                    num_float_args += 1;
                }
                _ => {
                    let Some(reg) = gp_regs.next() else {
                        todo!("ran out of gp regs");
                    };
                    let [_] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputIntoRegAndClobber(arg.clone(), *reg)],
                    );
                }
            }
        }
        // If the function we called has a return value, then store it into a local variable.
        //
        // FIXME: We only support up to register-sized return values at the moment.
        let ret_ty = fty.ret_type(self.m);
        #[cfg(debug_assertions)]
        if !matches!(ret_ty, Ty::Void) {
            debug_assert!(ret_ty.byte_size().unwrap() <= REG64_SIZE);
        }
        match ret_ty {
            Ty::Void => (),
            Ty::Float(_) => {
                let [_] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::OutputFromReg(Rx::XMM0)],
                );
            }
            Ty::Integer(_) | Ty::Ptr => {
                let [_] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::OutputFromReg(Rq::RAX)],
                );
            }
            Ty::Func(_) => todo!(),
            Ty::Unimplemented(_) => todo!(),
        }

        if fty.is_vararg() {
            // SysV x64 ABI says "rax is used to indicate the number of vector arguments passed
            // to a function requiring a variable number of arguments". Float arguments are passed
            // in vector registers.
            dynasm!(self.asm; mov rax, num_float_args);
        }

        // Actually perform the call.
        match (callee, callee_op) {
            (Some(p), None) => {
                let [reg] = self.ra.assign_gp_regs_avoiding(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Temporary],
                    RegSet::from_vec(&CALLER_CLOBBER_REGS),
                );
                dynasm!(self.asm
                    ; mov Rq(reg.code()), QWORD p as i64
                    ; call Rq(reg.code())
                );
            }
            (None, Some(op)) => {
                let [reg] = self.ra.assign_gp_regs_avoiding(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(op)],
                    RegSet::from_vec(&CALLER_CLOBBER_REGS),
                );
                dynasm!(self.asm; call Rq(reg.code()));
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn cg_icmp(&mut self, iidx: InstIdx, inst: &jit_ir::ICmpInst) {
        let (lhs, pred, rhs) = (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m));
        let size = lhs.byte_size(self.m);
        let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
        );
        match size {
            1 => dynasm!(self.asm; cmp Rb(lhs_reg.code()), Rb(rhs_reg.code())),
            2 => dynasm!(self.asm; cmp Rw(lhs_reg.code()), Rw(rhs_reg.code())),
            4 => dynasm!(self.asm; cmp Rd(lhs_reg.code()), Rd(rhs_reg.code())),
            8 => dynasm!(self.asm; cmp Rq(lhs_reg.code()), Rq(rhs_reg.code())),
            _ => todo!(),
        }

        // Interpret the flags assignment WRT the predicate.
        //
        // We use a SETcc instruction to do so.
        //
        // Remember, in Intel's tongue:
        //  - "above"/"below" -- unsigned predicate. e.g. `seta`.
        //  - "greater"/"less" -- signed predicate. e.g. `setle`.
        //
        //  Note that the equal/not-equal predicates are signedness agnostic.
        match pred {
            jit_ir::Predicate::Equal => dynasm!(self.asm; sete Rb(lhs_reg.code())),
            jit_ir::Predicate::NotEqual => dynasm!(self.asm; setne Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedGreater => dynasm!(self.asm; seta Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedGreaterEqual => dynasm!(self.asm; setae Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedLess => dynasm!(self.asm; setb Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedLessEqual => dynasm!(self.asm; setb Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedGreater => dynasm!(self.asm; setg Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedGreaterEqual => dynasm!(self.asm; setge Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedLess => dynasm!(self.asm; setl Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedLessEqual => dynasm!(self.asm; setle Rb(lhs_reg.code())),
        }
    }

    fn cg_fcmp(&mut self, iidx: InstIdx, inst: &jit_ir::FCmpInst) {
        let (lhs, pred, rhs) = (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m));
        let size = lhs.byte_size(self.m);
        let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Input(lhs), RegConstraint::Input(rhs)],
        );
        let [tgt_reg] = self
            .ra
            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Output]);

        match pred.is_ordered() {
            Some(true) => match size {
                4 => dynasm!(self.asm; comiss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                8 => dynasm!(self.asm; comisd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                _ => panic!(),
            },
            Some(false) => match size {
                4 => dynasm!(self.asm; ucomiss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                8 => dynasm!(self.asm; ucomisd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                _ => panic!(),
            },
            None => todo!(),
        }

        // Interpret the flags assignment WRT the predicate.
        //
        // Note that although floats are signed values, `{u,}comis{s,d}` sets CF (not SF and OF, as
        // you might expect). So when checking the outcome you have to use the "above" and "below"
        // variants of `setcc`, as if you were comparing unsigned integers.
        match pred {
            jit_ir::FloatPredicate::OrderedEqual | jit_ir::FloatPredicate::UnorderedEqual => {
                dynasm!(self.asm; sete Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::UnorderedNotEqual => {
                dynasm!(self.asm; setne Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::OrderedGreater => dynasm!(self.asm; seta Rb(tgt_reg.code())),
            jit_ir::FloatPredicate::OrderedGreaterEqual => {
                dynasm!(self.asm; setae Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::OrderedLess => dynasm!(self.asm; setb Rb(tgt_reg.code())),
            jit_ir::FloatPredicate::OrderedLessEqual => dynasm!(self.asm; setbe Rb(tgt_reg.code())),
            jit_ir::FloatPredicate::False
            | jit_ir::FloatPredicate::OrderedNotEqual
            | jit_ir::FloatPredicate::Ordered
            | jit_ir::FloatPredicate::Unordered
            | jit_ir::FloatPredicate::UnorderedGreater
            | jit_ir::FloatPredicate::UnorderedGreaterEqual
            | jit_ir::FloatPredicate::UnorderedLess
            | jit_ir::FloatPredicate::UnorderedLessEqual
            | jit_ir::FloatPredicate::True => todo!("{}", pred),
        }

        // But we have to be careful to check that the computation didn't produce "unordered". This
        // happens when at least one of the value compared was NaN. The unordered result is flagged
        // by PF (parity flag) being set.
        //
        // We follow the precedent set by clang and follow the IEE-754 spec with regards to
        // comparisons with NaN:
        //  - Any "not equal" comparison involving NaN is true.
        //  - All other comparisons are false.
        let [tmp_reg] = self
            .ra
            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Temporary]);
        match pred {
            jit_ir::FloatPredicate::OrderedNotEqual | jit_ir::FloatPredicate::UnorderedNotEqual => {
                dynasm!(self.asm
                    ; setp Rb(tmp_reg.code())
                    ; or Rb(tgt_reg.code()), Rb(tmp_reg.code())
                );
            }
            _ => {
                dynasm!(self.asm
                    ; setnp Rb(tmp_reg.code())
                    ; and Rb(tgt_reg.code()), Rb(tmp_reg.code())
                );
            }
        }
    }

    /// Move live values from their source location into the target location when doing a jump back
    /// to the beginning of a trace (or a jump from a side-trace to the beginning of its root
    /// trace).
    ///
    /// # Arguments
    ///
    /// * `tgt_vars` - The target locations. If `None` use `self.loop_start_locs` instead.
    fn write_jump_vars(&mut self, tgt_vars: Option<&[VarLocation]>) {
        // If we pass in `None` use `self.loop_start_locs` instead. We need to do this since we
        // can't pass in `&self.loop_start_locs` directly due to borrowing restrictions.
        let tgt_vars = tgt_vars.map_or(self.loop_start_locs.as_slice(), |v| v);
        for (i, op) in self.m.loop_jump_vars().iter().enumerate() {
            let (iidx, src) = match op {
                Operand::Var(iidx) => (*iidx, self.ra.var_location(*iidx)),
                _ => panic!(),
            };
            let dst = tgt_vars[i];
            if dst == src {
                // The value is already in the correct place, so there's nothing we need to
                // do.
                continue;
            }
            match dst {
                VarLocation::Stack {
                    frame_off: off_dst,
                    size: size_dst,
                } => {
                    match src {
                        VarLocation::Register(reg_alloc::Register::GP(reg)) => match size_dst {
                            8 => dynasm!(self.asm;
                                mov QWORD [rbp - i32::try_from(off_dst).unwrap()], Rq(reg.code())
                            ),
                            _ => todo!(),
                        },
                        VarLocation::ConstInt { bits, v } => match bits {
                            32 => dynasm!(self.asm;
                                mov DWORD [rbp - i32::try_from(off_dst).unwrap()], v as i32
                            ),
                            _ => todo!(),
                        },
                        VarLocation::Stack {
                            frame_off: off_src,
                            size: size_src,
                        } => match size_src {
                            // FIXME: Better to ask register allocator for a free register
                            // rather than pushing/popping RAX here?
                            8 => dynasm!(self.asm;
                                push rax;
                                mov rax, QWORD [rbp - i32::try_from(off_src).unwrap()];
                                mov QWORD [rbp - i32::try_from(off_dst).unwrap()], rax;
                                pop rax
                            ),
                            _ => todo!(),
                        },
                        e => todo!("{:?}", e),
                    }
                }
                VarLocation::Direct { .. } => {
                    // Direct locations are read-only, so it doesn't make sense to write to
                    // them. This is likely a case where the direct value has been moved
                    // somewhere else (register/normal stack) so dst and src no longer
                    // match. But since the value can't change we can safely ignore this.
                }
                VarLocation::Register(reg) => {
                    // Copy the value into a register. We can ask the register allocator to
                    // this for us, but telling it to load the source operand into the
                    // target register.
                    match reg {
                        reg_alloc::Register::GP(r) => {
                            let [_] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [RegConstraint::InputIntoReg(op.clone(), r)],
                            );
                        }
                        reg_alloc::Register::FP(r) => {
                            let [_] = self.ra.assign_fp_regs(
                                &mut self.asm,
                                iidx,
                                [RegConstraint::InputIntoReg(op.clone(), r)],
                            );
                        }
                    }
                }
                _ => todo!(),
            }
        }
    }

    fn cg_traceloopjump(&mut self) {
        // Loop the JITted code if the `tloop_start` label is present (not relevant for IR created
        // by a test or a side-trace).
        let label = StaticLabel::global("tloop_start");
        match self.asm.labels().resolve_static(&label) {
            Ok(_) => {
                // Found the label, emit a jump to it.
                self.write_jump_vars(None);
                dynasm!(self.asm; jmp ->tloop_start);
            }
            Err(DynasmError::UnknownLabel(_)) => {
                // Label not found. This is OK for unit testing, where we sometimes construct
                // traces that don't loop.
                #[cfg(test)]
                {
                    self.comment(self.asm.offset(), "Unterminated trace".to_owned());
                    dynasm!(self.asm; ud2);
                }
                // FIXME: Side traces don't currently loop back to the parent trace and deopt into
                // the main interpreter instead. Once we implement proper looping, this check can
                // be reinstated.
                // #[cfg(not(test))]
                // panic!("unterminated trace in non-unit-test");
            }
            Err(e) => {
                // Any other error suggests something has gone quite wrong. Just crash.
                panic!("{}", e.to_string())
            }
        }
    }

    fn cg_rootjump(&mut self, addr: *const libc::c_void) {
        // The end of a side-trace. Map live variables of this side-trace to the entry variables of
        // the root parent trace, then jump to it.
        self.write_jump_vars(Some(self.m.root_entry_vars()));
        self.ra.align_stack(SYSV_CALL_STACK_ALIGN);
        // Get the interpreter frame size, so we can subtract it from the current stack size (we
        // only want to reset things to the state before the root trace, but the stack size
        // includes the interpreter frame too).
        // FIXME: Pass this in rather than computing it. Once we have multiple control points this
        // will be incorrect.
        let (rec, pinfo) = AOT_STACKMAPS.as_ref().unwrap().get(0);
        let loopsize = if pinfo.hasfp {
            // The frame size includes the pushed RBP, but since we only care about the size of
            // the local variables we need to subtract it again.
            rec.size - u64::try_from(REG64_SIZE).unwrap()
        } else {
            rec.size
        };

        // The call to `__yk_reenter_jit" has the potential to clobber most of our registers, so
        // save and restore them here.
        dynasm!(self.asm
            ; push rax
            ; push rcx
            ; push rdx
            ; push rsi
            ; push rdi
            ; push r8
            ; push r9
            ; push r10
            ; push r11
        );

        for reg in lsregalloc::FP_REGS.iter() {
            dynasm!(self.asm
                ; movq rcx, Rx(reg.code())
                ; push rcx
            );
        }

        // Set the current executed trace in MT.
        #[allow(clippy::fn_to_numeric_cast)]
        {
            dynasm!(self.asm
                ; mov rdi, QWORD __yk_reenter_jit as i64
                ; call rdi
            );
        }

        // Restore saved registers.
        for reg in lsregalloc::FP_REGS.iter().rev() {
            dynasm!(self.asm
                ; pop rcx
                ; movq Rx(reg.code()), rcx
            );
        }
        dynasm!(self.asm
            ; pop r11
            ; pop r10
            ; pop r9
            ; pop r8
            ; pop rdi
            ; pop rsi
            ; pop rdx
            ; pop rcx
            ; pop rax
            ; add rsp, i32::try_from(self.ra.stack_size() - usize::try_from(loopsize).unwrap()).unwrap()
            ; mov rdi, QWORD addr as i64
            // We can safely use RDI here, since the root trace won't expect live variables in this
            // register since it's being used as an argument to the control point.
            ; jmp rdi);
    }

    fn cg_traceloopstart(&mut self) {
        debug_assert_eq!(self.loop_start_locs.len(), 0);
        // Remember the locations of the live variables at the beginning of the trace. When we loop
        // back around here we need to write the live variables back into these same locations.
        for var in self.m.loop_start_vars() {
            let loc = match var {
                Operand::Var(iidx) => self.ra.var_location(*iidx),
                _ => panic!(),
            };
            self.loop_start_locs.push(loc);
        }
        // FIXME: peel the initial iteration of the loop to allow us to hoist loop invariants.
        // When doing so, update the jump target inside side-traces.
        dynasm!(self.asm; ->tloop_start:);
    }

    fn cg_sext(&mut self, iidx: InstIdx, i: &jit_ir::SExtInst) {
        let [reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(i.val(self.m))],
        );

        let src_val = i.val(self.m);
        let src_type = self.m.type_(src_val.tyidx(self.m));
        let Ty::Integer(src_bitsize) = src_type else {
            unreachable!(); // must be an integer
        };

        let dest_type = self.m.type_(i.dest_tyidx());
        let Ty::Integer(dest_bitsize) = dest_type else {
            unreachable!(); // must be an integer
        };

        // FIXME: assumes the input and output fit in a register.
        match (src_bitsize, dest_bitsize) {
            (1, 64) => {
                // FIXME: find a way to efficiently generalise all the non-byte-sized extends.
                // Copy what LLVM does?
                dynasm!(self.asm
                    ; and Rb(reg.code()), 1
                    ; neg Rb(reg.code())
                    ; movsx Rq(reg.code()), Rb(reg.code()));
            }
            (8, 32) => dynasm!(self.asm; movsx Rd(reg.code()), Rb(reg.code())),
            (8, 64) => dynasm!(self.asm; movsx Rq(reg.code()), Rb(reg.code())),
            (16, 32) => dynasm!(self.asm; movsx Rd(reg.code()), Rw(reg.code())),
            (16, 64) => dynasm!(self.asm; movsx Rq(reg.code()), Rw(reg.code())),
            (32, 64) => dynasm!(self.asm; movsx Rq(reg.code()), Rd(reg.code())),
            _ => todo!("{} {}", src_bitsize, dest_bitsize),
        }
    }

    fn cg_zeroextend(&mut self, iidx: InstIdx, i: &jit_ir::ZeroExtendInst) {
        let [reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(i.val(self.m))],
        );

        let from_val = i.val(self.m);
        let from_type = self.m.type_(from_val.tyidx(self.m));
        let from_size = from_type.byte_size().unwrap();

        let to_type = self.m.type_(i.dest_tyidx());
        let to_size = to_type.byte_size().unwrap();

        // Note that the src/dest types may be pointers for cases where non-truncating
        // inttoptr/ptrtoint are serialised to zero-extends by ykllvm.
        debug_assert!(
            matches!(to_type, jit_ir::Ty::Integer(_)) || matches!(to_type, jit_ir::Ty::Ptr)
        );
        debug_assert!(
            matches!(from_type, jit_ir::Ty::Integer(_)) || matches!(from_type, jit_ir::Ty::Ptr)
        );
        // You can only zero-extend a smaller integer to a larger integer.
        debug_assert!(from_size <= to_size);

        // FIXME: assumes the input and output fit in a register.
        debug_assert!(to_size <= REG64_SIZE);

        // FIXME: Assumes we don't assign to sub-registers.
        match (to_size, from_size) {
            (4, 1) => dynasm!(self.asm; movzx Rq(reg.code()), Rb(reg.code())),
            (4, 2) => dynasm!(self.asm; movzx Rq(reg.code()), Rw(reg.code())),
            (8, 1) => dynasm!(self.asm; movzx Rq(reg.code()), Rb(reg.code())),
            (8, 4) => (), // 32-bit regs are already zero-extended
            (8, 8) => (), // Nothing to extend
            _ => todo!("{to_size} {from_size}"),
        }
    }

    fn cg_sitofp(&mut self, iidx: InstIdx, inst: &jit_ir::SIToFPInst) {
        let [src_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Input(inst.val(self.m))],
        );
        let [tgt_reg] = self
            .ra
            .assign_fp_regs(&mut self.asm, iidx, [RegConstraint::Output]);

        let src_size = inst.val(self.m).byte_size(self.m);
        match self.m.type_(inst.dest_tyidx()) {
            jit_ir::Ty::Float(jit_ir::FloatTy::Float) => {
                debug_assert_eq!(src_size, 4);
                dynasm!(self.asm; cvtsi2ss Rx(tgt_reg.code()), Rd(src_reg.code()));
            }
            jit_ir::Ty::Float(jit_ir::FloatTy::Double) => match src_size {
                4 => dynasm!(self.asm; cvtsi2sd Rx(tgt_reg.code()), Rd(src_reg.code())),
                8 => dynasm!(self.asm; cvtsi2sd Rx(tgt_reg.code()), Rq(src_reg.code())),
                _ => todo!(),
            },
            _ => panic!(),
        }
    }

    fn cg_fptosi(&mut self, iidx: InstIdx, inst: &jit_ir::FPToSIInst) {
        let from_val = inst.val(self.m);
        let to_ty = self.m.type_(inst.dest_tyidx());
        // Unwrap cannot fail: floats and integers are sized.
        let from_size = self.m.type_(from_val.tyidx(self.m)).byte_size().unwrap();
        let to_size = to_ty.byte_size().unwrap();

        let [src_reg] =
            self.ra
                .assign_fp_regs(&mut self.asm, iidx, [RegConstraint::Input(from_val)]);
        let [tgt_reg] = self
            .ra
            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Output]);

        match from_size {
            4 => dynasm!(self.asm; cvttss2si Rq(tgt_reg.code()), Rx(src_reg.code())),
            8 => dynasm!(self.asm; cvttsd2si Rq(tgt_reg.code()), Rx(src_reg.code())),
            _ => panic!(),
        }

        // Now we have a (potentially rounded) 64-bit integer in a register.
        //
        // If the integer type we are casting to is smaller than or of equal size to `from_size`
        // then we don't need to do anything else because either:
        //
        // a) the desired value fits in `to_size` bytes and, due to two's compliment, you can
        //    truncate away higher-order bytes and have the same numeric integer value, or
        //
        // b) the desired value doesn't fit in `to_size`, which is UB and we can do anything.
        //
        // FIXME: If however, we are casting to a larger-sized integer type, we will need to sign
        // extend the value to keep the same numeric value.
        if to_size > from_size {
            todo!("fptosi requires sign extend: {} -> {}", from_size, to_size);
        }
    }

    fn cg_fpext(&mut self, iidx: InstIdx, i: &jit_ir::FPExtInst) {
        let from_val = i.val(self.m);
        let from_type = self.m.type_(from_val.tyidx(self.m));
        let to_type = self.m.type_(i.dest_tyidx());

        let [tgt_reg] =
            self.ra
                .assign_fp_regs(&mut self.asm, iidx, [RegConstraint::InputOutput(from_val)]);

        match (from_type, to_type) {
            (
                jit_ir::Ty::Float(jit_ir::FloatTy::Float),
                jit_ir::Ty::Float(jit_ir::FloatTy::Double),
            ) => dynasm!(self.asm; cvtss2sd Rx(tgt_reg.code()), Rx(tgt_reg.code())),
            _ => {
                // since we only support 32- and 64-bit floats, any other conversion is
                // nonsensical.
                panic!();
            }
        }
    }

    fn cg_trunc(&mut self, iidx: InstIdx, i: &jit_ir::TruncInst) {
        let [_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(i.val(self.m))],
        );

        let from_val = i.val(self.m);
        let from_type = self.m.type_(from_val.tyidx(self.m));
        let from_size = from_type.byte_size().unwrap();

        let to_type = self.m.type_(i.dest_tyidx());
        let to_size = to_type.byte_size().unwrap();

        debug_assert!(matches!(to_type, jit_ir::Ty::Integer(_)));
        debug_assert!(
            matches!(from_type, jit_ir::Ty::Integer(_)) || matches!(from_type, jit_ir::Ty::Ptr)
        );
        // You can only truncate a bigger integer to a smaller integer.
        debug_assert!(from_size > to_size);

        // FIXME: assumes the input and output fit in a register.
        debug_assert!(to_size <= REG64_SIZE);

        // FIXME: There's no instruction on x86_64 to mov from a bigger register into a smaller
        // register. The simplest way to truncate the value is to zero out the higher order bits.
        // Currently the AOT IR follows `trunc` by `and` to do this for us, but relying on that
        // feels rather fragile.
    }

    fn cg_select(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::SelectInst) {
        // First load the true case. We then immediately follow this up with a conditional move,
        // overwriting the value with the false case, if the condition was false.
        let [true_reg, cond_reg, false_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                RegConstraint::InputOutput(inst.trueval(self.m)),
                RegConstraint::Input(inst.cond(self.m)),
                RegConstraint::Input(inst.falseval(self.m)),
            ],
        );
        dynasm!(self.asm ; cmp Rb(cond_reg.code()), 0);
        dynasm!(self.asm ; cmove Rq(true_reg.code()), Rq(false_reg.code()));
    }

    fn cg_guard(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::GuardInst) {
        // Convert the guard info into deopt info and store it on the heap.
        let mut lives = Vec::new();
        let gi = inst.guard_info(self.m);
        for (iid, pop) in gi.live_vars() {
            match pop.unpack(self.m) {
                Operand::Var(x) => {
                    lives.push((iid.clone(), self.ra.var_location(x)));
                }
                Operand::Const(x) => {
                    // The live variable is a constant (e.g. this can happen during inlining), so
                    // it doesn't have an allocation. We can just push the actual value instead
                    // which will be written as is during deoptimisation.
                    match self.m.const_(x) {
                        Const::Int(tyidx, c) => {
                            let Ty::Integer(bits) = self.m.type_(*tyidx) else {
                                panic!()
                            };
                            lives.push((iid.clone(), VarLocation::ConstInt { bits: *bits, v: *c }))
                        }
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

        let fail_label = self.asm.new_dynamic_label();
        // FIXME: Move `frames` instead of copying them (requires JIT module to be consumable).
        let deoptinfo = DeoptInfo {
            bid: gi.bid().clone(),
            fail_label,
            live_vars: lives,
            inlined_frames: gi.inlined_frames().to_vec(),
            guard: Guard::new(),
        };
        self.deoptinfo.insert(inst.gidx.into(), deoptinfo);

        let cond = inst.cond(self.m);
        // ICmp instructions evaluate to a one-byte zero/one value.
        debug_assert_eq!(cond.byte_size(self.m), 1);
        let [reg] = self
            .ra
            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Input(cond)]);
        dynasm!(self.asm
            ; cmp Rb(reg.code()), inst.expect() as i8 // `as` intentional.
            ; jne =>fail_label
        );
    }
}

/// Information required by deoptimisation.
#[derive(Debug)]
struct DeoptInfo {
    /// The AOT block that the failing guard originated from.
    bid: aot_ir::BBlockId,
    fail_label: DynamicLabel,
    /// Live variables, mapping AOT vars to JIT vars.
    live_vars: Vec<(aot_ir::InstID, VarLocation)>,
    inlined_frames: Vec<InlinedFrame>,
    /// Keeps track of deopt amount and compiled side-trace.
    guard: Guard,
}

#[derive(Debug)]
pub(super) struct X64CompiledTrace {
    // Reference to the meta-tracer required for side tracing.
    mt: Arc<MT>,
    /// The executable code itself.
    buf: ExecutableBuffer,
    /// Deoptimisation info: maps a [GuardIdx] to [DeoptInfo].
    deoptinfo: HashMap<usize, DeoptInfo>,
    /// Stack pointer offset from the base pointer of interpreter frame as defined in
    /// [YkSideTraceInfo::sp_offset].
    sp_offset: usize,
    /// The locations of live variables this trace expects to exist upon entering. These are the
    /// live variables at the time of the control point.
    entry_vars: Vec<VarLocation>,
    /// Reference to the HotLocation, required for side tracing.
    hl: Weak<Mutex<HotLocation>>,
    /// Comments to be shown when printing the compiled trace using `AsmPrinter`.
    ///
    /// Maps a byte offset in the native JITted code to a collection of line comments to show when
    /// disassembling the trace.
    ///
    /// Used for testing and debugging.
    comments: IndexMap<usize, Vec<String>>,
    #[cfg(any(debug_assertions, test))]
    gdb_ctx: GdbCtx,
}

impl X64CompiledTrace {
    /// Return the locations of the live variables this trace expects as input.
    fn entry_vars(&self) -> &[VarLocation] {
        &self.entry_vars
    }
}

impl CompiledTrace for X64CompiledTrace {
    fn entry(&self) -> *const libc::c_void {
        self.buf.ptr(AssemblyOffset(0)) as *const libc::c_void
    }

    fn sidetraceinfo(&self, gidx: GuardIdx) -> Arc<dyn SideTraceInfo> {
        // FIXME: Can we reference these instead of copying them, e.g. by passing in a reference to
        // the `CompiledTrace` and `gidx` or better a reference to `DeoptInfo`?
        let deoptinfo = &self.deoptinfo[&usize::from(gidx)];
        let lives = deoptinfo
            .live_vars
            .iter()
            .map(|(a, l)| (a.clone(), l.into()))
            .collect();
        let callframes = deoptinfo.inlined_frames.clone();

        // Get the root trace from the HotLocation.
        // FIXME: It might be better if we had a consistent mechanism for doing this rather than
        // fishing it out of the HotLocationKind`.
        let hlarc = self.hl.upgrade().unwrap();
        let hl = hlarc.lock();
        let root_ctr = match &hl.kind {
            HotLocationKind::Compiled(root_ctr) => Arc::clone(root_ctr)
                .as_any()
                .downcast::<X64CompiledTrace>()
                .unwrap(),
            HotLocationKind::SideTracing { root_ctr, .. } => Arc::clone(root_ctr)
                .as_any()
                .downcast::<X64CompiledTrace>()
                .unwrap(),
            _ => panic!("Unexpected HotLocationKind"),
        };

        Arc::new(YkSideTraceInfo {
            bid: deoptinfo.bid.clone(),
            lives,
            callframes,
            root_addr: RootTracePtr(root_ctr.entry()),
            entry_vars: root_ctr.entry_vars().to_vec(),
            sp_offset: self.sp_offset,
        })
    }

    fn guard(&self, gidx: GuardIdx) -> &crate::compile::Guard {
        &self.deoptinfo[&usize::from(gidx)].guard
    }

    fn hl(&self) -> &std::sync::Weak<parking_lot::Mutex<crate::location::HotLocation>> {
        &self.hl
    }

    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
        self
    }

    fn disassemble(&self) -> Result<String, Box<dyn Error>> {
        AsmPrinter::new(&self.buf, &self.comments).to_string()
    }
}

/// Disassembles emitted code for testing and debugging purposes.
struct AsmPrinter<'a> {
    buf: &'a ExecutableBuffer,
    comments: &'a IndexMap<usize, Vec<String>>,
}

impl<'a> AsmPrinter<'a> {
    fn new(buf: &'a ExecutableBuffer, comments: &'a IndexMap<usize, Vec<String>>) -> Self {
        Self { buf, comments }
    }

    /// Returns the disassembled trace.
    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        let mut out = Vec::new();
        let len = self.buf.len();
        let bptr = self.buf.ptr(AssemblyOffset(0));
        let code = unsafe { slice::from_raw_parts(bptr, len) };
        let fmt = zydis::Formatter::intel();
        let dec = zydis::Decoder::new64();
        for insn_info in dec.decode_all::<zydis::VisibleOperands>(code, 0) {
            let (off, _raw_bytes, insn) = insn_info.unwrap();
            if let Some(lines) = self.comments.get(&usize::try_from(off).unwrap()) {
                for line in lines {
                    out.push(format!("; {line}"));
                }
            }
            let istr = fmt.format(Some(off), &insn).unwrap();
            out.push(format!(
                "{:016x} {:08x}: {}",
                (bptr as u64) + off,
                off,
                istr
            ));
        }
        Ok(out.join("\n"))
    }
}

/// x64 tests. These use an unusual form of pattern matching. Instead of using concrete register
/// names, one can refer to a class of registers e.g. `r.8` is all 8-bit registers. To match a
/// register name but ignore its value uses `r.8._`: to match a register name use `r.8.x`.
///
/// Suffixes: must be unique (i.e. `r.8.x` and `r.8.y` must refer to different registers); and they
/// refer to the "underlying" register (e.g. `r.8.x` and `r.64.x` might match `al` and `RAX` which
/// is considered the same register, but will fail to match against `al` and `RBX`).
///
/// Note that general purpose (`r.`) and floating point (`fp.`) registers occupy different suffix
/// classes (i.e. `r.8.x` and `fp.128.x` do not match the same register "x").
#[cfg(test)]
mod tests {
    use super::{Assemble, X64CompiledTrace};
    use crate::compile::{
        jitc_yk::jit_ir::{self, Module},
        CompiledTrace,
    };
    use crate::location::{HotLocation, HotLocationKind};
    use crate::mt::MT;
    use fm::{FMBuilder, FMatcher};
    use lazy_static::lazy_static;
    use parking_lot::Mutex;
    use regex::{Regex, RegexBuilder};
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };
    use ykaddr::addr::symbol_to_ptr;

    /// All x64 registers sorted by class. Later we allow users to match all (e.g.) 8 bit registers with `r.8._`.
    const X64_REGS: [(&str, &str); 4] = [
        (
            "8",
            "al|bl|cl|dl|sil|dil|spl|bpl|r8b|r9b|r10b|r11b|r12b|r13b|r14b|r15b|ah|bh|ch|dh",
        ),
        (
            "16",
            "ax|bx|cx|dx|si|di|sp|bp|r8w|r9w|r10w|r11w|r12w|r13w|r14w|r15w",
        ),
        (
            "32",
            "eax|ebx|ecx|edx|esi|edi|esp|ebp|r8d|r9d|r10d|r11d|r12d|r13d|r14d|r15d",
        ),
        (
            "64",
            "rax|rbx|rcx|rdx|rsi|rdi|rsp|rbp|r8|r9|r10|r11|r12|r13|r14|r15",
        ),
    ];

    const X64_RAW_REGS_MAP: [[&str; 5]; 16] = [
        ["rax", "eax", "ax", "ah", "al"],
        ["rbx", "ebx", "bx", "bh", "bl"],
        ["rcx", "ecx", "cx", "ch", "cl"],
        ["rdx", "edx", "dx", "dh", "dl"],
        ["rsi", "esi", "si", "", "sil"],
        ["rdi", "edi", "di", "", "dil"],
        ["rsp", "esp", "sp", "", "spl"],
        ["rbp", "ebp", "bp", "", "bpl"],
        ["r8", "r8d", "r8w", "", "r8b"],
        ["r9", "r9d", "r9w", "", "r9b"],
        ["r10", "r10d", "r10w", "", "r10b"],
        ["r11", "r11d", "r11w", "", "r11b"],
        ["r12", "r12d", "r12w", "", "r12b"],
        ["r13", "r13d", "r13w", "", "r13b"],
        ["r14", "r14d", "r14w", "", "r14b"],
        ["r15", "r15d", "r15w", "", "r15b"],
    ];

    lazy_static! {
        static ref X64_REGS_MAP: HashMap<&'static str, usize> = {
            let mut map = HashMap::new();
            for (i, regs) in X64_RAW_REGS_MAP.iter().enumerate() {
                for r in regs.iter() {
                    if *r != "" {
                        map.insert(*r, i);
                    }
                }
            }
            map
        };

        /// Use `{{name}}` to match non-literal strings in tests.
        static ref PTN_RE: Regex = {
            Regex::new(r"\{\{.+?\}\}").unwrap()
        };

        static ref PTN_RE_IGNORE: Regex = {
            Regex::new(r"\{\{_}\}").unwrap()
        };

        static ref FP_REG_NAME_RE: Regex = {
            Regex::new(r"fp\.128\.[0-9a-z]+").unwrap()
        };

        static ref FP_REG_TEXT_RE: Regex = {
            Regex::new(r"xmm[0-9][0-5]?").unwrap()
        };

        static ref TEXT_RE: Regex = {
            Regex::new(r"[a-zA-Z0-9\._]+").unwrap()
        };
    }

    fn fmatcher<'a>(ptn: &'a str) -> FMatcher<'a> {
        let mut fmb = FMBuilder::new(ptn)
            .unwrap()
            .name_matching_validator(|names| {
                let mut gp_reg_vals = vec![None; 16];
                let mut fp_reg_vals = HashSet::new();
                let mut fp_reg_count = 0;
                let mut result = true;
                for (hl_name, reg) in names.iter() {
                    if hl_name.starts_with("r.") {
                        let hl_name = hl_name.split('.').nth(2).unwrap();
                        let reg_i = X64_REGS_MAP[reg];
                        if let Some(x) = gp_reg_vals[reg_i] {
                            if x != hl_name {
                                result = false;
                                break;
                            }
                        } else {
                            gp_reg_vals[reg_i] = Some(hl_name);
                        }
                    } else if hl_name.starts_with("fp.") {
                        fp_reg_vals.insert(reg);
                        fp_reg_count += 1;
                        if fp_reg_vals.len() != fp_reg_count {
                            result = false;
                            break;
                        }
                    }
                }
                result
            })
            .name_matcher_ignore(PTN_RE_IGNORE.clone(), TEXT_RE.clone())
            .name_matcher(PTN_RE.clone(), TEXT_RE.clone())
            .name_matcher(FP_REG_NAME_RE.clone(), FP_REG_TEXT_RE.clone());

        for (class_name, regs) in X64_REGS {
            let class_re = RegexBuilder::new(&format!("r\\.{class_name}\\.[0-9a-z]+"))
                .case_insensitive(true)
                .build()
                .unwrap();
            let class_ignore_re = RegexBuilder::new(&format!("r\\.{class_name}\\._"))
                .case_insensitive(true)
                .build()
                .unwrap();
            let regs_re = RegexBuilder::new(regs)
                .case_insensitive(true)
                .build()
                .unwrap();
            fmb = fmb
                .name_matcher_ignore(class_ignore_re, regs_re.clone())
                .name_matcher(class_re, regs_re);
        }

        fmb.build().unwrap()
    }

    #[test]
    fn check_matching() {
        let fmm = fmatcher("r.8.x r.8.y r.64.x");
        assert!(fmm.matches("al bl rax").is_ok());
        assert!(fmm.matches("al al rax").is_err());
        assert!(fmm.matches("al bl rbx").is_err());
        assert!(fmm.matches("al bl eax").is_err());

        let fmm = fmatcher("r.8.x r.8._ r.64.x");
        assert!(fmm.matches("al bl rax").is_ok());
        assert!(fmm.matches("al al rax").is_ok());

        let fmm = fmatcher("fp.128.x fp.128.y fp.128.x");
        assert!(fmm.matches("xmm0 xmm1 xmm0").is_ok());
        assert!(fmm.matches("xmm0 xmm0 xmm0").is_err());
    }

    fn test_module() -> jit_ir::Module {
        jit_ir::Module::new_testing()
    }

    /// Test helper to use `fm` to match a disassembled trace.
    fn match_asm(cgo: Arc<X64CompiledTrace>, ptn: &str) {
        let dis = cgo.disassemble().unwrap();

        // The disassembler alternates between upper- and lowercase hex, making matching addresses
        // difficult. So just lowercase both pattern and text to avoid tests randomly breaking when
        // addresses change.
        let ptn = ptn.to_lowercase();
        match fmatcher(&ptn).matches(&dis.to_lowercase()) {
            Ok(()) => (),
            Err(e) => panic!("{e}"),
        }
    }

    fn codegen_and_test(mod_str: &str, patt_lines: &str) {
        let m = Module::from_str(mod_str);
        let mt = MT::new().unwrap();
        let hl = HotLocation {
            kind: HotLocationKind::Tracing,
            tracecompilation_errors: 0,
        };
        match_asm(
            Assemble::new(&m, None)
                .unwrap()
                .codegen(mt, Arc::new(Mutex::new(hl)), false)
                .unwrap()
                .as_any()
                .downcast::<X64CompiledTrace>()
                .unwrap(),
            patt_lines,
        );
    }

    #[test]
    fn cg_load_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = load_ti 0
                %1: ptr = load %0
            ",
            "
                ...
                ; %1: ptr = load %0
                {{_}} {{_}}: mov [rbp-0x08], r.64.x
                {{_}} {{_}}: mov r.64.x, [r.64.x]
                ...
                ",
        );
    }

    #[test]
    fn cg_load_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load %0
            ",
            "
                ...
                ; %1: i8 = load %0
                {{_}} {{_}}: mov [rbp-0x01], r.8.x
                {{_}} {{_}}: movzx r.64.x, byte ptr [r.64.x]
                ...
                ",
        );
    }

    #[test]
    fn cg_load_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = load %0
            ",
            "
                ...
                ; %1: i32 = Load %0
                {{_}} {{_}}: mov [rbp-0x04], r.32.x
                {{_}} {{_}}: mov r.32.x, [r.64.x]
                ...
                ",
        );
    }

    #[test]
    fn cg_load_const_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = load_ti 0
                *%0 = 0x0
            ",
            "
                ...
                ; *%0 = 0x0
                {{_}} {{_}}: mov r.64.x, 0x00
                {{_}} {{_}}: mov [r.64.y], r.64.x
                ...
                ",
        );
    }

    #[test]
    fn cg_ptradd() {
        codegen_and_test(
            "
              entry:
                %0: ptr = load_ti 0
                %1: i32 = ptr_add %0, 64
            ",
            "
                ...
                ; %1: ptr = ptr_add %0, 64
                {{_}} {{_}}: mov ...
                {{_}} {{_}}: add r.64.x, 0x40
                ...
                ",
        );
    }

    #[test]
    fn cg_dynptradd() {
        codegen_and_test(
            "
              entry:
                %0: ptr = load_ti 0
                %1: i32 = load_ti 1
                %2: ptr = dyn_ptr_add %0, %1, 32
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 32
                {{_}} {{_}}: mov [rbp-{{_}}], r.32.x
                {{_}} {{_}}: imul r.64.x, r.64.x, 0x20
                {{_}} {{_}}: add r.64.x, r.64.y
                ...
                ",
        );
    }

    #[test]
    fn cg_store_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = load_ti 0
                %1: ptr = load_ti 1
                *%1 = %0
            ",
            "
                ...
                ; %0: ptr = load_ti ...
                ; %1: ptr = load_ti ...
                ; *%1 = %0
                {{_}} {{_}}: mov [r.64.x], r.64.y
                ...
                ",
        );
    }

    #[test]
    fn cg_add_i16() {
        codegen_and_test(
            "
              entry:
                %0: i16 = load_ti 0
                %1: i16 = load_ti 1
                %3: i16 = add %0, %1
            ",
            "
                ...
                ; %2: i16 = add %0, %1
                ......
                {{_}} {{_}}: add r.16.x, r.16.y
                ...
                ",
        );
    }

    #[test]
    fn cg_add_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i64 = load_ti 1
                %3: i64 = add %0, %1
            ",
            "
                ...
                ; %2: i64 = add %0, %1
                ......
                {{_}} {{_}}: add r.64.x, r.64.y
                ...
                ",
        );
    }

    #[test]
    fn cg_const_add_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i64 = add %0, 1i64
            ",
            "
                ...
                ; %1: i64 = add %0, 1i64
                ......
                {{_}} {{_}}: add r.64.x, 0x01
                ...
                ",
        );
    }

    #[test]
    fn cg_const_add_minus_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i64 = add %0, 18446744073709551615i64
            ",
            "
                ...
                ; %1: i64 = add %0, 18446744073709551615i64
                ......
                {{_}} {{_}}: add r.64.x, 0xffffffffffffffff
                ...
                ",
        );
        // note: disassembler sign-extended the immediate when displaying it.
    }

    #[test]
    fn cg_const_add_i32max_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i64 = add %0, 2147483647i64
            ",
            "
                ...
                ; %1: i64 = add %0, 2147483647i64
                ......
                {{_}} {{_}}: add r.64.x, 0x7fffffff
                ...
                ",
        );
    }

    #[test]
    fn cg_const_add_i32max_plus_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i64 = add %0, 2147483648i64
            ",
            "
                ...
                ; %1: i64 = add %0, 2147483648i64
                ......
                {{_}} {{_}}: mov r.64.x, 0x80000000
                {{_}} {{_}}: add r.64.y, r.64.x
                ...
                ",
        );
    }

    #[test]
    fn cg_const_add_one_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = add %0, 1i32
            ",
            "
                ...
                ; %1: i32 = add %0, 1i32
                ......
                {{_}} {{_}}: add r.32.x, 0x01
                ...
                ",
        );
    }

    #[test]
    fn cg_const_add_minus_one_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = add %0, 4294967295i32
            ",
            "
                ...
                ; %1: i32 = add %0, 4294967295i32
                ......
                {{_}} {{_}}: add r.32.x, 0xffffffff
                ...
                ",
        );
    }

    #[test]
    fn cg_call_simple() {
        let sym_addr = symbol_to_ptr("puts").unwrap().addr();
        codegen_and_test(
            "
              func_decl puts ()

              entry:
                call @puts()
            ",
            &format!(
                "
                ...
                ; call @puts()
                {{{{_}}}} {{{{_}}}}: mov r.64.x, 0x{sym_addr:X}
                {{{{_}}}} {{{{_}}}}: call r.64.x
                ...
            "
            ),
        );
    }

    #[test]
    fn cg_call_with_args() {
        let sym_addr = symbol_to_ptr("puts").unwrap().addr();
        codegen_and_test(
            "
              func_decl puts (i32, i32, i32)

              entry:
                %0: i32 = load_ti 0
                %1: i32 = load_ti 1
                %2: i32 = load_ti 2
                call @puts(%0, %1, %2)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2)
                ...
                {{{{_}}}} {{{{_}}}}: mov edi, ...
                {{{{_}}}} {{{{_}}}}: mov esi, ...
                {{{{_}}}} {{{{_}}}}: mov edx, ...
                {{{{_}}}} {{{{_}}}}: mov r.64.tgt, 0x{sym_addr:X}
                {{{{_}}}} {{{{_}}}}: call r.64.tgt
                ...
            "
            ),
        );
    }

    #[test]
    fn cg_call_with_different_args() {
        let sym_addr = symbol_to_ptr("puts").unwrap().addr();
        codegen_and_test(
            "
              func_decl puts (i8, i16, ptr)

              entry:
                %0: i8 = load_ti 0
                %1: i16 = load_ti 1
                %2: ptr = load_ti 2
                call @puts(%0, %1, %2)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2)
                ...
                {{{{_}}}} {{{{_}}}}: movzx rdi, ...
                {{{{_}}}} {{{{_}}}}: movzx rsi, ...
                {{{{_}}}} {{{{_}}}}: mov rdx, ...
                {{{{_}}}} {{{{_}}}}: mov r.64.tgt, 0x{sym_addr:X}
                {{{{_}}}} {{{{_}}}}: call r.64.tgt
                ...
            "
            ),
        );
    }

    #[should_panic] // until we implement spill args
    #[test]
    fn cg_call_spill_args() {
        codegen_and_test(
            "
              func_decl f(...)
              entry:
                %1: i32 = call @f(0, 1, 2, 3, 4, 5, 6, 7)
            ",
            "",
        );
    }

    #[test]
    fn cg_call_ret() {
        codegen_and_test(
            "
             func_decl puts() -> i32
             entry:
               %0: i32 = call @puts()
               %1: i32 = add %0, %0
            ",
            "
                ...
                ; %0: i32 = call @puts()
                {{_}} {{_}}: mov r.64.x, ...
                {{_}} {{_}}: call r.64.x
                ; %1: i32 = add %0, %0
                {{_}} {{_}}: mov [rbp-0x04], eax
                {{_}} {{_}}: mov r.32.x, [rbp-0x04]
                {{_}} {{_}}: add eax, r.32.x
                ...
            ",
        );
    }

    #[test]
    fn cg_eq_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = load_ti 0
                %1: i1 = eq %0, %0
            ",
            "
                ...
                ; %1: i1 = eq %0, %0
                {{_}} {{_}}: mov [rbp-{{0x08}}], r.64.x
                {{_}} {{_}}: mov r.64.y, [rbp-{{0x08}}]
                {{_}} {{_}}: cmp r.64.x, r.64.y
                {{_}} {{_}}: setz r.8.x
                ...
            ",
        );
    }

    #[test]
    fn cg_eq_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i1 = eq %0, %0
            ",
            "
                ...
                ; %1: i1 = eq %0, %0
                {{_}} {{_}}: mov [rbp-{{0x01}}], r.8.x
                {{_}} {{_}}: movzx r.64.y, byte ptr [rbp-{{0x01}}]
                {{_}} {{_}}: cmp r.8.x, r.8.y
                {{_}} {{_}}: setz r.8.x
                ...
            ",
        );
    }

    #[test]
    fn cg_guard_true() {
        codegen_and_test(
            "
              entry:
                %0: i1 = load_ti 0
                guard true, %0, []
            ",
            "
                ...
                ; guard true, %0, []
                {{_}} {{_}}: cmp r.8.b, 0x01
                {{_}} {{_}}: jnz 0x...
                ...
                ; deopt id for guard 0
                {{_}} {{_}}: push rsi
                ... mov rsi, 0x00
                ... jmp ...
                ; call __yk_deopt
                ...
                ... mov rdi, rbp
                ... mov rax, 0x...
                ... sub rsp, 0x08
                ... call rax
            ",
        );
    }

    #[test]
    fn cg_guard_false() {
        codegen_and_test(
            "
              entry:
                %0: i1 = load_ti 0
                guard false, %0, []
            ",
            "
                ...
                ; guard false, %0, []
                {{_}} {{_}}: cmp r.8.b, 0x00
                {{_}} {{_}}: jnz 0x...
                ...
                ; deopt id for guard 0
                {{_}} {{_}}: push rsi
                ... mov rsi, 0x00
                ... jmp ...
                ; call __yk_deopt
                ...
                ... mov rdi, rbp
                ... mov rax, 0x...
                ... sub rsp, 0x08
                ... call rax
            ",
        );
    }

    #[test]
    fn cg_guard_const() {
        codegen_and_test(
            "
              entry:
                %0: i1 = load_ti 0
                %1: i8 = 10i8
                %2: i8 = 32i8
                %3: i8 = add %1, %2
                guard false, %0, [%0, 10i8, 32i8, 42i8]
            ",
            "
                ...
                ; guard false, %0, [0:%0_0: %0, 0:%0_1: 10i8, 0:%0_2: 32i8, 0:%0_3: 42i8]
                {{_}} {{_}}: cmp r.8.b, 0x00
                {{_}} {{_}}: jnz 0x...
                ...
                ; deopt id for guard 0
                {{_}} {{_}}: push rsi
                ... mov rsi, 0x00
                ... jmp ...
                ; call __yk_deopt
                ...
                ... mov rdi, rbp
                ... mov rax, 0x...
                ... sub rsp, 0x08
                ... call rax
            ",
        );
    }

    #[test]
    fn unterminated_trace() {
        codegen_and_test(
            "
              entry:
                 tloop_jump []
                ",
            "
                ...
                {{_}} {{_}}: ud2
                ",
        );
    }

    #[test]
    fn looped_trace_smallest() {
        // FIXME: make the offset and disassembler format hex the same so we can match
        // easier (capitalisation of hex differs).
        codegen_and_test(
            "
              entry:
                tloop_start []
                tloop_jump []
            ",
            "
                ...
                ; tloop_start []:
                ; tloop_jump []:
                {{_}} {{_}}: jmp {{target}}
            ",
        );
    }

    #[test]
    fn looped_trace_bigger() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                tloop_start [%0]
                %2: i8 = add %0, %0
                tloop_jump [%2]
            ",
            "
                ...
                ; %0: i8 = load_ti ...
                ...
                ; tloop_start [%0]:
                ; %2: i8 = add %0, %0
                ...
                ; tloop_jump [%2]:
                ...
                ...: jmp ...
            ",
        );
    }

    #[test]
    fn cg_srem() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = load_ti 1
                %2: i32 = srem %0, %1
            ",
            "
                ...
                ; %2: i32 = srem %0, %1
                {{_}} {{_}}: mov eax, r.32.y
                {{_}} {{_}}: xor rdx, rdx
                {{_}} {{_}}: idiv r.32.x
                ...
            ",
        );
    }

    #[test]
    fn cg_trunc() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i8 = trunc %0
            ",
            "
                ...
                ; %0: i32 = load_ti ...
                ...
                ; %1: i8 = trunc %0
                {{_}} {{_}}: mov [rbp-0x04], r.32.x
                ...
            ",
        );
    }

    #[test]
    fn cg_select() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = %0 ? 1i32 : 2i32
            ",
            "
                ...
                ; %1: i32 = %0 ? 1i32 : 2i32
                {{_}} {{_}}: mov r.32.x, 0x01
                {{_}} {{_}}: mov r.32.y, 0x02
                {{_}} {{_}}: cmp r.8.z, 0x00
                {{_}} {{_}}: cmovz r.64.x, r.64.y
                ...
            ",
        );
    }

    #[test]
    fn cg_sdiv() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = sdiv %0, %1
            ",
            "
                ...
                ; %2: i8 = sdiv %0, %1
                {{_}} {{_}}: movzx rax, r.8.x
                {{_}} {{_}}: movsx ax, al
                {{_}} {{_}}: idiv r.8.y
                ...
            ",
        );
    }

    #[test]
    fn cg_udiv() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = udiv %0, %1
            ",
            "
                ...
                ; %2: i8 = udiv %0, %1
                {{_}} {{_}}: movzx rax, r.8.x
                {{_}} {{_}}: xor rdx, rdx
                {{_}} {{_}}: div r.8.y
                ...
            ",
        );
    }

    #[test]
    fn cg_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = 1i8
                %2: i8 = add %0, %1
            ",
            "
                ...
                ; %2: i8 = add %0, 1i8
                ......
                {{_}} {{_}}: mov r.8.x, 0x01
                {{_}} {{_}}: add r.8.y, r.8.x
                ...
            ",
        );
    }

    #[test]
    fn cg_shl() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = shl %0, %1
            ",
            "
                ...
                ; %2: i8 = shl %0, %1
                ...
                {{_}} {{_}}: movzx rcx, r.8.a
                {{_}} {{_}}: shl r.8.b, cl
                ...
            ",
        );
    }

    #[test]
    fn cg_sitofp_float() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: float = si_to_fp %0
            ",
            "
                ...
                ; %1: float = si_to_fp %0
                {{_}} {{_}}: cvtsi2ss fp.128.x, r.32.x
                ...
                ",
        );
    }

    #[test]
    fn cg_sitofp_double() {
        codegen_and_test(
            "
              entry:
                %0: i32 = load_ti 0
                %1: double = si_to_fp %0
            ",
            "
                ...
                ; %1: double = si_to_fp %0
                {{_}} {{_}}: cvtsi2sd fp.128.x, r.32.x
                ...
                ",
        );
    }

    #[test]
    fn cg_fpext_float_double() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: double = fp_ext %0
            ",
            "
                ...
                ; %0: float = load_ti ...
                ; %1: double = fp_ext %0
                {{_}} {{_}}: movss [rbp-{{0x04}}], fp.128.x
                {{_}} {{_}}: cvtss2sd fp.128.x, fp.128.x
                ...
                ",
        );
    }

    #[test]
    fn cg_fptosi_float() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: i32 = fp_to_si %0
            ",
            "
                ...
                ; %1: i32 = fp_to_si %0
                {{_}} {{_}}: cvttss2si r.64.x, fp.128.x
                ...
                ",
        );
    }

    #[test]
    fn cg_fptosi_double() {
        codegen_and_test(
            "
              entry:
                %0: double = load_ti 0
                %1: i32 = fp_to_si %0
            ",
            "
                ...
                ; %1: i32 = fp_to_si %0
                {{_}} {{_}}: cvttsd2si r.64.x, fp.128.x
                ...
                ",
        );
    }

    #[test]
    fn cg_fdiv_float() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: float = fdiv %0, %1
            ",
            "
                ...
                ; %2: float = fdiv %0, %1
                ......
                {{_}} {{_}}: divss fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fdiv_double() {
        codegen_and_test(
            "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: double = fdiv %0, %1
            ",
            "
                ...
                ; %2: double = fdiv %0, %1
                ......
                {{_}} {{_}}: divsd fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fadd_float() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: float = fadd %0, %1
            ",
            "
                ...
                ; %2: float = fadd %0, %1
                ......
                {{_}} {{_}}: addss fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fadd_double() {
        codegen_and_test(
            "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: double = fadd %0, %1
            ",
            "
                ...
                ; %2: double = fadd %0, %1
                ......
                {{_}} {{_}}: addsd fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fsub_float() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: float = fsub %0, %1
            ",
            "
                ...
                ; %2: float = fsub %0, %1
                ......
                {{_}} {{_}}: subss fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fsub_double() {
        codegen_and_test(
            "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: double = fsub %0, %1
            ",
            "
                ...
                ; %2: double = fsub %0, %1
                ......
                {{_}} {{_}}: subsd fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fmul_float() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: float = fmul %0, %1
            ",
            "
                ...
                ; %2: float = fmul %0, %1
                ......
                {{_}} {{_}}: mulss fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fmul_double() {
        codegen_and_test(
            "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: double = fmul %0, %1
            ",
            "
                ...
                ; %2: double = fmul %0, %1
                ......
                {{_}} {{_}}: mulsd fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fcmp_float() {
        codegen_and_test(
            "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: i1 = f_ueq %0, %1
            ",
            "
                ...
                ; %2: i1 = f_ueq %0, %1
                {{_}} {{_}}: ucomiss fp.128.x, fp.128.y
                {{_}} {{_}}: setz r.8.x
                {{_}} {{_}}: setnp r.8.y
                {{_}} {{_}}: and r.8.x, r.8.y
                ...
                ",
        );
    }

    #[test]
    fn cg_fcmp_double() {
        codegen_and_test(
            "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: i1 = f_ueq %0, %1
            ",
            "
                ...
                ; %2: i1 = f_ueq %0, %1
                {{_}} {{_}}: ucomisd fp.128.x, fp.128.y
                {{_}} {{_}}: setz r.8.x
                {{_}} {{_}}: setnp r.8.y
                {{_}} {{_}}: and r.8.x, r.8.y
                ...
                ",
        );
    }

    #[test]
    fn cg_const_float() {
        codegen_and_test(
            "
              entry:
                %0: float = fadd 1.2float, 3.4float
            ",
            "
                ...
                ; %0: float = fadd 1.2float, 3.4float
                ...
                {{_}} {{_}}: mov r.32.x, 0x3f99999a
                {{_}} {{_}}: movd fp.128.x, r.32.x
                ...
                {{_}} {{_}}: mov r.32.x, 0x4059999a
                {{_}} {{_}}: movd fp.128.y, r.32.x
                ...
                {{_}} {{_}}: addss fp.128.x, fp.128.y
                ...
                ",
        );
    }

    #[test]
    fn cg_const_double() {
        codegen_and_test(
            "
              entry:
                %0: double = fadd 1.2double, 3.4double
            ",
            "
                ...
                ; %0: double = fadd 1.2double, 3.4double
                ...
                {{_}} {{_}}: mov r.64.x, 0x3ff3333333333333
                ...
                {{_}} {{_}}: mov r.64.x, 0x400b333333333333
                ...
                ",
        );
    }

    #[test]
    fn loop_jump_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
                tloop_start [%0]
                %1: i8 = 42i8
                tloop_jump [%1]
            ",
            "
                ...
                ; %0: i8 = load_ti ...
                ...
                ; tloop_start [%0]:
                ; tloop_jump [42i8]:
                {{_}} {{_}}: mov ...
                {{_}} {{_}}: mov r.8.x, 0x2a
                ...: jmp ...
            ",
        );
    }
}
