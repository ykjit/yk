//! The X86_64 JIT Code Generator.
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
        jit_ir::{self, BinOp, FloatTy, Inst, InstIdx, Module, Operand, Ty},
        CompilationError,
    },
    reg_alloc::{self, StackDirection, VarLocation},
    CodeGen,
};
#[cfg(any(debug_assertions, test))]
use crate::compile::jitc_yk::gdb::GdbCtx;
use crate::{
    compile::{
        jitc_yk::{
            aot_ir,
            jit_ir::{Const, IndirectCallIdx},
            trace_builder::Frame,
            YkSideTraceInfo,
        },
        CompiledTrace, Guard, GuardIdx, SideTraceInfo,
    },
    location::HotLocation,
    mt::MT,
};
use dynasmrt::{
    components::StaticLabel,
    dynasm,
    x64::{Rq, Rx},
    AssemblyOffset, DynamicLabel, DynasmApi, DynasmError, DynasmLabelApi, ExecutableBuffer,
    Register,
};
#[cfg(any(debug_assertions, test))]
use indexmap::IndexMap;
use parking_lot::Mutex;
use std::error::Error;
use std::sync::{Arc, Weak};
#[cfg(any(debug_assertions, test))]
use std::{cell::Cell, slice};
use ykaddr::addr::symbol_to_ptr;

mod deopt;
mod lsregalloc;

use deopt::__yk_deopt;
use lsregalloc::{LSRegAlloc, RegConstraint, RegSet};

/// General purpose argument registers as defined by the X86_64 SysV ABI.
static ARG_GP_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// The registers clobbered by a function call in the X86_64 SysV ABI.
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

/// Floating point argument registers as defined by the X86_64 SysV ABI.
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
static REG64_SIZE: usize = 8;
static RBP_DWARF_NUM: u16 = 6;

/// The X86_64 SysV ABI requires a 16-byte aligned stack prior to any call.
const SYSV_CALL_STACK_ALIGN: usize = 16;

/// On X86_64 the stack grows down.
const STACK_DIRECTION: StackDirection = StackDirection::GrowsDown;

/// A function that we can put a debugger breakpoint on.
/// FIXME: gross hack.
#[cfg(debug_assertions)]
#[no_mangle]
#[inline(never)]
pub extern "C" fn __yk_break() {}

/// A simple front end for the X86_64 code generator.
pub(crate) struct X86_64CodeGen;

impl CodeGen for X86_64CodeGen {
    fn codegen(
        &self,
        m: Module,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        Assemble::new(&m)?.codegen(mt, hl)
    }
}

impl X86_64CodeGen {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self))
    }
}

/// The X86_64 code generator.
struct Assemble<'a> {
    m: &'a jit_ir::Module,
    ra: LSRegAlloc<'a>,
    tloop_gp_reg_states: Option<[lsregalloc::RegState; 16]>,
    tloop_fp_reg_states: Option<[lsregalloc::RegState; 16]>,
    asm: dynasmrt::x64::Assembler,
    /// Deopt info, with one entry per guard, in the order that the guards appear in the trace.
    deoptinfo: Vec<DeoptInfo>,
    ///
    /// Maps assembly offsets to comments.
    ///
    /// Comments used by the trace printer for debugging and testing only.
    ///
    /// Each assembly offset can have zero or more comment lines.
    #[cfg(any(debug_assertions, test))]
    comments: Cell<IndexMap<usize, Vec<String>>>,
}

impl<'a> Assemble<'a> {
    fn new(m: &'a jit_ir::Module) -> Result<Box<Assemble<'a>>, CompilationError> {
        #[cfg(debug_assertions)]
        m.assert_well_formed();

        let mut inst_vals_alive_until = vec![InstIdx::new(0).unwrap(); m.insts_len()];
        for iidx in m.iter_all_inst_idxs() {
            let inst = m.inst_all(iidx);
            inst.map_packed_operand_locals(m, &mut |x| {
                inst_vals_alive_until[usize::from(x)] = iidx;
            });
        }

        // FIXME: this is a hack.
        for (iidx, inst) in m.iter_skipping_insts() {
            if let Inst::TraceLoopStart = inst {
                break;
            }
            inst_vals_alive_until[usize::from(iidx)] =
                InstIdx::new(usize::from(m.last_inst_idx()) + 1)?;
        }

        let asm = dynasmrt::x64::Assembler::new()
            .map_err(|e| CompilationError::ResourceExhausted(Box::new(e)))?;
        Ok(Box::new(Self {
            m,
            ra: LSRegAlloc::new(m, inst_vals_alive_until),
            asm,
            tloop_gp_reg_states: None,
            tloop_fp_reg_states: None,
            deoptinfo: Vec::new(),
            #[cfg(any(debug_assertions, test))]
            comments: Cell::new(IndexMap::new()),
        }))
    }

    fn codegen(
        mut self: Box<Self>,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let alloc_off = self.emit_prologue();

        for (iidx, inst) in self.m.iter_skipping_insts() {
            self.ra.expire_regs(iidx);
            self.cg_inst(iidx, inst)?;
        }

        // Loop the JITted code if the `tloop_start` label is present.
        let label = StaticLabel::global("tloop_start");
        match self.asm.labels().resolve_static(&label) {
            Ok(_) => {
                // Found the label, emit a jump to it.
                #[cfg(any(debug_assertions, test))]
                self.comment(self.asm.offset(), "tloop_backedge:".to_owned());
                self.ra
                    .restore_gp_reg_states_hack(&mut self.asm, self.tloop_gp_reg_states.unwrap());
                self.ra
                    .restore_fp_reg_states_hack(&mut self.asm, self.tloop_fp_reg_states.unwrap());
                dynasm!(self.asm; jmp ->tloop_start);
            }
            Err(DynasmError::UnknownLabel(_)) => {
                // Label not found. This is OK for unit testing, where we sometimes construct
                // traces that don't loop.
                #[cfg(test)]
                {
                    #[cfg(any(debug_assertions, test))]
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

        if !self.deoptinfo.is_empty() {
            // We now have to construct the "full" deopt points. Inside the trace itself, are just
            // a pair of instructions: a `cmp` followed by a `jnz` to a `fail_label` that has not
            // yet been defined. We now have to construct a full call to `__yk_deopt` for each of
            // those labels. Since, in general, we'll have multiple guards, we construct a simple
            // stub which puts an ID in a register then JMPs to (shared amongst all guards) code
            // which does the full call to __yk_deopt.
            let fail_labels = self
                .deoptinfo
                .iter()
                .map(|x| x.fail_label)
                .collect::<Vec<_>>();
            let deopt_label = self.asm.new_dynamic_label();
            for (deoptid, fail_label) in fail_labels.into_iter().enumerate() {
                #[cfg(any(debug_assertions, test))]
                self.comment(self.asm.offset(), format!("Deopt ID for guard {deoptid}"));
                dynasm!(self.asm
                    ;=> fail_label
                    ; mov rsi, QWORD deoptid as i64
                    ; jmp =>deopt_label
                );
            }

            #[cfg(any(debug_assertions, test))]
            self.comment(self.asm.offset(), "Call __yk_deopt".to_string());
            // Clippy points out that `__yk_depot as i64` isn't portable, but since this entire module
            // is x86 only, we don't need to worry about portability.
            #[allow(clippy::fn_to_numeric_cast)]
            {
                dynasm!(self.asm
                    ;=> deopt_label
                    ; mov rdi, [rbp]
                    ; mov rdx, rbp
                    ; mov rax, QWORD __yk_deopt as i64
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
        let (comments, gdb_ctx) = {
            use super::super::gdb;
            let comments = self.comments.take();
            let gdb_ctx = gdb::register_jitted_code(
                self.m.ctr_id(),
                buf.ptr(AssemblyOffset(0)),
                buf.size(),
                &comments,
            )?;
            (comments, gdb_ctx)
        };

        Ok(Arc::new(X64CompiledTrace {
            mt,
            buf,
            deoptinfo: self.deoptinfo,
            hl: Arc::downgrade(&hl),
            #[cfg(any(debug_assertions, test))]
            comments,
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
        #[cfg(any(debug_assertions, test))]
        self.comment(self.asm.offset(), inst.display(iidx, self.m).to_string());

        match inst {
            #[cfg(test)]
            jit_ir::Inst::BlackBox(_) => unreachable!(),
            jit_ir::Inst::ProxyConst(_) | jit_ir::Inst::ProxyInst(_) | jit_ir::Inst::Tombstone => {
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
    #[cfg(any(debug_assertions, test))]
    fn comment(&mut self, off: AssemblyOffset, line: String) {
        self.comments.get_mut().entry(off.0).or_default().push(line);
    }

    /// Emit the prologue of the JITted code.
    ///
    /// The JITted code is a function, so it has to stash the old stack poninter, open a new frame
    /// and allocate space for local variables etc.
    ///
    /// Note that there is no correspoinding `emit_epilogue()`. This is because the only way out of
    /// JITted code is via deoptimisation, which will rewrite the whole stack anyway.
    ///
    /// Returns the offset at which to patch up the stack allocation later.
    fn emit_prologue(&mut self) -> AssemblyOffset {
        #[cfg(any(debug_assertions, test))]
        self.comment(self.asm.offset(), "prologue".to_owned());

        // Start a frame for the JITted code.
        dynasm!(self.asm
            // Save pointers to ctrlp_vars and frameaddr for later use.
            ; push rdi
            ; push rsi
            // Save base pointer which we use to access the parent stack.
            ; push rbp
            // Reset base pointer.
            ; mov rbp, rsp
        );

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
                    ; mov r11, QWORD __yk_break as i64
                    ; call r11
                );
            }
        }

        alloc_off
    }

    fn patch_frame_allocation(&mut self, asm_off: AssemblyOffset) {
        // The stack should be 16-byte aligned after allocation. This ensures that calls in the
        // trace also get a 16-byte aligned stack, as per the SysV ABI.
        let stack_size = self.ra.align_stack(SYSV_CALL_STACK_ALIGN);

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

    fn cg_binop(
        &mut self,
        iidx: jit_ir::InstIdx,
        jit_ir::BinOpInst { lhs, binop, rhs }: &jit_ir::BinOpInst,
    ) {
        let lhs = lhs.unpack(self.m);
        let rhs = rhs.unpack(self.m);

        match binop {
            BinOp::Add => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match size {
                    1 => dynasm!(self.asm; add Rb(lhs_reg.code()), Rb(rhs_reg.code())),
                    2 => dynasm!(self.asm; add Rw(lhs_reg.code()), Rw(rhs_reg.code())),
                    4 => dynasm!(self.asm; add Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                    8 => dynasm!(self.asm; add Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::And => {
                let size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
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
                let [lhs_reg, _rhs_reg] = self.ra.get_gp_regs(
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
                let [lhs_reg, _rhs_reg] = self.ra.get_gp_regs(
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
                let [lhs_reg, _rhs_reg] = self.ra.get_gp_regs(
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
                let [_lhs_reg, rhs_reg] = self.ra.get_gp_regs_avoiding(
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
                let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
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
                let [_lhs_reg, rhs_reg] = self.ra.get_gp_regs_avoiding(
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
                let [_lhs_reg, rhs_reg, _rem_reg] = self.ra.get_gp_regs(
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
                let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
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
                let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
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
                let [_lhs_reg, rhs_reg] = self.ra.get_gp_regs_avoiding(
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
                let [lhs_reg, rhs_reg] = self.ra.get_fp_regs(
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
                let [lhs_reg, rhs_reg] = self.ra.get_fp_regs(
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
                let [lhs_reg, rhs_reg] = self.ra.get_fp_regs(
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
                let [lhs_reg, rhs_reg] = self.ra.get_fp_regs(
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

    fn cg_loadtraceinput(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadTraceInputInst) {
        // Find the argument register containing the pointer to the live variables struct.
        match self.m.type_(inst.tyidx()) {
            Ty::Integer(_) | Ty::Ptr => {
                let [tgt_reg] = self.ra.get_gp_regs_avoiding(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Output],
                    RegSet::from_vec(&STACKMAP_GP_REGS),
                );
                let size = self.m.inst_no_proxies(iidx).def_byte_size(self.m);
                debug_assert!(size <= REG64_SIZE);

                match self.m.tilocs()[usize::try_from(inst.locidx()).unwrap()] {
                    VarLocation::Register(reg_alloc::Register::GP(reg)) => {
                        // FIXME: Map this iidx to `reg` without this `mov`. Requires way to
                        // initialise register allocator with existing values.
                        dynasm!(self.asm; mov Rq(tgt_reg.code()), Rq(reg.code()));
                    }
                    VarLocation::Register(reg_alloc::Register::FP(_)) => {
                        // The value is a integer and thus can't be stored in a float register.
                        panic!()
                    }
                    VarLocation::Direct { frame_off, size } => {
                        // FIXME If we prime the register allocator with the interpreter frames
                        // stack size and don't create a new frame in the trace epilogue we can
                        // reference values directly using the current RBP value.
                        dynasm!(self.asm; mov Rq(tgt_reg.code()), QWORD [Rq(Rq::RBP.code())]);
                        match size {
                            8 => {
                                dynasm!(self.asm; lea Rq(tgt_reg.code()), [Rq(tgt_reg.code()) + frame_off])
                            }
                            _ => todo!(),
                        }
                    }
                    VarLocation::Indirect { frame_off, size } => {
                        dynasm!(self.asm; mov Rq(tgt_reg.code()), QWORD [Rq(Rq::RBP.code())]);
                        match size {
                            8 => {
                                dynasm!(self.asm; mov Rq(tgt_reg.code()), [Rq(tgt_reg.code()) + frame_off])
                            }
                            _ => todo!(),
                        }
                    }
                    _ => panic!(),
                }
            }
            Ty::Float(_fty) => {
                // FIXME: Work out which registers are used by stackmaps and avoid them.
                let [tgt_reg] = self
                    .ra
                    .get_fp_regs(&mut self.asm, iidx, [RegConstraint::Output]);
                let size = self.m.inst_no_proxies(iidx).def_byte_size(self.m);
                debug_assert!(size <= REG64_SIZE);

                match self.m.tilocs()[usize::try_from(inst.locidx()).unwrap()] {
                    VarLocation::Register(reg_alloc::Register::FP(reg)) => {
                        // FIXME: Map this iidx to `reg` without this `mov`. Requires way to
                        // initialise register allocator with existing values.
                        dynasm!(self.asm; movss Rx(tgt_reg.code()), Rx(reg.code()));
                    }
                    VarLocation::Register(reg_alloc::Register::GP(_)) => {
                        // The value is a float and thus can't be stored in a normal register.
                        panic!()
                    }
                    VarLocation::Direct { .. } => {
                        // The value is a pointer and thus can't be of type float.
                        panic!()
                    }
                    VarLocation::Indirect { .. } => {
                        // The value is a pointer and thus can't be of type float.
                        panic!()
                    }
                    _ => panic!(),
                }
            }
            Ty::Func(_) | Ty::Void | Ty::Unimplemented(_) => unreachable!(),
        }
    }

    fn cg_load(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadInst) {
        match self.m.type_(inst.tyidx()) {
            Ty::Integer(_) | Ty::Ptr => {
                let [reg] = self.ra.get_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(inst.operand(self.m))],
                );
                let size = self.m.inst_no_proxies(iidx).def_byte_size(self.m);
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
                let [src_reg] = self.ra.get_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(inst.operand(self.m))],
                );
                let [tgt_reg] = self
                    .ra
                    .get_fp_regs(&mut self.asm, iidx, [RegConstraint::Output]);
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
        let [reg] = self.ra.get_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(inst.ptr(self.m))],
        );

        dynasm!(self.asm ; add Rq(reg.code()), inst.off());
    }

    fn cg_dynptradd(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::DynPtrAddInst) {
        let [num_elems_reg, ptr_reg] = self.ra.get_gp_regs(
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
                let [tgt_reg, val_reg] = self.ra.get_gp_regs(
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
                let [tgt_reg] = self.ra.get_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(inst.tgt(self.m))],
                );
                let [val_reg] =
                    self.ra
                        .get_fp_regs(&mut self.asm, iidx, [RegConstraint::Input(val)]);
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
            .get_gp_regs(&mut self.asm, iidx, [RegConstraint::Output]);
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
                    let [_] = self.ra.get_fp_regs(
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
                    let [_] = self.ra.get_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputIntoRegAndClobber(arg.clone(), *reg)],
                    );
                }
            }
        }
        // If the function we called has a return value, then store it into a local variable.
        if fty.ret_type(self.m) != &Ty::Void {
            let [_] =
                self.ra
                    .get_gp_regs(&mut self.asm, iidx, [RegConstraint::OutputFromReg(Rq::RAX)]);
        }

        if fty.is_vararg() {
            // SysV X86_64 ABI says "rax is used to indicate the number of vector arguments passed
            // to a function requiring a variable number of arguments". Float arguments are passed
            // in vector registers.
            dynasm!(self.asm; mov rax, num_float_args);
        }

        // Actually perform the call.
        match (callee, callee_op) {
            (Some(p), None) => {
                let [reg] = self.ra.get_gp_regs_avoiding(
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
                let [reg] = self.ra.get_gp_regs_avoiding(
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
        let [lhs_reg, rhs_reg] = self.ra.get_gp_regs(
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
        let [lhs_reg, rhs_reg] = self.ra.get_fp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Input(lhs), RegConstraint::Input(rhs)],
        );
        let [tgt_reg] = self
            .ra
            .get_gp_regs(&mut self.asm, iidx, [RegConstraint::Output]);

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
            .get_gp_regs(&mut self.asm, iidx, [RegConstraint::Temporary]);
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

    fn cg_traceloopstart(&mut self) {
        // FIXME: peel the initial iteration of the loop to allow us to hoist loop invariants.
        self.tloop_gp_reg_states = Some(self.ra.snapshot_gp_reg_states_hack());
        self.tloop_fp_reg_states = Some(self.ra.snapshot_fp_reg_states_hack());
        dynasm!(self.asm; ->tloop_start:);
    }

    fn cg_sext(&mut self, iidx: InstIdx, i: &jit_ir::SExtInst) {
        let [reg] = self.ra.get_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::InputOutput(i.val(self.m))],
        );

        let src_val = i.val(self.m);
        let src_type = self.m.type_(src_val.tyidx(self.m));
        let src_size = src_type.byte_size().unwrap();

        let dest_type = self.m.type_(i.dest_tyidx());
        let dest_size = dest_type.byte_size().unwrap();

        // FIXME: assumes the input and output fit in a register.
        match (src_size, dest_size) {
            (1, 4) => dynasm!(self.asm; movsx Rd(reg.code()), Rb(reg.code())),
            (1, 8) => dynasm!(self.asm; movsx Rq(reg.code()), Rb(reg.code())),
            (2, 4) => dynasm!(self.asm; movsx Rd(reg.code()), Rw(reg.code())),
            (2, 8) => dynasm!(self.asm; movsx Rq(reg.code()), Rw(reg.code())),
            (4, 8) => dynasm!(self.asm; movsx Rq(reg.code()), Rd(reg.code())),
            _ => todo!("{} {}", src_size, dest_size),
        }
    }

    fn cg_zeroextend(&mut self, iidx: InstIdx, i: &jit_ir::ZeroExtendInst) {
        let [reg] = self.ra.get_gp_regs(
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
        let [src_reg] = self.ra.get_gp_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Input(inst.val(self.m))],
        );
        let [tgt_reg] = self
            .ra
            .get_fp_regs(&mut self.asm, iidx, [RegConstraint::Output]);

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

        let [src_reg] = self
            .ra
            .get_fp_regs(&mut self.asm, iidx, [RegConstraint::Input(from_val)]);
        let [tgt_reg] = self
            .ra
            .get_gp_regs(&mut self.asm, iidx, [RegConstraint::Output]);

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
                .get_fp_regs(&mut self.asm, iidx, [RegConstraint::InputOutput(from_val)]);

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
        let [_reg] = self.ra.get_gp_regs(
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
        let [true_reg, cond_reg, false_reg] = self.ra.get_gp_regs(
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
        let mut locs: Vec<VarLocation> = Vec::new();
        let gi = inst.guard_info(self.m);
        for lidx in gi.lives() {
            match self.m.inst_all(*lidx) {
                jit_ir::Inst::ProxyConst(c) => {
                    // The live variable is a constant (e.g. this can happen during inlining), so
                    // it doesn't have an allocation. We can just push the actual value instead
                    // which will be written as is during deoptimisation.
                    match self.m.const_(*c) {
                        Const::Int(tyidx, c) => {
                            let Ty::Integer(bits) = self.m.type_(*tyidx) else {
                                panic!()
                            };
                            locs.push(VarLocation::ConstInt { bits: *bits, v: *c })
                        }
                        _ => todo!(),
                    };
                }
                _ => {
                    // FIXME: This is a temporary hack (notice the function name!), where we force
                    // every live instruction to be spilled. This is only needed until deopt
                    // supports reloading from registers.
                    let frame_off = self.ra.stack_offset_gp_hack(&mut self.asm, *lidx);
                    let size = self.m.inst_no_proxies(*lidx).def_byte_size(self.m);
                    locs.push(VarLocation::Stack { frame_off, size });
                }
            }
        }

        let fail_label = self.asm.new_dynamic_label();
        // FIXME: Move `frames` instead of copying them (requires JIT module to be consumable).
        let deoptinfo = DeoptInfo {
            fail_label,
            frames: gi.frames().clone(),
            lives: locs,
            aotlives: gi.aotlives().to_vec(),
            callframes: gi.callframes.clone(),
            guard: Guard {
                failed: 0.into(),
                ct: None.into(),
            },
        };
        self.deoptinfo.push(deoptinfo);

        let cond = inst.cond(self.m);
        // ICmp instructions evaluate to a one-byte zero/one value.
        debug_assert_eq!(cond.byte_size(self.m), 1);
        let [reg] = self
            .ra
            .get_gp_regs(&mut self.asm, iidx, [RegConstraint::Input(cond)]);
        dynasm!(self.asm
            ; cmp Rb(reg.code()), inst.expect() as i8 // `as` intentional.
            ; jne =>fail_label
        );
    }
}

/// Information required by deoptimisation.
#[derive(Debug)]
struct DeoptInfo {
    fail_label: DynamicLabel,
    /// Vector of AOT stackmap IDs.
    frames: Vec<u64>,
    // Vector of live JIT variable locations.
    lives: Vec<VarLocation>,
    // Vector of live AOT variables.
    aotlives: Vec<aot_ir::InstID>,
    callframes: Vec<Frame>,
    // Keeps track of deopt amount and compiled side-trace.
    guard: Guard,
}

#[derive(Debug)]
pub(super) struct X64CompiledTrace {
    // Reference to the meta-tracer required for side tracing.
    mt: Arc<MT>,
    /// The executable code itself.
    buf: ExecutableBuffer,
    /// Vector of deopt info, tracked here so they can be freed when the compiled trace is
    /// dropped.
    deoptinfo: Vec<DeoptInfo>,
    /// Reference to the HotLocation, required for side tracing.
    hl: Weak<Mutex<HotLocation>>,
    /// Comments to be shown when printing the compiled trace using `AsmPrinter`.
    ///
    /// Maps a byte offset in the native JITted code to a collection of line comments to show when
    /// disassembling the trace.
    ///
    /// Used for testing and debugging.
    #[cfg(any(debug_assertions, test))]
    comments: IndexMap<usize, Vec<String>>,
    #[cfg(any(debug_assertions, test))]
    gdb_ctx: GdbCtx,
}

impl CompiledTrace for X64CompiledTrace {
    fn entry(&self) -> *const libc::c_void {
        self.buf.ptr(AssemblyOffset(0)) as *const libc::c_void
    }

    fn sidetraceinfo(&self, gidx: GuardIdx) -> Arc<dyn SideTraceInfo> {
        // FIXME: Can we reference these instead of copying them?
        let aotlives = self.deoptinfo[usize::from(gidx)].aotlives.clone();
        let callframes = self.deoptinfo[usize::from(gidx)].callframes.clone();
        Arc::new(YkSideTraceInfo {
            aotlives,
            callframes,
        })
    }

    fn guard(&self, id: crate::compile::GuardIdx) -> &crate::compile::Guard {
        &self.deoptinfo[id.0].guard
    }

    fn hl(&self) -> &std::sync::Weak<parking_lot::Mutex<crate::location::HotLocation>> {
        &self.hl
    }

    fn as_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
        self
    }

    #[cfg(any(debug_assertions, test))]
    fn disassemble(&self) -> Result<String, Box<dyn Error>> {
        AsmPrinter::new(&self.buf, &self.comments).to_string()
    }
}

/// Disassembles emitted code for testing and debugging purposes.
#[cfg(any(debug_assertions, test))]
struct AsmPrinter<'a> {
    buf: &'a ExecutableBuffer,
    comments: &'a IndexMap<usize, Vec<String>>,
}

#[cfg(any(debug_assertions, test))]
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
            Assemble::new(&m)
                .unwrap()
                .codegen(mt, Arc::new(Mutex::new(hl)))
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
                {{_}} {{_}}: mov r.64.x, ...
                ; %1: ptr = load_ti ...
                {{_}} {{_}}: mov r.64.y, ...
                ; *%1 = %0
                {{_}} {{_}}: mov [r.64.y], r.64.x
                ...
                ",
        );
    }

    #[test]
    fn cg_loadtraceinput_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = load_ti 0
            ",
            "
                ...
                ; %0: i8 = load_ti ...
                {{_}} {{_}}: mov r.64.x, rbx
                ...
                ",
        );
    }

    #[test]
    fn cg_loadtraceinput_i16_with_offset() {
        codegen_and_test(
            "
              entry:
                %0: i16 = load_ti 0
            ",
            "
                ...
                ; %0: i16 = load_ti ...
                {{_}} {{_}}: mov r.64.x, rbx
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
                {{{{_}}}} {{{{_}}}}: mov edi, [rbp-...
                {{{{_}}}} {{{{_}}}}: mov esi, [rbp-...
                {{{{_}}}} {{{{_}}}}: mov edx, [rbp-...
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
              func_decl puts (i8, i16, i32, i64, ptr, i8)

              entry:
                %0: i8 = load_ti 0
                %1: i16 = load_ti 1
                %2: i32 = load_ti 2
                %3: i64 = load_ti 3
                %4: ptr = load_ti 4
                %5: i8 = load_ti 5
                call @puts(%0, %1, %2, %3, %4, %5)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2, %3, %4, %5)
                ...
                {{{{_}}}} {{{{_}}}}: movzx rdi, byte ptr [rbp-...
                {{{{_}}}} {{{{_}}}}: movzx rsi, word ptr [rbp-...
                {{{{_}}}} {{{{_}}}}: mov edx, [rbp-...
                {{{{_}}}} {{{{_}}}}: mov rcx, [rbp-0x18]
                {{{{_}}}} {{{{_}}}}: mov r8, [rbp-0x10]
                {{{{_}}}} {{{{_}}}}: movzx r9, byte ptr [rbp-0x01]
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
                ... mov rsi, 0x00
                ... jmp ...
                ; call __yk_deopt
                ... mov rdi, [rbp]
                ... mov rdx, rbp
                ... mov rax, ...
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
                ... mov rsi, 0x00
                ... jmp ...
                ; call __yk_deopt
                ... mov rdi, [rbp]
                ... mov rdx, rbp
                ... mov rax, ...
                ... call rax
            ",
        );
    }

    #[test]
    fn unterminated_trace() {
        codegen_and_test(
            "
              entry:
                ",
            "
                ...
                ; Unterminated trace
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
                tloop_start
            ",
            "
                ...
                ; tloop_start:
                ; tloop_backedge:
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
                tloop_start
                %2: i8 = add %0, %0
            ",
            "
                ...
                ; %0: i8 = load_ti ...
                ...
                ; tloop_start:
                ; %2: i8 = add %0, %0
                ...
                ; tloop_backedge:
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
                ; unterminated trace
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
    fn cg_proxyconst() {
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
                {{_}} {{_}}: movss fp.128.x, xmm0
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
}
