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
        jit_ir::{self, BinOp, InstIdx, Operand, Ty},
        CompilationError,
    },
    abs_stack::AbstractStack,
    reg_alloc::{LocalAlloc, RegisterAllocator, StackDirection},
    CodeGen,
};
#[cfg(any(debug_assertions, test))]
use crate::compile::jitc_yk::gdb::GdbCtx;
use crate::compile::{
    jitc_yk::jit_ir::{Const, IndirectCallIdx},
    CompiledTrace,
};
use dynasmrt::{
    components::StaticLabel,
    dynasm,
    x64::{Rq, Rx},
    AssemblyOffset, DynasmApi, DynasmError, DynasmLabelApi, ExecutableBuffer, Register,
};
#[cfg(any(debug_assertions, test))]
use indexmap::IndexMap;
use std::sync::Arc;
#[cfg(any(debug_assertions, test))]
use std::{cell::Cell, error::Error, slice};
use ykaddr::addr::symbol_to_ptr;

mod deopt;

use deopt::__yk_deopt;

/// General purpose argument registers as defined by the X86_64 SysV ABI.
static ARG_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// Floating point argument registers as defined by the X86_64 SysV ABI.
static FLOAT_ARG_REGS: [Rx; 8] = [
    Rx::XMM0,
    Rx::XMM1,
    Rx::XMM2,
    Rx::XMM3,
    Rx::XMM4,
    Rx::XMM5,
    Rx::XMM6,
    Rx::XMM7,
];

/// The argument index of the live variables struct argument in the JITted code function.
static JITFUNC_LIVEVARS_ARGIDX: usize = 0;

/// The size of a 64-bit register in bytes.
static REG64_SIZE: usize = 8;
static RBP_DWARF_NUM: u16 = 6;

/// General purpose "work registers", i.e. the registers we use temporarily (where possible) for
/// operands to, and results of, intermediate computations.
///
/// We choose callee-save registers so that we don't have to worry about storing/restoring them
/// when we do a function call to external code.
static WR0: Rq = Rq::R12;
static WR1: Rq = Rq::R13;
static WR2: Rq = Rq::R14;

/// Floating point work registers.
static WF0: Rx = Rx::XMM0;
static WF1: Rx = Rx::XMM1;

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

/// Load an SSE register with an immediate float/double.
///
/// Note that if `size < mem::sizeof::<double>()`, then `val` may be rounded to the closest
/// possible value (as per `as` cast semantics).
///
/// # Panics
///
/// This will panic if `size` isn't either 4 or 8 (bytes).
fn imm_float_into_reg(val: f64, reg: Rx, size: usize, asm: &mut dynasmrt::x64::Assembler) {
    // Note that there's no way to directly move an immediate value into an xmm, so we
    // have to go via a general purpose register. Because we don't know if this
    // register is already in use by the parent codegen routine, we have no option but
    // the save/restore it.
    //
    // FIXME: when we have a proper register allocator, it can have a "with_temp_reg()" method. It
    // would either pass the closure a free register, or if there isn't one, do the save/restore on
    // our behalf.
    match size {
        4 => {
            let val = val as f32; // NB: potential rounding!
            let val = val.to_bits() as i32;
            dynasm!(asm
                ; push Rq(WR0.code())
                ; mov Rd(WR0.code()), val
                ; movd Rx(reg.code()), Rd(WR0.code())
                ; pop Rq(WR0.code())
            );
        }
        8 => {
            let val = val.to_bits() as i64;
            dynasm!(asm
                ; push Rq(WR0.code())
                ; mov Rq(WR0.code()), QWORD val
                ; movq Rx(reg.code()), Rq(WR0.code())
                ; pop Rq(WR0.code())
            );
        }
        _ => panic!(),
    }
}

/// The X86_64 code generator.
pub(crate) struct X64CodeGen<'a> {
    m: &'a jit_ir::Module,
    asm: dynasmrt::x64::Assembler,
    /// Abstract stack pointer, as a relative offset from `RBP`. The higher this number, the larger
    /// the JITted code's stack. That means that even on a host where the stack grows down, this
    /// value grows up.
    stack: AbstractStack,
    /// Register allocator.
    ra: Box<dyn RegisterAllocator>,
    /// Deopt info.
    deoptinfo: Vec<DeoptInfo>,
    /// Maps assembly offsets to comments.
    ///
    /// Comments used by the trace printer for debugging and testing only.
    ///
    /// Each assembly offset can have zero or more comment lines.
    #[cfg(any(debug_assertions, test))]
    comments: Cell<IndexMap<usize, Vec<String>>>,
}

impl<'a> CodeGen<'a> for X64CodeGen<'a> {
    fn new(
        m: &'a jit_ir::Module,
        ra: Box<dyn RegisterAllocator>,
    ) -> Result<Box<X64CodeGen<'a>>, CompilationError> {
        #[cfg(debug_assertions)]
        m.assert_well_formed();
        let asm = dynasmrt::x64::Assembler::new()
            .map_err(|e| CompilationError::ResourceExhausted(Box::new(e)))?;
        Ok(Box::new(Self {
            m,
            asm,
            stack: Default::default(),
            ra,
            deoptinfo: Vec::new(),
            #[cfg(any(debug_assertions, test))]
            comments: Cell::new(IndexMap::new()),
        }))
    }

    fn codegen(mut self: Box<Self>) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let alloc_off = self.emit_prologue();

        for idx in self.m.iter_skipping_inst_idxs() {
            self.cg_inst(idx, self.m.inst(idx))?;
        }

        // Loop the JITted code if the `tloop_start` label is present.
        let label = StaticLabel::global("tloop_start");
        match self.asm.labels().resolve_static(&label) {
            Ok(_) => {
                // Found the label, emit a jump to it.
                #[cfg(any(debug_assertions, test))]
                self.comment(self.asm.offset(), "tloop_backedge:".to_owned());
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
                #[cfg(not(test))]
                panic!("unterminated trace in non-unit-test");
            }
            Err(e) => {
                // Any other error suggests something has gone quite wrong. Just crash.
                panic!("{}", e.to_string())
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
            buf,
            deoptinfo: self.deoptinfo,
            #[cfg(any(debug_assertions, test))]
            comments,
            #[cfg(any(debug_assertions, test))]
            gdb_ctx,
        }))
    }
}

impl<'a> X64CodeGen<'a> {
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
            jit_ir::Inst::Store(i) => self.cg_store(i),
            jit_ir::Inst::LookupGlobal(i) => self.cg_lookupglobal(iidx, i),
            jit_ir::Inst::Call(i) => self.cg_call(iidx, i)?,
            jit_ir::Inst::IndirectCall(i) => self.cg_indirectcall(iidx, i)?,
            jit_ir::Inst::Icmp(i) => self.cg_icmp(iidx, i),
            jit_ir::Inst::Guard(i) => self.cg_guard(i),
            jit_ir::Inst::Arg(i) => self.cg_arg(iidx, *i),
            jit_ir::Inst::TraceLoopStart => self.cg_traceloopstart(),
            jit_ir::Inst::SExt(i) => self.cg_sext(iidx, i),
            jit_ir::Inst::ZeroExtend(i) => self.cg_zeroextend(iidx, i),
            jit_ir::Inst::Trunc(i) => self.cg_trunc(iidx, i),
            jit_ir::Inst::Select(i) => self.cg_select(iidx, i),
            jit_ir::Inst::SIToFP(i) => self.cg_sitofp(iidx, i),
            jit_ir::Inst::FPExt(i) => self.cg_fpext(iidx, i),
            jit_ir::Inst::Fcmp(i) => self.cg_fcmp(iidx, i),
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
            ; push rbp
            ; mov rbp, rsp
            // Save pointers to ctrlp_vars and frameaddr for later use.
            ; push rdi
            ; push rsi
            // Reset the basepointer so the spill allocator doesn't overwrite the two values we
            // just pushed.
            ; mov rbp, rsp
        );

        // Emit a dummy frame allocation instruction that initially allocates 0 bytes, but will be
        // patched later when we know how big the frame needs to be.
        let alloc_off = self.asm.offset();
        dynasm!(self.asm
            ; sub rsp, DWORD 0
        );

        #[cfg(debug_assertions)]
        {
            self.comment(self.asm.offset(), "Breakpoint hack".into());
            self.stack.align(SYSV_CALL_STACK_ALIGN);
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
        self.stack.align(SYSV_CALL_STACK_ALIGN);

        match i32::try_from(self.stack.size()) {
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
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; add Rq(WR0.code()), Rq(WR1.code())),
                    4 => dynasm!(self.asm; add Rd(WR0.code()), Rd(WR1.code())),
                    2 => dynasm!(self.asm; add Rw(WR0.code()), Rw(WR1.code())),
                    1 => dynasm!(self.asm; add Rb(WR0.code()), Rb(WR1.code())),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::And => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; and Rq(WR0.code()), Rq(WR1.code())),
                    4 => dynasm!(self.asm; and Rd(WR0.code()), Rd(WR1.code())),
                    2 => dynasm!(self.asm; and Rw(WR0.code()), Rw(WR1.code())),
                    1 => dynasm!(self.asm; and Rb(WR0.code()), Rb(WR1.code())),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::AShr => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(Rq::RCX, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; sar Rq(WR0.code()), cl),
                    4 => dynasm!(self.asm; sar Rd(WR0.code()), cl),
                    2 => dynasm!(self.asm; sar Rw(WR0.code()), cl),
                    1 => dynasm!(self.asm; sar Rb(WR0.code()), cl),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::LShr => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(Rq::RCX, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; shr Rq(WR0.code()), cl),
                    4 => dynasm!(self.asm; shr Rd(WR0.code()), cl),
                    2 => dynasm!(self.asm; shr Rw(WR0.code()), cl),
                    1 => dynasm!(self.asm; shr Rb(WR0.code()), cl),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::Shl => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(Rq::RCX, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; shl Rq(WR0.code()), cl),
                    4 => dynasm!(self.asm; shl Rd(WR0.code()), cl),
                    2 => dynasm!(self.asm; shl Rw(WR0.code()), cl),
                    1 => dynasm!(self.asm; shl Rb(WR0.code()), cl),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::Mul => {
                self.load_operand(Rq::RAX, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; mul Rq(WR1.code())),
                    4 => dynasm!(self.asm; mul Rd(WR1.code())),
                    2 => dynasm!(self.asm; mul Rw(WR1.code())),
                    1 => dynasm!(self.asm; mul Rb(WR1.code())),
                    _ => todo!(),
                }
                // Note that because we are code-genning an unchecked multiply, the higher-order part of
                // the result in RDX is entirely ignored.
                self.store_new_local(iidx, Rq::RAX);
            }
            BinOp::Or => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; or Rq(WR0.code()), Rq(WR1.code())),
                    4 => dynasm!(self.asm; or Rd(WR0.code()), Rd(WR1.code())),
                    2 => dynasm!(self.asm; or Rw(WR0.code()), Rw(WR1.code())),
                    1 => dynasm!(self.asm; or Rb(WR0.code()), Rb(WR1.code())),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::SDiv => {
                // The dividend is hard-coded into DX:AX/EDX:EAX/RDX:RAX. However unless we have 128bit
                // values or want to optimise register usage, we won't be needing this, and just zero out
                // RDX.
                self.load_operand(Rq::RAX, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same

                // Signed division (idiv) operates on the DX:AX, EDX:EAX, RDX:RAX registers, so we
                // use `cdq`/`cqo` to double the size via sign extension and store the result in
                // DX:AX, EDX:EAX, RDX:RAX.
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; cqo; idiv Rq(WR1.code())),
                    4 => dynasm!(self.asm; cdq; idiv Rd(WR1.code())),
                    2 => dynasm!(self.asm; cwd; idiv Rw(WR1.code())),
                    // There's no `cwd` equivalent for byte-sized values, so we use `movsx`
                    // (sign-extend) instead.
                    1 => dynasm!(self.asm; movsx ax, al; idiv Rb(WR1.code())),
                    _ => todo!(),
                }
                // The quotient is stored in RAX. We don't care about the remainder stored in RDX.
                self.store_new_local(iidx, Rq::RAX);
            }
            BinOp::SRem => {
                // The dividend is hard-coded into DX:AX/EDX:EAX/RDX:RAX. However unless we have 128bit
                // values or want to optimise register usage, we won't be needing this, and just zero out
                // RDX.
                dynasm!(self.asm; xor rdx, rdx);
                self.load_operand(Rq::RAX, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; idiv Rq(WR1.code())),
                    4 => dynasm!(self.asm; idiv Rd(WR1.code())),
                    2 => dynasm!(self.asm; idiv Rw(WR1.code())),
                    1 => dynasm!(self.asm; idiv Rb(WR1.code())),
                    _ => todo!(),
                }
                // The remainder is stored in RDX. We don't care about the quotient stored in RAX.
                self.store_new_local(iidx, Rq::RDX);
            }
            BinOp::Sub => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; sub Rq(WR0.code()), Rq(WR1.code())),
                    4 => dynasm!(self.asm; sub Rd(WR0.code()), Rd(WR1.code())),
                    2 => dynasm!(self.asm; sub Rw(WR0.code()), Rw(WR1.code())),
                    1 => dynasm!(self.asm; sub Rb(WR0.code()), Rb(WR1.code())),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::Xor => {
                self.load_operand(WR0, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; xor Rq(WR0.code()), Rq(WR1.code())),
                    4 => dynasm!(self.asm; xor Rd(WR0.code()), Rd(WR1.code())),
                    2 => dynasm!(self.asm; xor Rw(WR0.code()), Rw(WR1.code())),
                    1 => dynasm!(self.asm; xor Rb(WR0.code()), Rb(WR1.code())),
                    _ => todo!(),
                }
                self.store_new_local(iidx, WR0);
            }
            BinOp::UDiv => {
                // Like SDiv the dividend goes into AX, DX:AX, EDX:EAX, RDX:RAX. But since the
                // values aren't signed we don't need to sign-extend them and can just zero out
                // `rdx`.
                dynasm!(self.asm; xor rdx, rdx);
                self.load_operand(Rq::RAX, &lhs); // FIXME: assumes value will fit in a reg.
                self.load_operand(WR1, &rhs); // ^^^ same
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; div Rq(WR1.code())),
                    4 => dynasm!(self.asm; div Rd(WR1.code())),
                    2 => dynasm!(self.asm; div Rw(WR1.code())),
                    1 => dynasm!(self.asm; div Rb(WR1.code())),
                    _ => todo!(),
                }
                // The quotient is stored in RAX. We don't care about the remainder stored in RDX.
                self.store_new_local(iidx, Rq::RAX);
            }
            BinOp::FDiv => {
                self.load_operand_float(WF0, &lhs);
                self.load_operand_float(WF1, &rhs);
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; divsd Rx(WF0.code()), Rx(WF1.code())),
                    4 => dynasm!(self.asm; divss Rx(WF0.code()), Rx(WF1.code())),
                    _ => todo!(),
                }
                self.store_new_local_float(iidx, WF0);
            }
            BinOp::FAdd => {
                self.load_operand_float(WF0, &lhs);
                self.load_operand_float(WF1, &rhs);
                match lhs.byte_size(self.m) {
                    8 => dynasm!(self.asm; addsd Rx(WF0.code()), Rx(WF1.code())),
                    4 => dynasm!(self.asm; addss Rx(WF0.code()), Rx(WF1.code())),
                    _ => todo!(),
                }
                self.store_new_local_float(iidx, WF0);
            }
            x => todo!("{x:?}"),
        }
    }

    fn cg_loadtraceinput(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadTraceInputInst) {
        // Find the argument register containing the pointer to the live variables struct.
        let base_reg = ARG_REGS[JITFUNC_LIVEVARS_ARGIDX].code();

        // Now load the value into a new local variable from [base_reg+off].
        match i32::try_from(inst.off()) {
            Ok(off) => {
                let size = self.m.inst(iidx).def_byte_size(self.m);
                debug_assert!(size <= REG64_SIZE);
                match size {
                    8 => dynasm!(self.asm ; mov Rq(WR0.code()), [Rq(base_reg) + off]),
                    4 => dynasm!(self.asm ; mov Rd(WR0.code()), [Rq(base_reg) + off]),
                    2 => dynasm!(self.asm ; movzx Rd(WR0.code()), WORD [Rq(base_reg) + off]),
                    1 => dynasm!(self.asm ; movzx Rq(WR0.code()), BYTE [Rq(base_reg) + off]),
                    _ => todo!("{}", size),
                };
                self.store_new_local(iidx, WR0);
            }
            _ => todo!(),
        }
    }

    fn cg_load(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadInst) {
        self.load_operand(WR0, &inst.operand(self.m)); // FIXME: assumes value will fit in a reg.
        let size = self.m.inst(iidx).def_byte_size(self.m);
        debug_assert!(size <= REG64_SIZE);
        match size {
            8 => dynasm!(self.asm ; mov Rq(WR0.code()), [Rq(WR0.code())]),
            4 => dynasm!(self.asm ; mov Rd(WR0.code()), [Rq(WR0.code())]),
            2 => dynasm!(self.asm ; movzx Rd(WR0.code()), WORD [Rq(WR0.code())]),
            1 => dynasm!(self.asm ; movzx Rq(WR0.code()), BYTE [Rq(WR0.code())]),
            _ => todo!("{}", size),
        };
        self.store_new_local(iidx, WR0);
    }

    fn cg_ptradd(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::PtrAddInst) {
        self.load_operand(WR0, &inst.ptr(self.m));
        // LLVM semantics dictate that the offset should be sign-extended/truncated up/down to the
        // size of the LLVM pointer index type. For address space zero on x86, truncation can't
        // happen, and when an immediate second operand is used for x86_64 `add`, it is implicitly
        // sign extended.
        dynasm!(self.asm ; add Rq(WR0.code()), inst.off());
        self.store_new_local(iidx, WR0);
    }

    fn cg_dynptradd(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::DynPtrAddInst) {
        self.load_operand(WR0, &inst.num_elems(self.m));
        self.load_operand(WR1, &inst.ptr(self.m));
        // LLVM semantics dictate that the element size and number of elements should be
        // sign-extended/truncated up/down to the size of the LLVM pointer index type. For address
        // space zero on x86_64, truncation can't happen, and when an immediate third operand is
        // used for x86_64 `mul`, it is implicitly sign extended.
        dynasm!(self.asm
            // multiply the element size by the number of elements.
            ; imul Rq(WR0.code()), Rq(WR0.code()), i32::from(inst.elem_size())
            // add the result to the pointer.
            ; add Rq(WR0.code()), Rq(WR1.code())
        );
        self.store_new_local(iidx, WR0);
    }

    fn cg_store(&mut self, inst: &jit_ir::StoreInst) {
        self.load_operand(WR0, &inst.tgt(self.m));
        let val = inst.val(self.m);
        self.load_operand(WR1, &val); // FIXME: assumes the value fits in a reg
        match val.byte_size(self.m) {
            8 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rq(WR1.code())),
            4 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rd(WR1.code())),
            2 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rw(WR1.code())),
            1 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rb(WR1.code())),
            _ => todo!(),
        }
    }

    #[cfg(not(test))]
    fn cg_lookupglobal(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LookupGlobalInst) {
        let decl = inst.decl(self.m);
        if decl.is_threadlocal() {
            todo!();
        }
        let sym_addr = self.m.globalvar_ptr(inst.global_decl_idx()).addr();
        dynasm!(self.asm ; mov Rq(WR0.code()), QWORD i64::try_from(sym_addr).unwrap());
        self.store_new_local(iidx, WR0);
    }

    #[cfg(test)]
    fn cg_lookupglobal(&mut self, _inst_idx: jit_ir::InstIdx, _inst: &jit_ir::LookupGlobalInst) {
        panic!("Cannot lookup globals in cfg(test) as ykllvm will not have compiled this binary");
    }

    fn emit_call(
        &mut self,
        iidx: InstIdx,
        fty: &jit_ir::FuncTy,
        callee: Rq,
        args: &[Operand],
    ) -> Result<(), CompilationError> {
        // FIXME: non-SysV ABIs
        //
        // FIXME: when we have a proper register allocator, this routine will be required to
        // adhere to caller-save conventions.

        // Arrange arguments according to the ABI.
        let mut gp_regs = ARG_REGS.iter();
        let mut float_regs = FLOAT_ARG_REGS.iter();
        let mut num_float_args = 0;
        for arg in args {
            if matches!(self.m.type_(arg.tyidx(self.m)), jit_ir::Ty::Float(_)) {
                let Some(reg) = float_regs.next() else {
                    todo!("ran out of floating point arg regs: requires spill");
                };
                self.load_operand_float(*reg, arg);
                num_float_args += 1;
            } else {
                let Some(reg) = gp_regs.next() else {
                    todo!("ran out of general purpose arg regs: requires spill");
                };
                self.load_operand(*reg, arg);
            }
        }

        if fty.is_vararg() {
            // SysV X86_64 ABI says "rax is used to indicate the number of vector arguments passed
            // to a function requiring a variable number of arguments". Float arguments are passed
            // in vector registers.
            dynasm!(self.asm; mov rax, num_float_args);
        }

        // The SysV x86_64 ABI requires the stack to be 16-byte aligned prior to a call.
        self.stack.align(SYSV_CALL_STACK_ALIGN);

        // Actually perform the call.
        dynasm!(self.asm; call Rq(callee.code()));

        // If the function we called has a return value, then store it into a local variable.
        if fty.ret_type(self.m) != &Ty::Void {
            self.store_new_local(iidx, Rq::RAX);
        }

        Ok(())
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
        dynasm!(self.asm; mov Rq(WR0.code()), QWORD va as i64);
        self.emit_call(iidx, fty, WR0, &args)
    }

    /// Codegen a indirect call.
    fn cg_indirectcall(
        &mut self,
        iidx: InstIdx,
        indirect_call_idx: &IndirectCallIdx,
    ) -> Result<(), CompilationError> {
        let inst = self.m.indirect_call(*indirect_call_idx);
        self.load_operand(WR0, &inst.target(self.m));
        let jit_ir::Ty::Func(fty) = self.m.type_(inst.ftyidx()) else {
            panic!()
        };
        let args = (0..(inst.num_args()))
            .map(|i| inst.operand(self.m, i))
            .collect::<Vec<_>>();
        self.emit_call(iidx, fty, WR0, &args)
    }

    fn cg_icmp(&mut self, iidx: InstIdx, inst: &jit_ir::IcmpInst) {
        let (lhs, pred, rhs) = (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m));

        // FIXME: assumes values fit in a registers
        self.load_operand(WR0, &lhs);
        self.load_operand(WR1, &rhs);

        // Perform the comparison.
        match rhs.byte_size(self.m) {
            8 => dynasm!(self.asm; cmp Rq(WR0.code()), Rq(WR1.code())),
            4 => dynasm!(self.asm; cmp Rd(WR0.code()), Rd(WR1.code())),
            2 => dynasm!(self.asm; cmp Rw(WR0.code()), Rw(WR1.code())),
            1 => dynasm!(self.asm; cmp Rb(WR0.code()), Rb(WR1.code())),
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
            jit_ir::Predicate::Equal => dynasm!(self.asm; sete Rb(WR0.code())),
            jit_ir::Predicate::NotEqual => dynasm!(self.asm; setne Rb(WR0.code())),
            jit_ir::Predicate::UnsignedGreater => dynasm!(self.asm; seta Rb(WR0.code())),
            jit_ir::Predicate::UnsignedGreaterEqual => dynasm!(self.asm; setae Rb(WR0.code())),
            jit_ir::Predicate::UnsignedLess => dynasm!(self.asm; setb Rb(WR0.code())),
            jit_ir::Predicate::UnsignedLessEqual => dynasm!(self.asm; setb Rb(WR0.code())),
            jit_ir::Predicate::SignedGreater => dynasm!(self.asm; setg Rb(WR0.code())),
            jit_ir::Predicate::SignedGreaterEqual => dynasm!(self.asm; setge Rb(WR0.code())),
            jit_ir::Predicate::SignedLess => dynasm!(self.asm; setl Rb(WR0.code())),
            jit_ir::Predicate::SignedLessEqual => dynasm!(self.asm; setle Rb(WR0.code())),
        }
        self.store_new_local(iidx, WR0);
    }

    fn cg_fcmp(&mut self, iidx: InstIdx, inst: &jit_ir::FcmpInst) {
        let (lhs, pred, rhs) = (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m));

        self.load_operand_float(WF0, &lhs);
        self.load_operand_float(WF1, &rhs);

        // Perform the comparison.
        let size = self
            .m
            .type_(inst.lhs.unpack(self.m).tyidx(self.m))
            .byte_size()
            .unwrap();

        match pred.is_ordered() {
            Some(true) => match size {
                8 => dynasm!(self.asm; comisd Rx(WF0.code()), Rx(WF1.code())),
                4 => dynasm!(self.asm; comiss Rx(WF0.code()), Rx(WF1.code())),
                _ => panic!(),
            },
            Some(false) => match size {
                8 => dynasm!(self.asm; ucomisd Rx(WF0.code()), Rx(WF1.code())),
                4 => dynasm!(self.asm; ucomiss Rx(WF0.code()), Rx(WF1.code())),
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
                dynasm!(self.asm; sete Rb(WR0.code()))
            }
            jit_ir::FloatPredicate::UnorderedNotEqual => {
                dynasm!(self.asm; setne Rb(WR0.code()))
            }
            jit_ir::FloatPredicate::OrderedGreater => dynasm!(self.asm; seta Rb(WR0.code())),
            jit_ir::FloatPredicate::OrderedGreaterEqual => {
                dynasm!(self.asm; setae Rb(WR0.code()))
            }
            jit_ir::FloatPredicate::OrderedLess => dynasm!(self.asm; setb Rb(WR0.code())),
            jit_ir::FloatPredicate::OrderedLessEqual => dynasm!(self.asm; setbe Rb(WR0.code())),
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
        match pred {
            jit_ir::FloatPredicate::OrderedNotEqual | jit_ir::FloatPredicate::UnorderedNotEqual => {
                dynasm!(self.asm
                    ; setp Rb(WR1.code())
                    ; or Rb(WR0.code()), Rb(WR1.code())
                );
            }
            _ => {
                dynasm!(self.asm
                    ; setnp Rb(WR1.code())
                    ; and Rb(WR0.code()), Rb(WR1.code())
                );
            }
        }
        self.store_new_local(iidx, WR0);
    }

    fn cg_arg(&mut self, iidx: InstIdx, idx: u16) {
        // For arguments passed into the trace function we simply inform the register allocator
        // where they are stored and let the allocator take things from there.
        self.store_new_local(iidx, ARG_REGS[usize::from(idx)]);
    }

    fn cg_traceloopstart(&mut self) {
        // FIXME: peel the initial iteration of the loop to allow us to hoist loop invariants.
        dynasm!(self.asm; ->tloop_start:);
    }

    fn cg_sext(&mut self, iidx: InstIdx, i: &jit_ir::SExtInst) {
        let src_val = i.val(self.m);
        let src_type = self.m.type_(src_val.tyidx(self.m));
        let src_size = src_type.byte_size().unwrap();

        let dest_type = self.m.type_(i.dest_tyidx());
        let dest_size = dest_type.byte_size().unwrap();

        // FIXME: assumes the input and output fit in a register.
        self.load_operand(WR0, &src_val);
        match (src_size, dest_size) {
            (1, 8) => dynasm!(self.asm; movsx Rq(WR0.code()), Rb(WR0.code())),
            (1, 4) => dynasm!(self.asm; movsx Rd(WR0.code()), Rb(WR0.code())),
            (2, 4) => dynasm!(self.asm; movsx Rd(WR0.code()), Rw(WR0.code())),
            (2, 8) => dynasm!(self.asm; movsx Rq(WR0.code()), Rw(WR0.code())),
            (4, 8) => dynasm!(self.asm; movsx Rq(WR0.code()), Rd(WR0.code())),
            _ => todo!("{} {}", src_size, dest_size),
        }
        self.store_new_local(iidx, WR0);
    }

    fn cg_zeroextend(&mut self, iidx: InstIdx, i: &jit_ir::ZeroExtendInst) {
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
        self.load_operand(WR0, &from_val);
        debug_assert!(to_size <= REG64_SIZE);

        // FIXME: Assumes we don't assign to sub-registers.
        dynasm!(self.asm; mov Rq(WR0.code()), Rq(WR0.code()));

        self.store_new_local(iidx, WR0);
    }

    fn cg_sitofp(&mut self, inst_idx: InstIdx, i: &jit_ir::SIToFPInst) {
        let from_val = i.val(self.m);
        let to_type = self.m.type_(i.dest_tyidx());

        self.load_operand(WR0, &from_val);
        match to_type {
            jit_ir::Ty::Float(jit_ir::FloatTy::Float) => {
                dynasm!(self.asm; cvtsi2ss Rx(WF0.code()), Rq(WR0.code()))
            }
            jit_ir::Ty::Float(jit_ir::FloatTy::Double) => {
                dynasm!(self.asm; cvtsi2sd Rx(WF0.code()), Rq(WR0.code()))
            }
            _ => panic!(),
        }
        self.store_new_local_float(inst_idx, WF0);
    }

    fn cg_fpext(&mut self, iidx: InstIdx, i: &jit_ir::FPExtInst) {
        let from_val = i.val(self.m);
        let from_type = self.m.type_(from_val.tyidx(self.m));
        let to_type = self.m.type_(i.dest_tyidx());

        self.load_operand_float(WF0, &from_val);
        if let (
            jit_ir::Ty::Float(jit_ir::FloatTy::Float),
            jit_ir::Ty::Float(jit_ir::FloatTy::Double),
        ) = (from_type, to_type)
        {
            dynasm!(self.asm; cvtss2sd Rx(WF0.code()), Rx(WF0.code()));
        } else {
            // since we only support 32- and 64-bit floats, any other conversion is nonsensical.
            panic!();
        }
        self.store_new_local_float(iidx, WF0);
    }

    fn cg_trunc(&mut self, iidx: InstIdx, i: &jit_ir::TruncInst) {
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
        self.load_operand(WR0, &from_val);
        debug_assert!(to_size <= REG64_SIZE);

        // FIXME: There's no instruction on x86_64 to mov from a bigger register into a smaller
        // register. The simplest way to truncate the value is to zero out the higher order bits.
        // At the moment this happens automatically when we load the value from the stack and then
        // store it back. This currently works because variables can only live on the stack, but
        // this will change once we have a proper register allocator at which point we need to
        // revisit this implementation.

        self.store_new_local(iidx, WR0);
    }

    fn cg_select(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::SelectInst) {
        // First load the true case. We then immediately follow this up with a conditional move,
        // overwriting the value with the false case, if the condition was false.
        self.load_operand(WR0, &inst.trueval(self.m));
        self.load_operand(WR1, &inst.cond(self.m));
        self.load_operand(WR2, &inst.falseval(self.m));
        dynasm!(self.asm ; cmp Rb(WR1.code()), 0);
        dynasm!(self.asm ; cmove Rq(WR0.code()), Rq(WR2.code()));
        self.store_new_local(iidx, WR0);
    }

    fn cg_guard(&mut self, inst: &jit_ir::GuardInst) {
        let cond = inst.cond(self.m);

        // ICmp instructions evaluate to a one-byte zero/one value.
        debug_assert_eq!(cond.byte_size(self.m), 1);

        // Convert the guard info into deopt info and store it on the heap.
        let mut locs: Vec<LocalAlloc> = Vec::new();
        let gi = inst.guard_info(self.m);
        for lidx in gi.lives() {
            match self.m.inst(*lidx) {
                jit_ir::Inst::ProxyConst(c) => {
                    // The live variable is a constant (e.g. this can happen during inlining), so
                    // it doesn't have an allocation. We can just push the actual value instead
                    // which will be written as is during deoptimisation.
                    match self.m.const_(*c) {
                        Const::Int(_, c) => locs.push(LocalAlloc::ConstInt(*c)),
                        _ => todo!(),
                    };
                }
                _ => {
                    locs.push(*self.ra.allocation(*lidx));
                }
            }
        }

        // FIXME: Move `frames` instead of copying them (requires JIT module to be consumable).
        let deoptinfo = DeoptInfo {
            frames: gi.frames().clone(),
            lives: locs,
        };
        // Unwrap is safe since in this architecture usize and i64 have the same size.
        let deoptid = self.deoptinfo.len().try_into().unwrap();
        self.deoptinfo.push(deoptinfo);

        // Clippy points out that `__yk_depot as i64` isn't portable, but since this entire module
        // is x86 only, we don't need to worry about portability.
        #[allow(clippy::fn_to_numeric_cast)]
        {
            dynasm!(self.asm
                ; jmp >check_cond
                ; guard_fail:
                ; mov rdi, [rbp]
                ; mov rsi, QWORD deoptid
                ; mov rdx, rbp
                ; mov rax, QWORD __yk_deopt as i64
                ; call rax
                ; check_cond:
                ; cmp Rb(WR0.code()), inst.expect() as i8 // `as` intentional.
                ; jne <guard_fail
            );
        }
    }

    /// Load an operand into a general purpose register.
    fn load_operand(&mut self, reg: Rq, op: &Operand) {
        match op {
            Operand::Local(li) => self.load_local(reg, *li),
            Operand::Const(c) => self.load_const(reg, *c),
        }
    }

    /// Load an operand into a floating point register.
    fn load_operand_float(&mut self, reg: Rx, op: &Operand) {
        match op {
            Operand::Local(li) => self.load_local_float(reg, *li),
            Operand::Const(ci) => self.load_const_float(reg, *ci),
        }
    }

    /// Load a local variable into the specified general purpose register.
    fn load_local(&mut self, reg: Rq, local: InstIdx) {
        let alloc = match self.m.inst(local) {
            jit_ir::Inst::ProxyConst(c) => match self.m.const_(*c) {
                Const::Int(_, c) => &LocalAlloc::ConstInt(*c),
                _ => todo!(),
            },
            jit_ir::Inst::ProxyInst(_) => todo!(),
            _ => self.ra.allocation(local),
        };
        match alloc {
            LocalAlloc::Stack { frame_off, size: _ } => {
                match i32::try_from(*frame_off) {
                    Ok(foff) => {
                        let size = self.m.inst(local).def_byte_size(self.m);
                        // We use `movzx` where possible to avoid partial register stalls.
                        match size {
                            1 => dynasm!(self.asm; movzx Rq(reg.code()), BYTE [rbp - foff]),
                            2 => dynasm!(self.asm; movzx Rq(reg.code()), WORD [rbp - foff]),
                            4 => dynasm!(self.asm; mov Rd(reg.code()), [rbp - foff]),
                            8 => dynasm!(self.asm; mov Rq(reg.code()), [rbp - foff]),
                            _ => todo!(),
                        }
                    }
                    Err(_) => todo!(),
                }
            }
            LocalAlloc::Register => todo!(),
            LocalAlloc::ConstInt(c) => {
                // FIXME: Adjust for different constant sizes. Requires storing the size in the
                // variant.
                dynasm!(self.asm; mov Rq(reg.code()), QWORD *c as i64);
            }
            LocalAlloc::ConstFloat(_) => todo!(),
        }
    }

    /// Load a local variable into the specified floating point register.
    fn load_local_float(&mut self, reg: Rx, local: InstIdx) {
        let alloc = match self.m.inst(local) {
            jit_ir::Inst::ProxyConst(c) => match self.m.const_(*c) {
                Const::Float(_, c) => &LocalAlloc::ConstFloat(*c),
                _ => todo!(),
            },
            jit_ir::Inst::ProxyInst(_) => todo!(),
            _ => self.ra.allocation(local),
        };
        let size = self.m.inst(local).def_byte_size(self.m);
        match alloc {
            LocalAlloc::Stack { frame_off, size: _ } => match i32::try_from(*frame_off) {
                Ok(foff) => match size {
                    4 => dynasm!(self.asm; movss Rx(reg.code()), [rbp - foff]),
                    8 => dynasm!(self.asm; movsd Rx(reg.code()), [rbp - foff]),
                    _ => todo!(),
                },
                Err(_) => todo!(),
            },
            LocalAlloc::Register => todo!(),
            LocalAlloc::ConstInt(_) => todo!(),
            LocalAlloc::ConstFloat(fv) => imm_float_into_reg(*fv, reg, size, &mut self.asm),
        }
    }

    /// Load a constant into the specified general purpose register.
    fn load_const(&mut self, reg: Rq, cidx: jit_ir::ConstIdx) {
        match self.m.const_(cidx) {
            jit_ir::Const::Float(_tyidx, _x) => todo!(),
            jit_ir::Const::Int(tyidx, x) => {
                let jit_ir::Ty::Integer(width) = self.m.type_(*tyidx) else {
                    panic!()
                };
                // The `as`s are all safe because the IR guarantees that no more than `width` bits
                // are set in the integer i.e. we are only ever truncating zeros.
                match width {
                    1 | 8 => dynasm!(self.asm; mov Rq(reg.code()), *x as i32),
                    16 => dynasm!(self.asm; mov Rw(reg.code()), WORD *x as i16),
                    32 => dynasm!(self.asm; mov Rq(reg.code()), DWORD *x as i32),
                    64 => dynasm!(self.asm; mov Rq(reg.code()), QWORD *x as i64),
                    _ => todo!(),
                }
            }
            jit_ir::Const::Ptr(x) => {
                dynasm!(self.asm; mov Rq(reg.code()), QWORD i64::try_from(*x).unwrap())
            }
        }
    }

    /// Load a constant into the specified floating point register.
    fn load_const_float(&mut self, reg: Rx, cidx: jit_ir::ConstIdx) {
        match self.m.const_(cidx) {
            jit_ir::Const::Float(tyidx, v) => {
                // Unwrap cannot fail. Constant floats are sized.
                imm_float_into_reg(
                    *v,
                    reg,
                    self.m.type_(*tyidx).byte_size().unwrap(),
                    &mut self.asm,
                );
            }
            jit_ir::Const::Int(..) | jit_ir::Const::Ptr(..) => todo!(),
        }
    }

    fn store_local(&mut self, l: &LocalAlloc, reg: Rq, size: usize) {
        match l {
            LocalAlloc::Stack { frame_off, size: _ } => match i32::try_from(*frame_off) {
                Ok(off) => match size {
                    8 => dynasm!(self.asm ; mov [rbp - off], Rq(reg.code())),
                    4 => dynasm!(self.asm ; mov [rbp - off], Rd(reg.code())),
                    2 => dynasm!(self.asm ; mov [rbp - off], Rw(reg.code())),
                    1 => dynasm!(self.asm ; mov [rbp - off], Rb(reg.code())),
                    _ => todo!("{}", size),
                },
                Err(_) => todo!("{}", size),
            },
            LocalAlloc::Register => todo!(),
            LocalAlloc::ConstInt(_) => todo!(),
            LocalAlloc::ConstFloat(_) => todo!(),
        }
    }

    /// Store a value held in a general purpose register into a new local variable.
    fn store_new_local(&mut self, local: InstIdx, reg: Rq) {
        let size = self.m.inst(local).def_byte_size(self.m);
        let l = self.ra.allocate(local, size, &mut self.stack);
        self.store_local(&l, reg, size);
    }

    fn store_local_float(&mut self, l: &LocalAlloc, reg: Rx, size: usize) {
        match l {
            LocalAlloc::Stack { frame_off, size: _ } => match i32::try_from(*frame_off) {
                Ok(off) => match size {
                    8 => dynasm!(self.asm ; movsd [rbp - off], Rx(reg.code())),
                    4 => dynasm!(self.asm ; movss [rbp - off], Rx(reg.code())),
                    _ => todo!("{}", size),
                },
                Err(_) => todo!("{}", size),
            },
            LocalAlloc::Register => todo!(),
            LocalAlloc::ConstInt(_) => todo!(),
            LocalAlloc::ConstFloat(_) => todo!(),
        }
    }

    /// Store a value held in a floating point register into a new local variable.
    fn store_new_local_float(&mut self, local: InstIdx, reg: Rx) {
        let size = self.m.inst(local).def_byte_size(self.m);
        let l = self.ra.allocate(local, size, &mut self.stack);
        self.store_local_float(&l, reg, size);
    }
}

/// Information required by deoptimisation.
#[derive(Debug)]
struct DeoptInfo {
    /// Vector of AOT stackmap IDs.
    frames: Vec<u64>,
    // Vector of live JIT variable locations.
    lives: Vec<LocalAlloc>,
}

#[derive(Debug)]
pub(super) struct X64CompiledTrace {
    /// The executable code itself.
    buf: ExecutableBuffer,
    /// Vector of deopt info, tracked here so they can be freed when the compiled trace is
    /// dropped.
    deoptinfo: Vec<DeoptInfo>,
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

    fn aotvals(&self) -> *const libc::c_void {
        todo!()
    }

    fn guard(&self, _id: crate::compile::GuardId) -> &crate::compile::Guard {
        todo!()
    }

    fn mt(&self) -> &std::sync::Arc<crate::MT> {
        todo!()
    }

    fn hl(&self) -> &std::sync::Weak<parking_lot::Mutex<crate::location::HotLocation>> {
        todo!()
    }
    fn is_last_guard(&self, _id: crate::compile::GuardId) -> bool {
        todo!()
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

#[cfg(test)]
mod tests {
    use super::{CodeGen, X64CodeGen, X64CompiledTrace, STACK_DIRECTION};
    use crate::compile::{
        jitc_yk::{
            codegen::reg_alloc::RegisterAllocator,
            jit_ir::{self, Module},
        },
        CompiledTrace,
    };
    use fm::FMBuilder;
    use regex::Regex;
    use std::sync::Arc;
    use ykaddr::addr::symbol_to_ptr;

    fn test_module() -> jit_ir::Module {
        jit_ir::Module::new_testing()
    }

    /// Test helper to use `fm` to match a disassembled trace.
    fn match_asm(cgo: Arc<X64CompiledTrace>, pattern: &str) {
        let dis = cgo.disassemble().unwrap();

        // Use `{{name}}` to match non-literal strings in tests.
        let ptn_re = Regex::new(r"\{\{.+?\}\}").unwrap();
        let text_re = Regex::new(r"[a-zA-Z0-9\._]+").unwrap();
        // The dissamebler alternates between upper- and lowercase hex, making matching addresses
        // difficult. So just lowercase both pattern and text to avoid tests randomly breaking when
        // addresses change.
        let lowerpattern = pattern.to_lowercase();
        let fmm = FMBuilder::new(&lowerpattern)
            .unwrap()
            .name_matcher(ptn_re, text_re)
            .build()
            .unwrap();

        match fmm.matches(&dis.to_lowercase()) {
            Ok(()) => (),
            Err(e) => panic!("{e}"),
        }
    }

    mod with_spillalloc {
        use super::*;
        use crate::compile::jitc_yk::codegen::reg_alloc::SpillAllocator;

        fn test_with_spillalloc(mod_str: &str, patt_lines: &str) {
            let m = Module::from_str(mod_str);
            match_asm(
                X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                    .unwrap()
                    .codegen()
                    .unwrap()
                    .as_any()
                    .downcast::<X64CompiledTrace>()
                    .unwrap(),
                patt_lines,
            );
        }

        #[test]
        fn cg_load_ptr() {
            test_with_spillalloc(
                "
              entry:
                %0: ptr = load_ti 0
                %1: ptr = load %0
            ",
                "
                ...
                ; %1: ptr = load %0
                ... mov r12, [rbp-0x08]
                ... mov r12, [r12]
                ... mov [rbp-0x10], r12
                ...
                ",
            );
        }

        #[test]
        fn cg_load_i8() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load %0
            ",
                "
                ...
                ; %1: i8 = load %0
                ... movzx r12, byte ptr [rbp-0x01]
                ... movzx r12, byte ptr [r12]
                ... mov [rbp-0x02], r12b
                ...
                ",
            );
        }

        #[test]
        fn cg_load_i32() {
            test_with_spillalloc(
                "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = load %0
            ",
                "
                ...
                ; %1: i32 = Load %0
                ... mov r12d, [rbp-0x04]
                ... mov r12d, [r12]
                ... mov [rbp-0x08], r12d
                ...
                ",
            );
        }

        #[test]
        fn cg_load_const_ptr() {
            test_with_spillalloc(
                "
              entry:
                %0: ptr = load_ti 0
                *%0 = 0x0
            ",
                "
                ...
                ; *%0 = 0x0
                ... mov r12, [rbp-0x08]
                ... mov r13, 0x00
                ... mov [r12], r13
                ...
                ",
            );
        }

        #[test]
        fn cg_ptradd() {
            test_with_spillalloc(
                "
              entry:
                %0: ptr = load_ti 0
                %1: i32 = ptr_add %0, 64
            ",
                "
                ...
                ; %1: ptr = ptr_add %0, 64
                ... mov r12, [rbp-0x08]
                ... add r12, 0x40
                ... mov [rbp-0x10], r12
                ...
                ",
            );
        }

        #[test]
        fn cg_dynptradd() {
            test_with_spillalloc(
                "
              entry:
                %0: ptr = load_ti 0
                %1: i32 = load_ti 8
                %2: ptr = dyn_ptr_add %0, %1, 32
            ",
                "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 32
               ... mov r12d, [rbp-0x0c]
               ... mov r13, [rbp-0x08]
               ... imul r12, r12, 0x20
               ... add r12, r13
               ... mov [rbp-0x18], r12
                ...
                ",
            );
        }

        #[test]
        fn cg_store_ptr() {
            test_with_spillalloc(
                "
              entry:
                %0: ptr = load_ti 0
                %1: ptr = load_ti 8
                *%1 = %0
            ",
                "
                ...
                ; *%1 = %0
                ... mov r12, [rbp-0x10]
                ... mov r13, [rbp-0x08]
                ... mov [r12], r13
                ...
                ",
            );
        }

        #[test]
        fn cg_loadtraceinput_i8() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
            ",
                "
                ...
                ; %0: i8 = load_ti 0
                ... movzx r12, byte ptr [rdi]
                ... mov [rbp-0x01], r12b
                ...
                ",
            );
        }

        #[test]
        fn cg_loadtraceinput_i16_with_offset() {
            test_with_spillalloc(
                "
              entry:
                %0: i16 = load_ti 32
            ",
                "
                ...
                ; %0: i16 = load_ti 32
                ... movzx r12d, word ptr [rdi+0x20]
                ... mov [rbp-0x02], r12w
                ...
                ",
            );
        }

        #[test]
        fn cg_loadtraceinput_many_offset() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = load_ti 2
                %3: i8 = load_ti 3
                %4: ptr = load_ti 8
            ",
                "
                ...
                ; %0: i8 = load_ti 0
                ... movzx r12, byte ptr [rdi]
                ... mov [rbp-0x01], r12b
                ; %1: i8 = load_ti 1
                ... movzx r12, byte ptr [rdi+0x01]
                ... mov [rbp-0x02], r12b
                ; %2: i8 = load_ti 2
                ... movzx r12, byte ptr [rdi+0x02]
                ... mov [rbp-0x03], r12b
                ; %3: i8 = load_ti 3
                ... movzx r12, byte ptr [rdi+0x03]
                ... mov [rbp-0x04], r12b
                ; %4: ptr = load_ti 8
                ... mov r12, [rdi+0x08]
                ... mov [rbp-0x10], r12
                ...
                ",
            );
        }

        #[test]
        fn cg_add_i16() {
            test_with_spillalloc(
                "
              entry:
                %0: i16 = load_ti 0
                %1: i16 = load_ti 1
                %3: i16 = add %0, %1
            ",
                "
                ...
                ; %2: i16 = add %0, %1
                ... movzx r12, word ptr [rbp-0x02]
                ... movzx r13, word ptr [rbp-0x04]
                ... add r12w, r13w
                ... mov [rbp-0x06], r12w
                ...
                ",
            );
        }

        #[test]
        fn cg_add_i64() {
            test_with_spillalloc(
                "
              entry:
                %0: i64 = load_ti 0
                %1: i64 = load_ti 1
                %3: i64 = add %0, %1
            ",
                "
                ...
                ; %2: i64 = add %0, %1
                ... mov r12, [rbp-0x08]
                ... mov r13, [rbp-0x10]
                ... add r12, r13
                ... mov [rbp-0x18], r12
                ...
                ",
            );
        }

        #[test]
        fn cg_call_simple() {
            let sym_addr = symbol_to_ptr("puts").unwrap().addr();
            test_with_spillalloc(
                "
              func_decl puts ()

              entry:
                call @puts()
            ",
                &format!(
                    "
                ...
                ... mov r12, 0x{sym_addr:X}
                ... call r12
                ...
            "
                ),
            );
        }

        #[test]
        fn cg_call_with_args() {
            let sym_addr = symbol_to_ptr("puts").unwrap().addr();
            test_with_spillalloc(
                "
              func_decl puts (i32, i32, i32)

              entry:
                %0: i32 = load_ti 0
                %1: i32 = load_ti 4
                %2: i32 = load_ti 8
                call @puts(%0, %1, %2)
            ",
                &format!(
                    "
                ...
                ; call @puts(%0, %1, %2)
                ... mov r12, 0x{sym_addr:X}
                ... mov edi, [rbp-0x04]
                ... mov esi, [rbp-0x08]
                ... mov edx, [rbp-0x0C]
                ... call r12
                ...
            "
                ),
            );
        }

        #[test]
        fn cg_call_with_different_args() {
            let sym_addr = symbol_to_ptr("puts").unwrap().addr();
            test_with_spillalloc(
                "
              func_decl puts (i8, i16, i32, i64, ptr, i8)

              entry:
                %0: i8 = load_ti 0
                %1: i16 = load_ti 8
                %2: i32 = load_ti 16
                %3: i64 = load_ti 24
                %4: ptr = load_ti 32
                %5: i8 = load_ti 40
                call @puts(%0, %1, %2, %3, %4, %5)
            ",
                &format!(
                    "
                ...
                ; call @puts(%0, %1, %2, %3, %4, %5)
                ... mov r12, 0x{sym_addr:X}
                ... movzx rdi, byte ptr [rbp-0x01]
                ... movzx rsi, word ptr [rbp-0x04]
                ... mov edx, [rbp-0x08]
                ... mov rcx, [rbp-0x10]
                ... mov r8, [rbp-0x18]
                ... movzx r9, byte ptr [rbp-0x19]
                ... call r12
                ...
            "
                ),
            );
        }

        #[should_panic] // until we implement spill args
        #[test]
        fn cg_call_spill_args() {
            test_with_spillalloc(
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
            let sym_addr = symbol_to_ptr("puts").unwrap().addr();
            test_with_spillalloc(
                "
             func_decl puts() -> i32
             entry:
               %0: i32 = call @puts()
            ",
                &format!(
                    "
                ...
                ... mov r12, 0x{sym_addr:X}
                ... call r12
                ... mov [rbp-0x04], eax
                ...
            "
                ),
            );
        }

        #[test]
        fn cg_eq_i64() {
            test_with_spillalloc(
                "
              entry:
                %0: i64 = load_ti 0
                %1: i1 = eq %0, %0
            ",
                "
                ...
                ; %1: i1 = eq %0, %0
                ... mov r12, [rbp-0x08]
                ... mov r13, [rbp-0x08]
                ... cmp r12, r13
                ... setz r12b
                ... mov [rbp-0x09], r12b
                ...
            ",
            );
        }

        #[test]
        fn cg_eq_i8() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i1 = eq %0, %0
            ",
                "
                ...
                ; %1: i1 = eq %0, %0
                ... movzx r12, byte ptr [rbp-0x01]
                ... movzx r13, byte ptr [rbp-0x01]
                ... cmp r12b, r13b
                ... setz r12b
                ... mov [rbp-0x02], r12b
                ...
            ",
            );
        }

        #[test]
        fn cg_guard_true() {
            test_with_spillalloc(
                "
              entry:
                %0: i1 = load_ti 0
                guard true, %0, []
            ",
                "
                ...
                ; guard true, %0, []
                {{vaddr1}} {{off1}}: jmp 0x00000000{{cmpoff}}
                {{vaddr2}} {{failoff}}: mov rdi, [rbp]
                ... mov rsi, 0x00
                ... mov rdx, rbp
                ... mov rax, ...
                ... call rax
                {{vaddr3}} {{cmpoff}}: cmp r12b, 0x01
                {{vaddr4}} {{off4}}: jnz 0x00000000{{failoff}}
                ...
            ",
            );
        }

        #[test]
        fn cg_guard_false() {
            test_with_spillalloc(
                "
              entry:
                %0: i1 = load_ti 0
                guard false, %0, []
            ",
                "
                ...
                ; guard false, %0, []
                {{vaddr1}} {{off1}}: jmp 0x00000000{{cmpoff}}
                {{vaddr2}} {{failoff}}: mov rdi, [rbp]
                ... mov rsi, 0x00
                ... mov rdx, rbp
                ... mov rax, ...
                ... call rax
                {{vaddr3}} {{cmpoff}}: cmp r12b, 0x00
                {{vaddr4}} {{off4}}: jnz 0x00000000{{failoff}}
                ...
            ",
            );
        }

        #[test]
        fn unterminated_trace() {
            test_with_spillalloc(
                "
              entry:
                ",
                "
                ...
                ; Unterminated trace
                {{vaddr}} {{off}}: ud2
                ",
            );
        }

        #[test]
        fn looped_trace_smallest() {
            // FIXME: make the offset and disassembler format hex the same so we can match
            // easier (capitalisation of hex differs).
            test_with_spillalloc(
                "
              entry:
                tloop_start
            ",
                "
                ...
                ; tloop_start:
                ; tloop_backedge:
                {{vaddr}} {{off}}: jmp {{target}}
            ",
            );
        }

        #[test]
        fn looped_trace_bigger() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                tloop_start
                %2: i8 = add %0, %0
            ",
                "
                ...
                ; %0: i8 = load_ti 0
                ...
                ; tloop_start:
                ; %2: i8 = add %0, %0
                ...
                ; tloop_backedge:
                ...: jmp ...
            ",
            );
        }

        #[test]
        fn cg_srem() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = srem %0, %1
            ",
                "
                ...
                ; %2: i8 = srem %0, %1
                ... xor rdx, rdx
                ... movzx rax, byte ptr [rbp-0x01]
                ... movzx r13, byte ptr [rbp-0x02]
                ... idiv r13b
                ... mov [rbp-0x03], dl
                ...
            ",
            );
        }

        #[test]
        fn cg_trunc() {
            test_with_spillalloc(
                "
              entry:
                %0: i32 = load_ti 0
                %1: i8 = trunc %0
            ",
                "
                ...
                ; %0: i32 = load_ti 0
                ...
                ; %1: i8 = trunc %0
                ... mov r12d, [rbp-0x04]
                ... mov [rbp-0x05], r12b
                ...
            ",
            );
        }

        #[test]
        fn cg_select() {
            test_with_spillalloc(
                "
              entry:
                %0: i32 = load_ti 0
                %1: i32 = %0 ? 1i32 : 2i32
            ",
                "
                ...
                ; %1: i32 = %0 ? 1i32 : 2i32
                ... mov r12, 0x01
                ... mov r13d, [rbp-0x04]
                ... mov r14, 0x02
                ... cmp r13b, 0x00
                ... cmovz r12, r14
                ... mov [rbp-0x08], r12d
                ...
            ",
            );
        }

        #[test]
        fn cg_sdiv() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = sdiv %0, %1
            ",
                "
                ...
                ; %2: i8 = sdiv %0, %1
                ... movzx rax, byte ptr [rbp-0x01]
                ... movzx r13, byte ptr [rbp-0x02]
                ... movsx ax, al
                ... idiv r13b
                ... mov [rbp-0x03], al
                ...
            ",
            );
        }

        #[test]
        fn cg_udiv() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = udiv %0, %1
            ",
                "
                ...
                ; %2: i8 = udiv %0, %1
                ... xor rdx, rdx
                ... movzx rax, byte ptr [rbp-0x01]
                ... movzx r13, byte ptr [rbp-0x02]
                ... div r13b
                ... mov [rbp-0x03], al
                ...
            ",
            );
        }

        #[test]
        fn cg_proxyconst() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = 1i8
                %2: i8 = add %0, %1
            ",
                "
                ...
                ; %2: i8 = add %0, 1i8
                ... movzx r12, byte ptr [rbp-0x01]
                ... mov r13, 0x01
                ... add r12b, r13b
                ...
            ",
            );
        }

        #[test]
        fn cg_shl() {
            test_with_spillalloc(
                "
              entry:
                %0: i8 = load_ti 0
                %1: i8 = load_ti 1
                %2: i8 = shl %0, %1
            ",
                "
                ...
                ; %2: i8 = shl %0, %1
                ... movzx r12, byte ptr [rbp-0x01]
                ... movzx rcx, byte ptr [rbp-0x02]
                ... shl r12b, cl
                ... mov [rbp-0x03], r12b
                ...
            ",
            );
        }

        #[test]
        fn cg_sitofp_float() {
            test_with_spillalloc(
                "
              entry:
                %0: i32 = load_ti 0
                %1: float = si_to_fp %0
            ",
                "
                ...
                ; %1: float = si_to_fp %0
                ... mov r12d, [rbp-0x04]
                ... cvtsi2ss xmm0, r12
                ... movss [rbp-0x08], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_sitofp_double() {
            test_with_spillalloc(
                "
              entry:
                %0: i32 = load_ti 0
                %1: double = si_to_fp %0
            ",
                "
                ...
                ; %1: double = si_to_fp %0
                ... mov r12d, [rbp-0x04]
                ... cvtsi2sd xmm0, r12
                ... movsd [rbp-0x10], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_fpext_float_double() {
            test_with_spillalloc(
                "
              entry:
                %0: float = load_ti 0
                %1: double = fp_ext %0
            ",
                "
                ...
                ; %1: double = fp_ext %0
                ... movss xmm0, dword ptr [rbp-0x04]
                ... cvtss2sd xmm0, xmm0
                ... movsd [rbp-0x10], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_fdiv_float() {
            test_with_spillalloc(
                "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: float = fdiv %0, %1
            ",
                "
                ...
                ... movss xmm0, dword ptr [rbp-0x04]
                ... movss xmm1, dword ptr [rbp-0x08]
                ... divss xmm0, xmm1
                ... movss [rbp-0x0c], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_fdiv_double() {
            test_with_spillalloc(
                "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: double = fdiv %0, %1
            ",
                "
                ...
                ... movsd xmm0, qword ptr [rbp-0x08]
                ... movsd xmm1, qword ptr [rbp-0x10]
                ... divsd xmm0, xmm1
                ... movsd [rbp-0x18], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_fadd_float() {
            test_with_spillalloc(
                "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: float = fadd %0, %1
            ",
                "
                ...
                ... movss xmm0, dword ptr [rbp-0x04]
                ... movss xmm1, dword ptr [rbp-0x08]
                ... addss xmm0, xmm1
                ... movss [rbp-0x0c], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_fcmp_float() {
            test_with_spillalloc(
                "
              entry:
                %0: float = load_ti 0
                %1: float = load_ti 1
                %2: i1 = f_ueq %0, %1
            ",
                "
                ...
                ; %2: i1 = f_ueq %0, %1
                ... movss xmm0, dword ptr [rbp-0x04]
                ... movss xmm1, dword ptr [rbp-0x08]
                ... ucomiss xmm0, xmm1
                ... setz r12b
                ... setnp r13b
                ... and r12b, r13b
                ... mov [rbp-0x09], r12b
                ...
                ",
            );
        }

        #[test]
        fn cg_fcmp_double() {
            test_with_spillalloc(
                "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: i1 = f_ueq %0, %1
            ",
                "
                ...
                ; %2: i1 = f_ueq %0, %1
                ... movsd xmm0, qword ptr [rbp-0x08]
                ... movsd xmm1, qword ptr [rbp-0x10]
                ... ucomisd xmm0, xmm1
                ... setz r12b
                ... setnp r13b
                ... and r12b, r13b
                ... mov [rbp-0x11], r12b
                ...
                ",
            );
        }

        #[test]
        fn cg_fadd_double() {
            test_with_spillalloc(
                "
              entry:
                %0: double = load_ti 0
                %1: double = load_ti 1
                %2: double = fadd %0, %1
            ",
                "
                ...
                ... movsd xmm0, qword ptr [rbp-0x08]
                ... movsd xmm1, qword ptr [rbp-0x10]
                ... addsd xmm0, xmm1
                ... movsd [rbp-0x18], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_const_float() {
            test_with_spillalloc(
                "
              entry:
                %0: float = fadd 1.2float, 3.4float
            ",
                "
                ...
                ; %0: float = fadd 1.2float, 3.4float
                ... push r12
                ... mov r12d, 0x3f99999a
                ... movd xmm0, r12d
                ... pop r12
                ... push r12
                ... mov r12d, 0x4059999a
                ... movd xmm1, r12d
                ... pop r12
                ... addss xmm0, xmm1
                ... movss [rbp-0x04], xmm0
                ...
                ",
            );
        }

        #[test]
        fn cg_const_double() {
            test_with_spillalloc(
                "
              entry:
                %0: double = fadd 1.2double, 3.4double
            ",
                "
                ...
                ; %0: double = fadd 1.2double, 3.4double
                ... push r12
                ... mov r12, 0x3ff3333333333333
                ... movq xmm0, r12
                ... pop r12
                ... push r12
                ... mov r12, 0x400b333333333333
                ... movq xmm1, r12
                ... pop r12
                ... addsd xmm0, xmm1
                ... movsd [rbp-0x08], xmm0
                ...
                ",
            );
        }
    }
}
