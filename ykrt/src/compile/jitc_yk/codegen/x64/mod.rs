//! The X64 JIT Code Generator.
//!
//! Conventions used in this module:
//!   * Functions with a `cg_X` prefix generate code for a [jit_ir] construct `X`.
//!   * Helper functions arguments are in order `(<destination>, <source_1>, ... <source_n>)`.
//!
//! Notes:
//!
//!   * The codegen routines try to avoid the use of operations which set 8 or 16-bit
//!     sub-registers. This is for two reasons: a) to simplify the code, but also b) to avoid
//!     partial register stalls in the CPU pipeline. Generally speaking we prefer to use 64-bit
//!     operations, but sometimes we special-case 32-bit operations since they are so common.
//!
//!   * If an object occupies only part of a 64-bit register, then there are no guarantees about
//!     the unused higher-order bits. Codegen routines must be mindful of this and, depending upon
//!     the operation, may be required to (for example) "mask off" or "sign extend into" the
//!     undefined bits in order to compute the correct result.
//!
//! FIXME: the code generator clobbers registers willy-nilly because at the time of writing we have
//! a register allocator that doesn't actually use any registers. Later we will have to audit the
//! backend and insert register save/restore for clobbered registers.

use super::{
    super::{
        int_signs::{SignExtend, Truncate},
        jit_ir::{self, BinOp, FloatTy, Inst, InstIdx, Module, Operand, TraceKind, Ty},
        CompilationError,
    },
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
    location::HotLocation,
    mt::MT,
};
use dynasmrt::{
    components::StaticLabel,
    dynasm,
    x64::{Rq, Rx},
    AssemblyOffset, DynamicLabel, DynasmApi, DynasmError, DynasmLabelApi, ExecutableBuffer,
    Register as dynasmrtRegister,
};
use indexmap::IndexMap;
use parking_lot::Mutex;
use std::{
    assert_matches::debug_assert_matches,
    cell::Cell,
    collections::HashMap,
    error::Error,
    slice,
    sync::{Arc, Weak},
};
use ykaddr::addr::symbol_to_ptr;

mod deopt;
pub(super) mod lsregalloc;
mod rev_analyse;

use deopt::{__yk_deopt, __yk_guardcheck};
use lsregalloc::{LSRegAlloc, RegConstraint};

/// General purpose argument registers as defined by the x64 SysV ABI.
static ARG_GP_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// The GP registers clobbered by a function call in the x64 SysV ABI.
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

/// The FP registers clobbered by a function call in the x64 SysV ABI.
static CALLER_FP_CLOBBER_REGS: [Rx; 16] = [
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
pub(crate) static REG64_BYTESIZE: usize = 8;
static REG64_BITSIZE: usize = REG64_BYTESIZE * 8;
static RBP_DWARF_NUM: u16 = 6;

/// The x64 SysV ABI requires a 16-byte aligned stack prior to any call.
const SYSV_CALL_STACK_ALIGN: usize = 16;

/// To stop us having to say `VarLocation<Register>` everywhere, we use this type alias so that
/// within this `x64` module and its descendants we can just say `VarLocation`.
pub(crate) type VarLocation = super::reg_alloc::VarLocation<Register>;

/// A function that we can put a debugger breakpoint on.
/// FIXME: gross hack.
#[cfg(debug_assertions)]
#[no_mangle]
#[inline(never)]
pub extern "C" fn __yk_break() {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Register {
    GP(Rq), // general purpose
    FP(Rx), // floating point
}

impl VarLocation {
    pub(crate) fn from_yksmp_location(m: &Module, iidx: InstIdx, x: &yksmp::Location) -> Self {
        match x {
            yksmp::Location::Register(0, ..) => VarLocation::Register(Register::GP(Rq::RAX)),
            yksmp::Location::Register(1, ..) => {
                // Since the control point passes the stackmap ID via RDX this case only happens in
                // side-traces.
                VarLocation::Register(Register::GP(Rq::RDX))
            }
            yksmp::Location::Register(2, ..) => VarLocation::Register(Register::GP(Rq::RCX)),
            yksmp::Location::Register(3, ..) => VarLocation::Register(Register::GP(Rq::RBX)),
            yksmp::Location::Register(4, ..) => VarLocation::Register(Register::GP(Rq::RSI)),
            yksmp::Location::Register(5, ..) => VarLocation::Register(Register::GP(Rq::RDI)),
            yksmp::Location::Register(8, ..) => VarLocation::Register(Register::GP(Rq::R8)),
            yksmp::Location::Register(9, ..) => VarLocation::Register(Register::GP(Rq::R9)),
            yksmp::Location::Register(10, ..) => VarLocation::Register(Register::GP(Rq::R10)),
            yksmp::Location::Register(11, ..) => VarLocation::Register(Register::GP(Rq::R11)),
            yksmp::Location::Register(12, ..) => VarLocation::Register(Register::GP(Rq::R12)),
            yksmp::Location::Register(13, ..) => VarLocation::Register(Register::GP(Rq::R13)),
            yksmp::Location::Register(14, ..) => VarLocation::Register(Register::GP(Rq::R14)),
            yksmp::Location::Register(15, ..) => VarLocation::Register(Register::GP(Rq::R15)),
            yksmp::Location::Register(x, ..) if *x >= 17 && *x <= 32 => VarLocation::Register(
                Register::FP(super::x64::lsregalloc::FP_REGS[usize::from(x - 17)]),
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
                let byte_size = m.inst(iidx).def_byte_size(m);
                debug_assert!(byte_size <= 8);
                VarLocation::ConstInt {
                    bits: u32::try_from(byte_size).unwrap() * 8,
                    v: u64::from(*v),
                }
            }
            e => {
                todo!("{:?}", e);
            }
        }
    }
}

impl From<&VarLocation> for yksmp::Location {
    fn from(val: &VarLocation) -> Self {
        match val {
            VarLocation::Stack { frame_off, size } => {
                // A stack location translates is an offset in relation to RBP which has the DWARF
                // number 6.
                yksmp::Location::Indirect(
                    6,
                    -i32::try_from(*frame_off).unwrap(),
                    u16::try_from(*size).unwrap(),
                )
            }
            VarLocation::Direct { frame_off, size } => {
                yksmp::Location::Direct(6, *frame_off, u16::try_from(*size).unwrap())
            }
            VarLocation::Register(reg) => {
                let dwarf = match reg {
                    Register::GP(reg) => match reg {
                        Rq::RAX => 0,
                        Rq::RDX => 1,
                        Rq::RCX => 2,
                        Rq::RBX => 3,
                        Rq::RSI => 4,
                        Rq::RDI => 5,
                        Rq::R8 => 8,
                        Rq::R9 => 9,
                        Rq::R10 => 10,
                        Rq::R11 => 11,
                        Rq::R12 => 12,
                        Rq::R13 => 13,
                        Rq::R14 => 14,
                        Rq::R15 => 15,
                        e => todo!("{:?}", e),
                    },
                    Register::FP(reg) => match reg {
                        Rx::XMM0 => 17,
                        Rx::XMM1 => 18,
                        Rx::XMM2 => 19,
                        Rx::XMM3 => 20,
                        Rx::XMM4 => 21,
                        Rx::XMM5 => 22,
                        Rx::XMM6 => 23,
                        Rx::XMM7 => 24,
                        Rx::XMM8 => 25,
                        Rx::XMM9 => 26,
                        Rx::XMM10 => 27,
                        Rx::XMM11 => 28,
                        Rx::XMM12 => 29,
                        Rx::XMM13 => 30,
                        Rx::XMM14 => 31,
                        Rx::XMM15 => 32,
                    },
                };
                // We currently only use 8 byte registers, so the size is constant. Since these are
                // JIT values there are no extra locations we need to worry about.
                yksmp::Location::Register(dwarf, 8, Vec::new())
            }
            VarLocation::ConstInt { bits, v } => {
                if *bits <= 32 {
                    yksmp::Location::Constant(u32::try_from(*v).unwrap())
                } else {
                    todo!(">32 bit constant")
                }
            }
            e => todo!("{:?}", e),
        }
    }
}

/// A simple front end for the X64 code generator.
pub(crate) struct X64CodeGen;

impl CodeGen for X64CodeGen {
    fn codegen(
        &self,
        m: Module,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
        sp_offset: Option<usize>,
        root_offset: Option<usize>,
        prevguards: Option<Vec<GuardIdx>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        Assemble::new(&m, sp_offset, root_offset)?.codegen(mt, hl, prevguards)
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
    /// The locations of the live variables at the begining of the trace header.
    header_start_locs: Vec<VarLocation>,
    /// The locations of the live variables at the beginning of the trace body.
    body_start_locs: Vec<VarLocation>,
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
    /// The stack pointer offset of the root trace's frame from the base pointer of the interpreter
    /// frame. If this is the root trace, this will be None.
    root_offset: Option<usize>,
    /// The offset after the trace's prologue. This is the re-entry point when returning from
    /// side-traces.
    prologue_offset: AssemblyOffset,
}

impl<'a> Assemble<'a> {
    fn new(
        m: &'a jit_ir::Module,
        sp_offset: Option<usize>,
        root_offset: Option<usize>,
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
                    rec.size - u64::try_from(REG64_BYTESIZE).unwrap()
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
            header_start_locs: Vec::new(),
            body_start_locs: Vec::new(),
            deoptinfo: HashMap::new(),
            comments: Cell::new(IndexMap::new()),
            sp_offset,
            root_offset,
            prologue_offset: AssemblyOffset(0),
        }))
    }

    fn codegen(
        mut self: Box<Self>,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
        prevguards: Option<Vec<GuardIdx>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        if prevguards.is_some() {
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

        self.cg_insts()?;

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

                // Push [GuardIdx]'s of previous guard failures.
                if let Some(ref guards) = prevguards {
                    for gidx in guards.iter().rev() {
                        let g = i64::try_from(usize::from(*gidx)).unwrap();
                        dynasm!(self.asm
                            ; mov r9, QWORD g
                            ; push r9
                        );
                    }
                }
                // Save the pointer to this list.
                dynasm!(self.asm
                    ; mov r8, rsp
                );

                let len = i64::try_from(prevguards.as_ref().map_or(0, |v| v.len())).unwrap();
                // Total alignment caused by pushing the parent guards.
                let mut totalalign = len * 8;

                // Pushing RSI above mis-aligned the stack to 8 bytes, but the calling convetion
                // requires us to be 16 bytes aligned. Unless we accidentally re-aligned it by
                // pushing an uneven amount of previous [GuardIdx]'s, we need to re-align it here.
                if len % 2 == 0 {
                    dynasm!(self.asm
                        ; sub rsp, 8
                    );
                    totalalign += 8;
                }

                // Check whether we need to deoptimise or jump into a side-trace.
                dynasm!(self.asm
                    ; push rsi  // Save `deoptid`.
                    ; push rdx  // Save `gp_regs` pointer.
                    ; push rcx  // Save `fp_regs` pointer.
                    ; push r8   // Save parent guards pointer.
                    ; mov rdi, rsi // Pass `deoptid`.
                    ; mov rsi, r8  // Pass pointer to parent guards.
                    ; mov rdx, QWORD len  // Pass length of parent guards.
                    ; mov rax, QWORD __yk_guardcheck as i64
                    ; call rax
                    ; pop r8
                    ; pop rcx
                    ; pop rdx
                    ; pop rsi
                    ; cmp rax, 0
                    ; je => deopt_label
                );

                // Jump into side-trace. The side-trace takes care of recovering the saved
                // registers.
                dynasm!(self.asm
                    // Remove pushed [GuardIdx]'s from the stack as they are no longer needed.
                    ; add rsp, i32::try_from(totalalign).unwrap()
                    ; jmp rax
                );

                // Deoptimise.
                dynasm!(self.asm; => deopt_label);
                dynasm!(self.asm
                    ; mov rdi, rbp
                    ; mov r9, QWORD len
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
            prevguards,
            sp_offset: self.ra.stack_size(),
            prologue_offset: self.prologue_offset.0,
            entry_vars: self.header_start_locs.clone(),
            hl: Arc::downgrade(&hl),
            comments: self.comments.take(),
            #[cfg(any(debug_assertions, test))]
            gdb_ctx,
        }))
    }

    /// Codegen an instruction.
    fn cg_insts(&mut self) -> Result<(), CompilationError> {
        let mut iter = self.m.iter_skipping_insts();
        let mut next = iter.next();
        let mut in_header = true;
        while let Some((iidx, inst)) = next {
            if self.ra.rev_an.is_inst_tombstone(iidx) {
                next = iter.next();
                continue;
            }
            self.comment_inst(self.asm.offset(), iidx, inst);
            self.ra.expire_regs(iidx);

            match &inst {
                #[cfg(test)]
                jit_ir::Inst::BlackBox(_) => (),
                jit_ir::Inst::Const(_) | jit_ir::Inst::Copy(_) | jit_ir::Inst::Tombstone => {
                    unreachable!();
                }

                jit_ir::Inst::BinOp(i) => self.cg_binop(iidx, i),
                jit_ir::Inst::Param(i) => {
                    // Right now, `Param`s in the body contain dummy values, and shouldn't be
                    // processed.
                    if in_header {
                        self.cg_param(iidx, i);
                    }
                }
                jit_ir::Inst::Load(i) => self.cg_load(iidx, i),
                jit_ir::Inst::PtrAdd(pa_inst) => self.cg_ptradd(iidx, pa_inst),
                jit_ir::Inst::DynPtrAdd(i) => self.cg_dynptradd(iidx, i),
                jit_ir::Inst::Store(i) => self.cg_store(iidx, i),
                jit_ir::Inst::LookupGlobal(i) => self.cg_lookupglobal(iidx, i),
                jit_ir::Inst::Call(i) => self.cg_call(iidx, i)?,
                jit_ir::Inst::IndirectCall(i) => self.cg_indirectcall(iidx, i)?,
                jit_ir::Inst::ICmp(ic_inst) => {
                    next = iter.next();
                    // We have a special optimisation for `ICmp`s iff they're immediately followed
                    // by a `Guard`.
                    if let Some((next_iidx, Inst::Guard(g_inst))) = next {
                        if let Operand::Var(cond_idx) = g_inst.cond(self.m) {
                            // NOTE: If the value of the condition will be used later, we have to
                            // materialise it.
                            if cond_idx == iidx
                                && !self
                                    .ra
                                    .rev_an
                                    .is_inst_var_still_used_after(next_iidx, cond_idx)
                            {
                                self.cg_icmp_guard(iidx, ic_inst, next_iidx, g_inst);
                                next = iter.next();
                                continue;
                            }
                        }
                    }
                    self.cg_icmp(iidx, ic_inst);
                    continue;
                }
                jit_ir::Inst::Guard(i) => self.cg_guard(iidx, i),
                jit_ir::Inst::TraceHeaderStart => self.cg_header_start(),
                jit_ir::Inst::TraceHeaderEnd => {
                    self.cg_header_end(iidx);
                    in_header = false;
                }
                jit_ir::Inst::TraceBodyStart => self.cg_body_start(),
                jit_ir::Inst::TraceBodyEnd => self.cg_body_end(iidx),
                jit_ir::Inst::SidetraceEnd => self.cg_sidetrace_end(iidx),
                jit_ir::Inst::SExt(i) => self.cg_sext(iidx, i),
                jit_ir::Inst::ZExt(i) => self.cg_zext(iidx, i),
                jit_ir::Inst::BitCast(i) => self.cg_bitcast(iidx, i),
                jit_ir::Inst::Trunc(i) => self.cg_trunc(iidx, i),
                jit_ir::Inst::Select(i) => self.cg_select(iidx, i),
                jit_ir::Inst::SIToFP(i) => self.cg_sitofp(iidx, i),
                jit_ir::Inst::FPExt(i) => self.cg_fpext(iidx, i),
                jit_ir::Inst::FCmp(i) => self.cg_fcmp(iidx, i),
                jit_ir::Inst::FPToSI(i) => self.cg_fptosi(iidx, i),
                jit_ir::Inst::FNeg(i) => self.cg_fneg(iidx, i),
            }

            next = iter.next();
        }
        Ok(())
    }

    /// Add a comment to the trace. Note: for instructions, use [Self::comment_inst] which formats
    /// things more appropriately for instructions.
    fn comment(&mut self, off: AssemblyOffset, line: String) {
        self.comments.get_mut().entry(off.0).or_default().push(line);
    }

    /// Add a comment to the trace for a "JIT IR" instruction. This function will format some
    /// instructions differently to the normal trace IR, because this x64 backend has some
    /// non-generic optimisations / modifications.
    fn comment_inst(&mut self, off: AssemblyOffset, iidx: InstIdx, inst: Inst) {
        match inst {
            Inst::Load(_) => {
                if let Some(painst) = self.ra.ptradd(iidx) {
                    self.comment(
                        off,
                        format!(
                            "%{iidx}: {} = load {} + {}",
                            self.m.type_(inst.tyidx(self.m)).display(self.m),
                            painst.ptr(self.m).display(self.m),
                            painst.off()
                        ),
                    );
                    return;
                }
            }
            Inst::Store(sinst) => {
                if let Some(painst) = self.ra.ptradd(iidx) {
                    self.comment(
                        off,
                        format!(
                            "*({} + {}) = {}",
                            painst.ptr(self.m).display(self.m),
                            painst.off(),
                            sinst.val(self.m).display(self.m)
                        ),
                    );
                    return;
                }
            }
            _ => (),
        }
        self.comment(off, inst.display(self.m, iidx).to_string())
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

    /// Sign extend the `from_bits`-sized integer stored in `reg` up to the full size of the 64-bit
    /// register.
    ///
    /// `from_bits` must be between 1 and 64.
    fn sign_extend_to_reg64(&mut self, reg: Rq, from_bits: u8) {
        debug_assert!(from_bits > 0 && from_bits <= 64);
        // For "regularly-sized" integers, we can use movsx to achieve the sign extend and without
        // fear of register stalls.
        match from_bits {
            8 => dynasm!(self.asm; movsx Rq(reg.code()), Rb(reg.code())),
            16 => dynasm!(self.asm; movsx Rq(reg.code()), Rw(reg.code())),
            32 => dynasm!(self.asm; movsx Rq(reg.code()), Rd(reg.code())),
            64 => (), // nothing to do.
            _ => {
                // For "oddly-sized" integers we have to do the sign extend ourselves.
                let shift = REG64_BITSIZE - usize::from(from_bits);
                dynasm!(self.asm
                    ; shl Rq(reg.code()), shift as i8 // shift all the way left.
                    ; sar Rq(reg.code()), shift as i8 // shift back with the correct leading bits.
                );
            }
        }
    }

    /// Zero extend the `from_bits`-sized integer stored in `reg` up to the full size of the 64-bit
    /// register.
    ///
    /// `from_bits` must be between 1 and 64.
    fn zero_extend_to_reg64(&mut self, reg: Rq, from_bits: u8) {
        debug_assert!(from_bits > 0 && from_bits <= 64);
        // For "regularly-sized" integers, we can use movzx to achieve the zero extend and without
        // fear of register stalls.
        match from_bits {
            8 => dynasm!(self.asm; movzx Rq(reg.code()), Rb(reg.code())),
            16 => dynasm!(self.asm; movzx Rq(reg.code()), Rw(reg.code())),
            32 => {
                // mov into a 32-bit register sign-extends up to 64 already.
                dynasm!(self.asm; mov Rd(reg.code()), Rd(reg.code()));
            }
            64 => (), // nothing to do.
            _ => {
                // For "oddly-sized" integers we have to do the zero extend ourselves.
                let shift = REG64_BITSIZE - usize::from(from_bits);
                dynasm!(self.asm
                    ; shl Rq(reg.code()), shift as i8 // shift all the way left.
                    ; shr Rq(reg.code()), shift as i8 // shift back with leading zeros.
                );
            }
        }
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
                // We only optimise the canonicalised case.
                if let Some(v) = self.op_to_sign_ext_i32(&rhs) {
                    let [lhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(lhs)],
                    );
                    match byte_size {
                        8 => dynasm!(self.asm; add Rq(lhs_reg.code()), v),
                        1..=4 => dynasm!(self.asm; add Rd(lhs_reg.code()), v),
                        _ => unreachable!(),
                    }
                } else {
                    let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                    );
                    match byte_size {
                        0 => unreachable!(),
                        1..=8 => {
                            // OK to ignore any undefined high-order bits here.
                            dynasm!(self.asm; add Rq(lhs_reg.code()), Rq(rhs_reg.code()));
                        }
                        _ => todo!(),
                    }
                }
            }
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let byte_size = lhs.byte_size(self.m);
                // We only optimise the canonicalised case.
                if let Some(v) = self.op_to_zero_ext_i32(&rhs) {
                    let [lhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(lhs)],
                    );
                    match inst.binop() {
                        BinOp::And => match byte_size {
                            8 => dynasm!(self.asm; and Rq(lhs_reg.code()), v),
                            1..=4 => dynasm!(self.asm; and Rd(lhs_reg.code()), v),
                            _ => unreachable!(),
                        },
                        BinOp::Or => match byte_size {
                            8 => dynasm!(self.asm; or Rq(lhs_reg.code()), v),
                            1..=4 => dynasm!(self.asm; or Rd(lhs_reg.code()), v),
                            _ => unreachable!(),
                        },
                        BinOp::Xor => match byte_size {
                            8 => dynasm!(self.asm; xor Rq(lhs_reg.code()), v),
                            1..=4 => dynasm!(self.asm; xor Rd(lhs_reg.code()), v),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    }
                } else {
                    let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                    );
                    match inst.binop() {
                        BinOp::And => {
                            match byte_size {
                                0 => unreachable!(),
                                1..=8 => {
                                    // OK to ignore any undefined high-order bits here.
                                    dynasm!(self.asm; and Rq(lhs_reg.code()), Rq(rhs_reg.code()));
                                }
                                _ => todo!(),
                            }
                        }
                        BinOp::Or => {
                            match byte_size {
                                0 => unreachable!(),
                                1..=8 => {
                                    // OK to ignore any undefined high-order bits here.
                                    dynasm!(self.asm; or Rq(lhs_reg.code()), Rq(rhs_reg.code()));
                                }
                                _ => todo!(),
                            }
                        }
                        BinOp::Xor => {
                            match byte_size {
                                0 => unreachable!(),
                                1..=8 => {
                                    // OK to ignore any undefined high-order bits here.
                                    dynasm!(self.asm; xor Rq(lhs_reg.code()), Rq(rhs_reg.code()));
                                }
                                _ => todo!(),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
            BinOp::AShr | BinOp::LShr | BinOp::Shl => {
                // LLVM defines that a poison value is computed if one shifts by >= the bit width
                // of the first operand. This allows us to ignore a lot of seemingly necessary
                // checks in the below. For example we get away with using the 8-bit register `cl`
                // because we don't support any types bigger than 64 bits. If at runtime someone
                // tries to shift a value bigger than `cl` can express, then that's their problem!
                let Ty::Integer(bit_size) = self.m.type_(lhs.tyidx(self.m)) else {
                    unreachable!()
                };
                if let Some(v) = self.op_to_zero_ext_i8(&rhs) {
                    let [lhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(lhs)],
                    );
                    match inst.binop() {
                        BinOp::AShr => match bit_size {
                            0 => unreachable!(),
                            32 => dynasm!(self.asm; sar Rd(lhs_reg.code()), v),
                            1..=64 => {
                                // Ensure we shift in zeros at the most-significant bits.
                                self.sign_extend_to_reg64(
                                    lhs_reg,
                                    u8::try_from(*bit_size).unwrap(),
                                );
                                dynasm!(self.asm; sar Rq(lhs_reg.code()), v);
                            }
                            _ => todo!(),
                        },
                        BinOp::LShr => match bit_size {
                            0 => unreachable!(),
                            32 => dynasm!(self.asm; shr Rd(lhs_reg.code()), v),
                            1..=64 => {
                                // Ensure we shift in zeros at the most-significant bits.
                                self.zero_extend_to_reg64(
                                    lhs_reg,
                                    u8::try_from(*bit_size).unwrap(),
                                );
                                dynasm!(self.asm; shr Rq(lhs_reg.code()), v);
                            }
                            _ => todo!(),
                        },
                        BinOp::Shl => match bit_size {
                            0 => unreachable!(),
                            32 => dynasm!(self.asm; shl Rd(lhs_reg.code()), v),
                            1..=64 => {
                                dynasm!(self.asm; shl Rq(lhs_reg.code()), v);
                            }
                            _ => todo!(),
                        },
                        _ => unreachable!(),
                    }
                } else {
                    let [lhs_reg, _rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [
                            RegConstraint::InputOutput(lhs),
                            // When using a register second operand, it has to be passed in CL.
                            RegConstraint::InputIntoReg(rhs, Rq::RCX),
                        ],
                    );
                    debug_assert_eq!(_rhs_reg, Rq::RCX);
                    match inst.binop() {
                        BinOp::AShr => match bit_size {
                            0 => unreachable!(),
                            32 => dynasm!(self.asm; sar Rd(lhs_reg.code()), cl),
                            1..=64 => {
                                // Ensure we shift in zeros at the most-significant bits.
                                self.sign_extend_to_reg64(
                                    lhs_reg,
                                    u8::try_from(*bit_size).unwrap(),
                                );
                                dynasm!(self.asm; sar Rq(lhs_reg.code()), cl);
                            }
                            _ => todo!(),
                        },
                        BinOp::LShr => match bit_size {
                            0 => unreachable!(),
                            32 => dynasm!(self.asm; shr Rd(lhs_reg.code()), cl),
                            1..=64 => {
                                // Ensure we shift in zeros at the most-significant bits.
                                self.zero_extend_to_reg64(
                                    lhs_reg,
                                    u8::try_from(*bit_size).unwrap(),
                                );
                                dynasm!(self.asm; shr Rq(lhs_reg.code()), cl);
                            }
                            _ => todo!(),
                        },
                        BinOp::Shl => match bit_size {
                            0 => unreachable!(),
                            32 => dynasm!(self.asm; shl Rd(lhs_reg.code()), cl),
                            1..=64 => {
                                dynasm!(self.asm; shl Rq(lhs_reg.code()), cl);
                            }
                            _ => todo!(),
                        },
                        _ => unreachable!(),
                    }
                }
            }
            BinOp::Mul => {
                let byte_size = lhs.byte_size(self.m);
                let [_lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::InputOutputIntoReg(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                        RegConstraint::Clobber(Rq::RDX),
                    ],
                );
                debug_assert_eq!(_lhs_reg, Rq::RAX);
                match byte_size {
                    0 => unreachable!(),
                    1..=8 => {
                        // OK to ignore any undefined high-order bits here.
                        dynasm!(self.asm; mul Rq(rhs_reg.code()));
                    }
                    _ => todo!(),
                }
                // Note that because we are code-genning an unchecked multiply, the higher-order part of
                // the result in RDX is entirely ignored.
            }
            BinOp::SDiv => {
                let Ty::Integer(bit_size) = self.m.type_(lhs.tyidx(self.m)) else {
                    unreachable!()
                };
                let [lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        // 64-bit (or 32-bit) signed division with idiv operates on RDX:RAX
                        // (EDX:EAX) and stores the quotient in RAX (EAX). We ignore the remainder
                        // stored into RDX (EDX).
                        RegConstraint::InputOutputIntoReg(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                        RegConstraint::Clobber(Rq::RDX),
                    ],
                );
                match bit_size {
                    0 => unreachable!(),
                    32 => dynasm!(self.asm
                        ; cdq // Sign extend EAX up to EDX:EAX.
                        ; idiv Rd(rhs_reg.code())
                    ),
                    1..=64 => {
                        self.sign_extend_to_reg64(lhs_reg, u8::try_from(*bit_size).unwrap());
                        self.sign_extend_to_reg64(rhs_reg, u8::try_from(*bit_size).unwrap());
                        dynasm!(self.asm
                            ; cqo // Sign extend RAX up to RDX:RAX.
                            ; idiv Rq(rhs_reg.code())
                        );
                    }
                    _ => todo!(),
                }
            }
            BinOp::SRem => {
                let Ty::Integer(bit_size) = self.m.type_(lhs.tyidx(self.m)) else {
                    unreachable!()
                };
                let [lhs_reg, rhs_reg, _rem_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        // 64-bit (or 32-bit) signed division with idiv operates on RDX:RAX
                        // (EDX:EAX) and stores the remainder in RDX (EDX). We ignore the
                        // quotient stored into RAX (EAX).
                        RegConstraint::InputIntoRegAndClobber(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                        RegConstraint::OutputFromReg(Rq::RDX),
                    ],
                );
                debug_assert_eq!(lhs_reg, Rq::RAX);
                debug_assert_eq!(_rem_reg, Rq::RDX);
                match bit_size {
                    0 => unreachable!(),
                    32 => dynasm!(self.asm
                        ; cdq // Sign extend EAX up to EDX:EAX.
                        ; idiv Rd(rhs_reg.code())
                    ),
                    1..=64 => {
                        self.sign_extend_to_reg64(lhs_reg, u8::try_from(*bit_size).unwrap());
                        self.sign_extend_to_reg64(rhs_reg, u8::try_from(*bit_size).unwrap());
                        dynasm!(self.asm
                            ; cqo // Sign extend RAX up to RDX:RAX.
                            ; idiv Rq(rhs_reg.code())
                        );
                    }
                    _ => todo!(),
                }
            }
            BinOp::Sub => {
                let Ty::Integer(bit_size) = self.m.type_(lhs.tyidx(self.m)) else {
                    unreachable!()
                };
                if let Some(0) = self.op_to_sign_ext_i32(&lhs) {
                    let [rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(rhs)],
                    );
                    match bit_size {
                        0 => unreachable!(),
                        32 => dynasm!(self.asm; neg Rd(rhs_reg.code())),
                        1..=64 => {
                            self.sign_extend_to_reg64(rhs_reg, u8::try_from(*bit_size).unwrap());
                            dynasm!(self.asm; neg Rq(rhs_reg.code()));
                        }
                        _ => todo!(),
                    }
                } else {
                    let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                    );
                    match bit_size {
                        0 => unreachable!(),
                        32 => dynasm!(self.asm; sub Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                        1..=64 => {
                            self.sign_extend_to_reg64(lhs_reg, u8::try_from(*bit_size).unwrap());
                            self.sign_extend_to_reg64(rhs_reg, u8::try_from(*bit_size).unwrap());
                            dynasm!(self.asm; sub Rq(lhs_reg.code()), Rq(rhs_reg.code()));
                        }
                        _ => todo!(),
                    }
                }
            }
            BinOp::UDiv => {
                let Ty::Integer(bit_size) = self.m.type_(lhs.tyidx(self.m)) else {
                    unreachable!()
                };
                let [lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        // 64-bit (or 32-bit) unsigned division with idiv operates on RDX:RAX
                        // (EDX:EAX) and stores the quotient in RAX (EAX). We ignore the remainder
                        // put into RDX (EDX).
                        RegConstraint::InputOutputIntoReg(lhs, Rq::RAX),
                        RegConstraint::Input(rhs),
                        RegConstraint::Clobber(Rq::RDX),
                    ],
                );
                debug_assert_eq!(lhs_reg, Rq::RAX);
                match bit_size {
                    0 => unreachable!(),
                    32 => dynasm!(self.asm
                        ; xor edx, edx
                        ; div Rd(rhs_reg.code())
                    ),
                    1..=64 => {
                        self.zero_extend_to_reg64(lhs_reg, u8::try_from(*bit_size).unwrap());
                        self.zero_extend_to_reg64(rhs_reg, u8::try_from(*bit_size).unwrap());
                        dynasm!(self.asm
                            ; xor rdx, rdx
                            ; div Rq(rhs_reg.code())
                        );
                    }
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
                let byte_size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match byte_size {
                    4 => dynasm!(self.asm; addss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; addsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FMul => {
                let byte_size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match byte_size {
                    4 => dynasm!(self.asm; mulss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; mulsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FSub => {
                let byte_size = lhs.byte_size(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match byte_size {
                    4 => dynasm!(self.asm; subss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    8 => dynasm!(self.asm; subsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            x => todo!("{x:?}"),
        }
    }

    /// Codegen a [jit_ir::ParamInst]. This only informs the register allocator about the
    /// locations of live variables without generating any actual machine code.
    fn cg_param(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::ParamInst) {
        let m = VarLocation::from_yksmp_location(self.m, iidx, self.m.param(inst.paramidx()));
        debug_assert!(self.m.inst(iidx).def_byte_size(self.m) <= REG64_BYTESIZE);
        match m {
            VarLocation::Register(Register::GP(reg)) => {
                // If this register is not used by a "meaningful" (i.e. non-`Guard`-or-`*End`)
                // instruction, we immediately spill it, so that the register allocator has more
                // free registers to play with from the very beginning.
                let mut meaningful = false;
                for iidx in self.ra.rev_an.iter_uses(iidx) {
                    match self.m.inst(iidx) {
                        Inst::Guard(_) | Inst::TraceHeaderEnd | Inst::TraceBodyEnd => (),
                        _ => {
                            meaningful = true;
                            break;
                        }
                    }
                }
                if meaningful {
                    self.ra.force_assign_inst_gp_reg(&mut self.asm, iidx, reg);
                } else {
                    self.ra
                        .force_assign_and_spill_inst_gp_reg(&mut self.asm, iidx, reg);
                }
            }
            VarLocation::Register(Register::FP(reg)) => {
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

    /// Generate code for a [LoadInst], loading from a `register + off`. `off` should only be
    /// non-zero if the [LoadInst] references a [PtrAddInst].
    fn cg_load(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadInst) {
        let (ptr_op, off) = match self.ra.ptradd(iidx) {
            Some(x) => (x.ptr(self.m), x.off()),
            None => (inst.operand(self.m), 0),
        };

        match self.m.type_(inst.tyidx()) {
            Ty::Integer(_) | Ty::Ptr => {
                let [in_reg, out_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        RegConstraint::Input(ptr_op.clone()),
                        RegConstraint::OutputCanBeSameAsInput(ptr_op),
                    ],
                );
                let size = self.m.inst(iidx).def_byte_size(self.m);
                debug_assert!(size <= REG64_BYTESIZE);
                match size {
                    1 => {
                        dynasm!(self.asm ; movzx Rq(out_reg.code()), BYTE [Rq(in_reg.code()) + off])
                    }
                    2 => {
                        dynasm!(self.asm ; movzx Rq(out_reg.code()), WORD [Rq(in_reg.code()) + off])
                    }
                    4 => dynasm!(self.asm ; mov Rd(out_reg.code()), [Rq(in_reg.code()) + off]),
                    8 => dynasm!(self.asm ; mov Rq(out_reg.code()), [Rq(in_reg.code()) + off]),
                    _ => todo!("{}", size),
                };
            }
            Ty::Float(fty) => {
                let ([src_reg], [tgt_reg]) = self.ra.assign_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(ptr_op)],
                    [RegConstraint::Output],
                );
                match fty {
                    FloatTy::Float => {
                        dynasm!(self.asm; movss Rx(tgt_reg.code()), [Rq(src_reg.code()) + off])
                    }
                    FloatTy::Double => {
                        dynasm!(self.asm; movsd Rx(tgt_reg.code()), [Rq(src_reg.code()) + off])
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
        let ptr_op = inst.ptr(self.m);
        let [in_reg, out_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                RegConstraint::Input(ptr_op.clone()),
                RegConstraint::OutputCanBeSameAsInput(ptr_op),
            ],
        );

        dynasm!(self.asm ; lea Rq(out_reg.code()), [Rq(in_reg.code()) + inst.off()]);
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
        // used, that also isn't a worry.
        match inst.elem_size() {
            1 => {
                dynasm!(self.asm; lea Rq(num_elems_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code())])
            }
            2 => {
                dynasm!(self.asm; lea Rq(num_elems_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code()) * 2])
            }
            4 => {
                dynasm!(self.asm; lea Rq(num_elems_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code()) * 4])
            }
            8 => {
                dynasm!(self.asm; lea Rq(num_elems_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code()) * 8])
            }
            _ => {
                if inst.elem_size().is_power_of_two() {
                    let v = i8::try_from(inst.elem_size().ilog2()).unwrap();
                    dynasm!(self.asm; shl Rq(num_elems_reg.code()), v);
                } else {
                    dynasm!(self.asm; imul Rq(num_elems_reg.code()), Rq(num_elems_reg.code()), i32::from(inst.elem_size()));
                }
                // Add the result to the pointer. We make use of addition's commutative property to
                // reverse the "obvious" ordering of registers: doing so allows us not to overwrite
                // ptr_reg.
                dynasm!(self.asm; add Rq(num_elems_reg.code()), Rq(ptr_reg.code()));
            }
        }
    }

    /// Generate code for a [StoreInst], storing it at a `register + off`. `off` should only be
    /// non-zero if the [StoreInst] references a [PtrAddInst].
    fn cg_store(&mut self, iidx: InstIdx, inst: &jit_ir::StoreInst) {
        let (tgt_op, off) = match self.ra.ptradd(iidx) {
            Some(x) => (x.ptr(self.m), x.off()),
            None => (inst.tgt(self.m), 0),
        };

        let val = inst.val(self.m);
        match self.m.type_(val.tyidx(self.m)) {
            Ty::Integer(_) | Ty::Ptr => {
                let byte_size = val.byte_size(self.m);
                if let Some(imm) = self.op_to_zero_ext_immediate(&val) {
                    let [tgt_reg] =
                        self.ra
                            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Input(tgt_op)]);
                    match (imm, byte_size) {
                        (Immediate::I8(v), 1) => {
                            dynasm!(self.asm ; mov BYTE [Rq(tgt_reg.code()) + off], v)
                        }
                        (Immediate::I16(v), 2) => {
                            dynasm!(self.asm ; mov WORD [Rq(tgt_reg.code()) + off], v)
                        }
                        (Immediate::I32(v), 4) => {
                            dynasm!(self.asm ; mov DWORD [Rq(tgt_reg.code()) + off], v)
                        }
                        (Immediate::I32(v), 8) => {
                            dynasm!(self.asm ; mov QWORD [Rq(tgt_reg.code()) + off], v)
                        }
                        _ => todo!(),
                    }
                } else {
                    let [tgt_reg, val_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [RegConstraint::Input(tgt_op), RegConstraint::Input(val)],
                    );
                    match byte_size {
                        1 => dynasm!(self.asm ; mov [Rq(tgt_reg.code()) + off], Rb(val_reg.code())),
                        2 => dynasm!(self.asm ; mov [Rq(tgt_reg.code()) + off], Rw(val_reg.code())),
                        4 => dynasm!(self.asm ; mov [Rq(tgt_reg.code()) + off], Rd(val_reg.code())),
                        8 => dynasm!(self.asm ; mov [Rq(tgt_reg.code()) + off], Rq(val_reg.code())),
                        _ => todo!(),
                    }
                }
            }
            Ty::Float(fty) => {
                let ([tgt_reg], [val_reg]) = self.ra.assign_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(tgt_op)],
                    [RegConstraint::Input(val)],
                );
                match fty {
                    FloatTy::Float => {
                        dynasm!(self.asm ; movss [Rq(tgt_reg.code()) + off], Rx(val_reg.code()));
                    }
                    FloatTy::Double => {
                        dynasm!(self.asm ; movsd [Rq(tgt_reg.code()) + off], Rx(val_reg.code()));
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
        // Calls on x64 with the SysV ABI have complex requirements, and GP registers and FP
        // registers are not treated the same. In essence, we build up constraints for every GP
        // register that's part of the ABI and all FP registers. We start by assuming those
        // registers are clobbered, and gradually refine them with more precise constraints as
        // needed.

        let mut gp_cnstrs = CALLER_CLOBBER_REGS
            .iter()
            .map(|reg| RegConstraint::Clobber(*reg))
            .collect::<Vec<_>>();
        let mut fp_cnstrs = CALLER_FP_CLOBBER_REGS
            .iter()
            .map(|reg| RegConstraint::Clobber(*reg))
            .collect::<Vec<_>>();

        // Deal with inputs.
        let mut gp_regs = ARG_GP_REGS.iter();
        let mut fp_regs = ARG_FP_REGS.iter();
        let mut num_float_args = 0;
        for arg in args.iter() {
            match self.m.type_(arg.tyidx(self.m)) {
                Ty::Float(_) => {
                    let reg = fp_regs.next().unwrap();
                    fp_cnstrs[num_float_args] =
                        RegConstraint::InputIntoRegAndClobber(arg.clone(), *reg);
                    num_float_args += 1;
                }
                Ty::Integer(_) | Ty::Ptr | Ty::Func(_) => {
                    let reg = gp_regs.next().unwrap();
                    let gp_i = CALLER_CLOBBER_REGS.iter().position(|x| x == reg).unwrap();
                    gp_cnstrs[gp_i] = RegConstraint::InputIntoRegAndClobber(arg.clone(), *reg);
                }
                Ty::Void => unreachable!(),
                Ty::Unimplemented(_) => todo!(),
            }
        }

        // Deal with outputs.
        let ret_ty = fty.ret_type(self.m);
        // FIXME: We only support up to register-sized return values at the moment.
        #[cfg(debug_assertions)]
        if !matches!(ret_ty, Ty::Void) {
            debug_assert!(ret_ty.byte_size().unwrap() <= REG64_BYTESIZE);
        }
        match ret_ty {
            Ty::Void => (),
            Ty::Float(_) => {
                let cnstr = match &fp_cnstrs[0] {
                    RegConstraint::InputIntoRegAndClobber(op, _) => {
                        RegConstraint::InputOutputIntoReg(op.clone(), Rx::XMM0)
                    }
                    RegConstraint::Clobber(_) => RegConstraint::OutputFromReg(Rx::XMM0),
                    _ => unreachable!(),
                };
                fp_cnstrs[0] = cnstr;
            }
            Ty::Integer(_) | Ty::Ptr => {
                let rax_i = CALLER_CLOBBER_REGS
                    .iter()
                    .position(|x| *x == Rq::RAX)
                    .unwrap();
                gp_cnstrs[rax_i] = RegConstraint::OutputFromReg(Rq::RAX)
            }
            Ty::Func(_) => todo!(),
            Ty::Unimplemented(_) => todo!(),
        }

        // We now have all the FP constraints, so assign those.
        let fp_cnstrs: [_; 16] = fp_cnstrs.try_into().unwrap();
        let num_float_args = i32::try_from(num_float_args).unwrap();

        // We now have most of the GP constraints, except the call target. We have to handle that
        // differently, depending on whether this is a direct or indirect call.
        match (callee, callee_op) {
            (Some(p), None) => {
                // Direct call

                if !fty.is_vararg() {
                    let _: ([Rq; CALLER_CLOBBER_REGS.len()], [Rx; 16]) = self.ra.assign_regs(
                        &mut self.asm,
                        iidx,
                        gp_cnstrs.try_into().unwrap(),
                        fp_cnstrs,
                    );
                    // rax is considered clobbered, but isn't used to pass an argument, so we can
                    // safely use it for the function pointer.
                    dynasm!(self.asm
                        ; mov rax, QWORD p as i64
                        ; call rax
                    );
                } else {
                    gp_cnstrs.push(RegConstraint::Temporary);
                    let ([.., tmp_reg], _): ([Rq; CALLER_CLOBBER_REGS.len() + 1], [Rx; 16]) =
                        self.ra.assign_regs(
                            &mut self.asm,
                            iidx,
                            gp_cnstrs.try_into().unwrap(),
                            fp_cnstrs,
                        );
                    dynasm!(self.asm
                        ; mov rax, num_float_args
                        ; mov Rq(tmp_reg.code()), QWORD p as i64
                        ; call Rq(tmp_reg.code())
                    );
                }
            }
            (None, Some(op)) => {
                // Indirect call
                gp_cnstrs.push(RegConstraint::Input(op));
                let ([.., op_reg], _): ([Rq; CALLER_CLOBBER_REGS.len() + 1], [Rx; 16]) =
                    self.ra.assign_regs(
                        &mut self.asm,
                        iidx,
                        gp_cnstrs.try_into().unwrap(),
                        fp_cnstrs,
                    );
                if fty.is_vararg() {
                    dynasm!(self.asm; mov rax, num_float_args); // SysV x64 ABI
                }
                dynasm!(self.asm; call Rq(op_reg.code()));
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Return the [VarLocation] an [Operand] relates to.
    fn op_to_var_location(&self, op: Operand) -> VarLocation {
        match op {
            Operand::Var(iidx) => self.ra.var_location(iidx),
            Operand::Const(cidx) => match self.m.const_(cidx) {
                Const::Float(_, v) => VarLocation::ConstFloat(*v),
                Const::Int(tyidx, v) => {
                    let Ty::Integer(bit_size) = self.m.type_(*tyidx) else {
                        panic!()
                    };
                    VarLocation::ConstInt {
                        bits: *bit_size,
                        v: *v,
                    }
                }
                Const::Ptr(v) => VarLocation::ConstPtr(*v),
            },
        }
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i8`, return
    /// it, otherwise return `None`.
    fn op_to_sign_ext_i8(&self, op: &Operand) -> Option<i8> {
        if let Operand::Const(cidx) = op {
            if let Const::Int(tyidx, v) = self.m.const_(*cidx) {
                let Ty::Integer(bit_size) = self.m.type_(*tyidx) else {
                    panic!()
                };
                if *bit_size <= 8 {
                    return Some(v.sign_extend(*bit_size, 8) as i8);
                } else if v.truncate(8).sign_extend(8, 64) == *v {
                    return Some(v.truncate(8) as i8);
                }
            }
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i8`, return
    /// it zero-extended to 8 bits, otherwise return `None`.
    fn op_to_zero_ext_i8(&self, op: &Operand) -> Option<i8> {
        if let Operand::Const(cidx) = op {
            if let Const::Int(tyidx, v) = self.m.const_(*cidx) {
                let Ty::Integer(bit_size) = self.m.type_(*tyidx) else {
                    panic!()
                };
                if *bit_size <= 8 {
                    debug_assert_eq!(v.truncate(*bit_size), *v);
                    return Some(*v as i8);
                } else if v.truncate(8) == *v {
                    return Some(v.truncate(8) as i8);
                }
            }
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i16`, return
    /// it zero extended to 16 bits, otherwise return `None`.
    fn op_to_zero_ext_i16(&self, op: &Operand) -> Option<i16> {
        if let Operand::Const(cidx) = op {
            if let Const::Int(tyidx, v) = self.m.const_(*cidx) {
                let Ty::Integer(bit_size) = self.m.type_(*tyidx) else {
                    panic!()
                };
                if *bit_size <= 16 {
                    debug_assert_eq!(v.truncate(*bit_size), *v);
                    return Some(*v as i16);
                } else if v.truncate(16) == *v {
                    return Some(v.truncate(16) as i16);
                }
            }
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i32`, return it
    /// sign-extended to 32 bits, otherwise return `None`.
    fn op_to_sign_ext_i32(&self, op: &Operand) -> Option<i32> {
        if let Operand::Const(cidx) = op {
            if let Const::Int(tyidx, v) = self.m.const_(*cidx) {
                let Ty::Integer(bit_size) = self.m.type_(*tyidx) else {
                    panic!()
                };
                if *bit_size <= 32 {
                    return Some(v.sign_extend(*bit_size, 32) as i32);
                } else if v.truncate(32).sign_extend(32, 64) == *v {
                    return Some(v.truncate(32) as i32);
                }
            }
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i32`, return it
    /// zero-extended to 32 bits, otherwise return `None`.
    fn op_to_zero_ext_i32(&self, op: &Operand) -> Option<i32> {
        if let Operand::Const(cidx) = op {
            if let Const::Int(tyidx, v) = self.m.const_(*cidx) {
                let Ty::Integer(bit_size) = self.m.type_(*tyidx) else {
                    panic!()
                };
                if *bit_size <= 32 {
                    debug_assert_eq!(v.truncate(*bit_size), *v);
                    return Some(*v as i32);
                } else if v.truncate(32) == *v {
                    return Some(v.truncate(32) as i32);
                }
            }
        }
        None
    }

    /// Return a zero-extended [Immediate] if `op` is a constant and is representable as an x64
    /// immediate. Note this embeds the follow assumptions:
    ///   1. 1 byte constants map to Immediate::I8.
    ///   2. 2 byte constants map to Immediate::I16.
    ///   3. 3 byte constants map to Immediate::I32.
    ///   4. 4 byte constants map to Immediate::I32.
    ///
    /// Note that number (4) breaks the pattern of the (1-3)!
    fn op_to_zero_ext_immediate(&self, op: &Operand) -> Option<Immediate> {
        match op {
            Operand::Const(cidx) => match self.m.const_(*cidx) {
                Const::Float(_, _) => todo!(),
                Const::Int(_, _) => match op.byte_size(self.m) {
                    1 => self.op_to_zero_ext_i8(op).map(Immediate::I8),
                    2 => self.op_to_zero_ext_i16(op).map(Immediate::I16),
                    4 | 8 => self.op_to_zero_ext_i32(op).map(Immediate::I32),
                    _ => todo!(),
                },
                Const::Ptr(_) => self.op_to_zero_ext_i32(op).map(Immediate::I32),
            },
            Operand::Var(_) => None,
        }
    }

    fn cg_cmp_const(&mut self, bit_size: usize, pred: jit_ir::Predicate, lhs_reg: Rq, rhs: i32) {
        match bit_size {
            0 => unreachable!(),
            32 => dynasm!(self.asm; cmp Rd(lhs_reg.code()), rhs),
            8 | 16 | 64 => {
                if pred.signed() {
                    self.sign_extend_to_reg64(lhs_reg, u8::try_from(bit_size).unwrap());
                } else {
                    self.zero_extend_to_reg64(lhs_reg, u8::try_from(bit_size).unwrap());
                }
                dynasm!(self.asm; cmp Rq(lhs_reg.code()), rhs);
            }
            _ => todo!(),
        }
    }

    fn cg_cmp_regs(&mut self, bit_size: usize, pred: jit_ir::Predicate, lhs_reg: Rq, rhs_reg: Rq) {
        match bit_size {
            0 => unreachable!(),
            32 => dynasm!(self.asm; cmp Rd(lhs_reg.code()), Rd(rhs_reg.code())),
            8 | 16 | 64 => {
                if pred.signed() {
                    self.sign_extend_to_reg64(lhs_reg, u8::try_from(bit_size).unwrap());
                    self.sign_extend_to_reg64(rhs_reg, u8::try_from(bit_size).unwrap());
                } else {
                    self.zero_extend_to_reg64(lhs_reg, u8::try_from(bit_size).unwrap());
                    self.zero_extend_to_reg64(rhs_reg, u8::try_from(bit_size).unwrap());
                }
                dynasm!(self.asm; cmp Rq(lhs_reg.code()), Rq(rhs_reg.code()));
            }
            _ => todo!("{bit_size}"),
        }
    }

    /// Optimise an `ICmpInst` iff it's immediately followed by a `GuardInst`. Calling this
    /// function in any other situation will lead to undefined results.
    fn cg_icmp_guard(
        &mut self,
        ic_iidx: InstIdx,
        ic_inst: &jit_ir::ICmpInst,
        g_iidx: InstIdx,
        g_inst: jit_ir::GuardInst,
    ) {
        debug_assert!(!self.ra.rev_an.is_inst_var_still_used_after(g_iidx, ic_iidx));

        // Codegen ICmp
        let (lhs, pred, rhs) = (
            ic_inst.lhs(self.m),
            ic_inst.predicate(),
            ic_inst.rhs(self.m),
        );
        let bit_size = self.m.type_(lhs.tyidx(self.m)).bit_size().unwrap();
        let imm = if pred.signed() {
            self.op_to_sign_ext_i32(&rhs)
        } else {
            self.op_to_zero_ext_i32(&rhs)
        };
        if let Some(v) = imm {
            let [lhs_reg] =
                self.ra
                    .assign_gp_regs(&mut self.asm, ic_iidx, [RegConstraint::Input(lhs)]);
            self.cg_cmp_const(bit_size, pred, lhs_reg, v);
        } else {
            let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                ic_iidx,
                [RegConstraint::Input(lhs), RegConstraint::Input(rhs)],
            );
            self.cg_cmp_regs(bit_size, pred, lhs_reg, rhs_reg);
        }

        // Codegen guard
        self.ra.expire_regs(g_iidx);
        self.comment(
            self.asm.offset(),
            Inst::Guard(g_inst).display(self.m, g_iidx).to_string(),
        );
        let fail_label = self.guard_to_deopt(&g_inst);

        if g_inst.expect() {
            match pred {
                jit_ir::Predicate::Equal => dynasm!(self.asm; jne => fail_label),
                jit_ir::Predicate::NotEqual => dynasm!(self.asm; je => fail_label),
                jit_ir::Predicate::UnsignedGreater => dynasm!(self.asm; jna => fail_label),
                jit_ir::Predicate::UnsignedGreaterEqual => dynasm!(self.asm; jnae => fail_label),
                jit_ir::Predicate::UnsignedLess => dynasm!(self.asm; jnb => fail_label),
                jit_ir::Predicate::UnsignedLessEqual => dynasm!(self.asm; jnbe => fail_label),
                jit_ir::Predicate::SignedGreater => dynasm!(self.asm; jng => fail_label),
                jit_ir::Predicate::SignedGreaterEqual => dynasm!(self.asm; jnge => fail_label),
                jit_ir::Predicate::SignedLess => dynasm!(self.asm; jnl => fail_label),
                jit_ir::Predicate::SignedLessEqual => dynasm!(self.asm; jnle => fail_label),
            }
        } else {
            match pred {
                jit_ir::Predicate::Equal => dynasm!(self.asm; je => fail_label),
                jit_ir::Predicate::NotEqual => dynasm!(self.asm; jne => fail_label),
                jit_ir::Predicate::UnsignedGreater => dynasm!(self.asm; ja => fail_label),
                jit_ir::Predicate::UnsignedGreaterEqual => dynasm!(self.asm; jae => fail_label),
                jit_ir::Predicate::UnsignedLess => dynasm!(self.asm; jb => fail_label),
                jit_ir::Predicate::UnsignedLessEqual => dynasm!(self.asm; jbe => fail_label),
                jit_ir::Predicate::SignedGreater => dynasm!(self.asm; jg => fail_label),
                jit_ir::Predicate::SignedGreaterEqual => dynasm!(self.asm; jge => fail_label),
                jit_ir::Predicate::SignedLess => dynasm!(self.asm; jl => fail_label),
                jit_ir::Predicate::SignedLessEqual => dynasm!(self.asm; jle => fail_label),
            }
        }
    }

    fn cg_icmp(&mut self, iidx: InstIdx, inst: &jit_ir::ICmpInst) {
        let (lhs, pred, rhs) = (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m));
        let bit_size = self.m.type_(lhs.tyidx(self.m)).bit_size().unwrap();
        let imm = if pred.signed() {
            self.op_to_sign_ext_i32(&rhs)
        } else {
            self.op_to_zero_ext_i32(&rhs)
        };
        let lhs_reg = if let Some(v) = imm {
            let [lhs_reg] =
                self.ra
                    .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::InputOutput(lhs)]);
            self.cg_cmp_const(bit_size, pred, lhs_reg, v);
            lhs_reg
        } else {
            let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
            );
            self.cg_cmp_regs(bit_size, pred, lhs_reg, rhs_reg);
            lhs_reg
        };

        // Interpret the flags assignment WRT the predicate.
        //
        // We use a SETcc instruction to do so.
        //
        // Remember, in Intel's tongue:
        //  - "above"/"below" -- unsigned predicate. e.g. `seta`.
        //  - "greater"/"less" -- signed predicate. e.g. `setle`.
        //
        // Note that the equal/not-equal predicates are signedness agnostic.
        match pred {
            jit_ir::Predicate::Equal => dynasm!(self.asm; sete Rb(lhs_reg.code())),
            jit_ir::Predicate::NotEqual => dynasm!(self.asm; setne Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedGreater => dynasm!(self.asm; seta Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedGreaterEqual => dynasm!(self.asm; setae Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedLess => dynasm!(self.asm; setb Rb(lhs_reg.code())),
            jit_ir::Predicate::UnsignedLessEqual => dynasm!(self.asm; setbe Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedGreater => dynasm!(self.asm; setg Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedGreaterEqual => dynasm!(self.asm; setge Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedLess => dynasm!(self.asm; setl Rb(lhs_reg.code())),
            jit_ir::Predicate::SignedLessEqual => dynasm!(self.asm; setle Rb(lhs_reg.code())),
        }
    }

    fn cg_fcmp(&mut self, iidx: InstIdx, inst: &jit_ir::FCmpInst) {
        let (lhs, pred, rhs) = (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m));
        let size = lhs.byte_size(self.m);
        let ([tgt_reg], [lhs_reg, rhs_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Output],
            [RegConstraint::Input(lhs), RegConstraint::Input(rhs)],
        );

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
            jit_ir::FloatPredicate::OrderedGreater | jit_ir::FloatPredicate::UnorderedGreater => {
                dynasm!(self.asm; seta Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::OrderedGreaterEqual => {
                dynasm!(self.asm; setae Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::OrderedLess => dynasm!(self.asm; setb Rb(tgt_reg.code())),
            jit_ir::FloatPredicate::OrderedLessEqual => dynasm!(self.asm; setbe Rb(tgt_reg.code())),
            jit_ir::FloatPredicate::False
            | jit_ir::FloatPredicate::OrderedNotEqual
            | jit_ir::FloatPredicate::Ordered
            | jit_ir::FloatPredicate::Unordered
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
    fn write_jump_vars(&mut self, iidx: InstIdx) {
        let (tgt_vars, src_ops) = match self.m.tracekind() {
            TraceKind::HeaderOnly => (self.header_start_locs.clone(), self.m.trace_header_end()),
            TraceKind::HeaderAndBody => (self.body_start_locs.clone(), self.m.trace_body_end()),
            TraceKind::Sidetrace(sti) => (
                Arc::clone(sti)
                    .as_any()
                    .downcast::<YkSideTraceInfo<Register>>()
                    .unwrap()
                    .entry_vars
                    .clone(),
                self.m.trace_header_end(),
            ),
        };
        // If we pass in `None` use `self.loop_start_locs` instead. We need to do this since we
        // can't pass in `&self.loop_start_locs` directly due to borrowing restrictions.
        let mut gp_regs = lsregalloc::GP_REGS
            .iter()
            .map(|_| RegConstraint::None)
            .collect::<Vec<_>>();
        let mut fp_regs = lsregalloc::FP_REGS
            .iter()
            .map(|_| RegConstraint::None)
            .collect::<Vec<_>>();
        for (i, op) in src_ops.iter().enumerate() {
            // FIXME: This is completely broken: see the FIXME later.
            let op = op.unpack(self.m);
            let src = self.op_to_var_location(op.clone());
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
                        VarLocation::Register(Register::GP(reg)) => match size_dst {
                            8 => dynasm!(self.asm;
                                mov QWORD [rbp - i32::try_from(off_dst).unwrap()], Rq(reg.code())
                            ),
                            4 => dynasm!(self.asm;
                                mov DWORD [rbp - i32::try_from(off_dst).unwrap()], Rd(reg.code())
                            ),
                            _ => todo!(),
                        },
                        VarLocation::ConstInt { bits, v } => match bits {
                            32 => dynasm!(self.asm;
                                mov DWORD [rbp - i32::try_from(off_dst).unwrap()], v as i32
                            ),
                            8 => dynasm!(self.asm;
                                mov BYTE [rbp - i32::try_from(off_dst).unwrap()], v as i8),
                            x => todo!("{x}"),
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
                            4 => dynasm!(self.asm;
                                push rax;
                                mov eax, DWORD [rbp - i32::try_from(off_src).unwrap()];
                                mov DWORD [rbp - i32::try_from(off_dst).unwrap()], eax;
                                pop rax
                            ),
                            e => todo!("{:?}", e),
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
                VarLocation::Register(reg) => match reg {
                    Register::GP(r) => {
                        gp_regs[usize::from(r.code())] = RegConstraint::InputIntoReg(op.clone(), r);
                    }
                    Register::FP(r) => {
                        fp_regs[usize::from(r.code())] = RegConstraint::InputIntoReg(op.clone(), r);
                    }
                },
                _ => todo!(),
            }
        }

        let _: (
            [Rq; lsregalloc::GP_REGS.len()],
            [Rx; lsregalloc::FP_REGS.len()],
        ) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            gp_regs.try_into().unwrap(),
            fp_regs.try_into().unwrap(),
        );
    }

    fn cg_body_end(&mut self, iidx: InstIdx) {
        debug_assert_matches!(self.m.tracekind(), TraceKind::HeaderAndBody);
        // Loop the JITted code if the `tloop_start` label is present (not relevant for IR created
        // by a test or a side-trace).
        let label = StaticLabel::global("tloop_start");
        match self.asm.labels().resolve_static(&label) {
            Ok(_) => {
                // Found the label, emit a jump to it.
                self.write_jump_vars(iidx);
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
                #[cfg(not(test))]
                panic!("unterminated trace in non-unit-test");
            }
            Err(e) => {
                // Any other error suggests something has gone quite wrong. Just crash.
                panic!("{}", e.to_string())
            }
        }
    }

    fn cg_sidetrace_end(&mut self, iidx: InstIdx) {
        match self.m.tracekind() {
            TraceKind::Sidetrace(sti) => {
                let sti = Arc::clone(sti)
                    .as_any()
                    .downcast::<YkSideTraceInfo<Register>>()
                    .unwrap();
                // The end of a side-trace. Map live variables of this side-trace to the entry variables of
                // the root parent trace, then jump to it.
                self.write_jump_vars(iidx);
                self.ra.align_stack(SYSV_CALL_STACK_ALIGN);

                dynasm!(self.asm
                    // Reset rsp to the root trace's frame.
                    ; mov rsp, rbp
                    ; sub rsp, i32::try_from(sti.root_offset()).unwrap()
                    ; mov rdi, QWORD sti.root_addr() as i64
                    // We can safely use RDI here, since the root trace won't expect live variables in this
                    // register since it's being used as an argument to the control point.
                    ; jmp rdi);
            }
            TraceKind::HeaderOnly | TraceKind::HeaderAndBody => panic!(),
        }
    }

    fn cg_header_start(&mut self) {
        debug_assert_eq!(self.header_start_locs.len(), 0);
        // Remember the locations of the live variables at the beginning of the trace. When we
        // re-enter the trace from a side-trace, we need to write the live variables back into
        // these same locations.
        for var in self.m.trace_header_start() {
            let loc = match var.unpack(self.m) {
                Operand::Var(iidx) => self.ra.var_location(iidx),
                _ => panic!(),
            };
            self.header_start_locs.push(loc);
        }
        match self.m.tracekind() {
            TraceKind::HeaderOnly => {
                dynasm!(self.asm; ->tloop_start:);
            }
            TraceKind::HeaderAndBody => {
                dynasm!(self.asm; ->reentry:);
            }
            TraceKind::Sidetrace(_) => todo!(),
        }
        self.prologue_offset = self.asm.offset();
    }

    fn cg_header_end(&mut self, iidx: InstIdx) {
        match self.m.tracekind() {
            TraceKind::HeaderOnly => {
                self.write_jump_vars(iidx);
                dynasm!(self.asm; jmp ->tloop_start);
            }
            TraceKind::HeaderAndBody => {
                // FIXME: This is a bit of a roundabout way of doing things. Especially, since it means
                // that the [ParamInst]s in the trace body are just placeholders. While, since a recent
                // change, the register allocator makes sure the values automatically end up in the
                // [VarLocation]s expected by the loop start, this only works for registers right now. We
                // can extend this to spill locations as well, but won't be able to do so for variables
                // that have become constants during the trace header. So we will always have to either
                // update the [ParamInst]s of the trace body, which isn't ideal since it requires the
                // [Module] the be mutable. Or we do what we do below just for constants.
                let varlocs = self
                    .m
                    .trace_header_end()
                    .iter()
                    .map(|pop| self.op_to_var_location(pop.unpack(self.m)))
                    .collect::<Vec<_>>();
                // Reset the register allocator before priming it with information about the trace body
                // inputs.
                self.ra.reset(varlocs.as_slice());
                for (i, op) in self.m.trace_body_start().iter().enumerate() {
                    // By definition these can only be variables.
                    let iidx = match op.unpack(self.m) {
                        Operand::Var(iidx) => iidx,
                        _ => panic!(),
                    };
                    let varloc = varlocs[i];

                    // Write the varlocations from the head jump to the body start.
                    // FIXME: This is copied verbatim from `cg_param` and can be reused.
                    match varloc {
                        VarLocation::Register(Register::GP(reg)) => {
                            self.ra.force_assign_inst_gp_reg(&mut self.asm, iidx, reg);
                        }
                        VarLocation::Register(Register::FP(reg)) => {
                            self.ra.force_assign_inst_fp_reg(iidx, reg);
                        }
                        VarLocation::Direct { frame_off, size: _ } => {
                            self.ra.force_assign_inst_direct(iidx, frame_off);
                        }
                        VarLocation::Stack { frame_off, size: _ } => {
                            self.ra.force_assign_inst_indirect(
                                iidx,
                                i32::try_from(frame_off).unwrap(),
                            );
                        }
                        VarLocation::ConstInt { bits, v } => {
                            self.ra.assign_const(iidx, bits, v);
                        }
                        e => panic!("{:?}", e),
                    }
                }
            }
            TraceKind::Sidetrace(_) => panic!(),
        }
    }

    fn cg_body_start(&mut self) {
        debug_assert_matches!(self.m.tracekind(), &TraceKind::HeaderAndBody);
        debug_assert_eq!(self.body_start_locs.len(), 0);
        // Remember the locations of the live variables at the beginning of the trace loop. When we
        // loop back around here we need to write the live variables back into these same
        // locations.
        for var in self.m.trace_body_start() {
            let loc = self.op_to_var_location(var.unpack(self.m));
            self.body_start_locs.push(loc);
        }
        dynasm!(self.asm; ->tloop_start:);
    }

    fn cg_sext(&mut self, iidx: InstIdx, sinst: &jit_ir::SExtInst) {
        let src_val = sinst.val(self.m);
        let src_type = self.m.type_(src_val.tyidx(self.m));
        let Ty::Integer(src_bitsize) = src_type else {
            unreachable!(); // must be an integer
        };

        let dest_type = self.m.type_(sinst.dest_tyidx());
        let Ty::Integer(dest_bitsize) = dest_type else {
            unreachable!(); // must be an integer
        };

        if *dest_bitsize <= u32::try_from(REG64_BITSIZE).unwrap() {
            if *src_bitsize == 64 {
                // The 64 bit registers are implicitly sign extended.
                self.ra
                    .assign_gp_pass_through(&mut self.asm, iidx, sinst.val(self.m));
            } else {
                let [reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(sinst.val(self.m))],
                );
                self.sign_extend_to_reg64(reg, u8::try_from(*src_bitsize).unwrap());
            }
        } else {
            todo!("{} {}", src_bitsize, dest_bitsize);
        }
    }

    fn cg_zext(&mut self, iidx: InstIdx, zinst: &jit_ir::ZExtInst) {
        let src_type = self.m.type_(zinst.val(self.m).tyidx(self.m));
        let src_bitsize = src_type.bit_size().unwrap();
        let dest_type = self.m.type_(zinst.dest_tyidx());
        let dest_bitsize = dest_type.bit_size().unwrap();

        if dest_bitsize <= REG64_BITSIZE {
            if src_bitsize == 32 || src_bitsize == 64 {
                // The 32 and 64 bit registers are implicitly zero extended on x64.
                self.ra
                    .assign_gp_pass_through(&mut self.asm, iidx, zinst.val(self.m));
            } else {
                let [reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(zinst.val(self.m))],
                );

                self.zero_extend_to_reg64(reg, u8::try_from(src_bitsize).unwrap());
            }
        } else {
            todo!("{} {}", src_bitsize, dest_bitsize);
        }
    }

    fn cg_bitcast(&mut self, iidx: InstIdx, inst: &jit_ir::BitCastInst) {
        let src_type = self.m.type_(inst.val(self.m).tyidx(self.m));
        let dest_type = self.m.type_(inst.dest_tyidx());

        match (src_type, dest_type) {
            (jit_ir::Ty::Float(_), jit_ir::Ty::Float(_)) => {
                todo!();
            }
            (jit_ir::Ty::Float(_), _gp_ty) => {
                todo!();
            }
            (gp_ty, jit_ir::Ty::Float(_)) => {
                let ([src_reg], [tgt_reg]) = self.ra.assign_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(inst.val(self.m))],
                    [RegConstraint::Output],
                );
                // unwrap safe: IR would be invalid otherwise.
                match gp_ty.byte_size().unwrap() {
                    4 => dynasm!(self.asm; cvtsi2ss Rx(tgt_reg.code()), Rd(src_reg.code())),
                    8 => dynasm!(self.asm; cvtsi2sd Rx(tgt_reg.code()), Rq(src_reg.code())),
                    _ => todo!(),
                }
            }
            (_gp_ty1, _gp_ty2) => todo!(),
        }
    }

    fn cg_sitofp(&mut self, iidx: InstIdx, inst: &jit_ir::SIToFPInst) {
        let ([src_reg], [tgt_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Input(inst.val(self.m))],
            [RegConstraint::Output],
        );

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

        let ([tgt_reg], [src_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Output],
            [RegConstraint::Input(from_val)],
        );

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

        // This instruction takes an integer and truncates it to a smaller one. We do nothing,
        // other than assume that the now-unused higher-order bits are undefined.
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
        debug_assert_eq!(
            self.m
                .type_(inst.cond(self.m).tyidx(self.m))
                .bit_size()
                .unwrap(),
            1
        );

        dynasm!(self.asm ; bt Rq(cond_reg.code()), 0);
        dynasm!(self.asm ; cmovnc Rq(true_reg.code()), Rq(false_reg.code()));
    }

    fn guard_to_deopt(&mut self, inst: &jit_ir::GuardInst) -> DynamicLabel {
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
        fail_label
    }

    fn cg_fneg(&mut self, iidx: InstIdx, inst: &jit_ir::FNegInst) {
        let val = inst.val(self.m);
        let ty = self.m.type_(val.tyidx(self.m));

        // There is no dedicated instruction for negating the value in an XMM register, so we flip
        // the sign bit manually. It's a bit of a dance since you can't XORPS with an immediate
        // float.
        let ([tmpi_reg], [io_reg, tmpf_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [RegConstraint::Temporary],
            [RegConstraint::InputOutput(val), RegConstraint::Temporary],
        );
        match ty {
            jit_ir::Ty::Float(jit_ir::FloatTy::Float) => {
                dynasm!(self.asm
                    ; mov Rd(tmpi_reg.code()), DWORD 0x80000000u32 as i32 // cast intentional
                    ; movd Rx(tmpf_reg.code()), Rd(tmpi_reg.code())
                    ; xorps Rx(io_reg.code()), Rx(tmpf_reg.code())
                );
            }
            jit_ir::Ty::Float(jit_ir::FloatTy::Double) => {
                dynasm!(self.asm
                    ; mov Rq(tmpi_reg.code()), QWORD 0x8000000000000000u64 as i64 // cast intentional
                    ; movq Rx(tmpf_reg.code()), Rq(tmpi_reg.code())
                    ; xorpd Rx(io_reg.code()), Rx(tmpf_reg.code())
                );
            }
            _ => {
                // This bytecode only operates on floating point values.
                panic!();
            }
        }
    }

    fn cg_guard(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::GuardInst) {
        let fail_label = self.guard_to_deopt(inst);
        let cond = inst.cond(self.m);
        let [reg] = self
            .ra
            .assign_gp_regs(&mut self.asm, iidx, [RegConstraint::Input(cond)]);
        dynasm!(self.asm ; bt Rq(reg.code()), 0);
        if inst.expect() {
            dynasm!(self.asm ; jnb =>fail_label);
        } else {
            dynasm!(self.asm ; jb =>fail_label);
        }
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
    /// [GuardIdx]'s of all failing guards leading up to this trace.
    prevguards: Option<Vec<GuardIdx>>,
    /// Stack pointer offset from the base pointer of interpreter frame as defined in
    /// [YkSideTraceInfo::sp_offset].
    sp_offset: usize,
    /// The instruction offset after the trace's prologue. Later this will also include one
    /// iteration of the trace.
    prologue_offset: usize,
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

    fn sidetraceinfo(
        &self,
        root_ctr: Arc<dyn CompiledTrace>,
        gidx: GuardIdx,
    ) -> Arc<dyn SideTraceInfo> {
        let root_ctr = root_ctr.as_any().downcast::<X64CompiledTrace>().unwrap();
        // FIXME: Can we reference these instead of copying them, e.g. by passing in a reference to
        // the `CompiledTrace` and `gidx` or better a reference to `DeoptInfo`?
        let deoptinfo = &self.deoptinfo[&usize::from(gidx)];
        let lives = deoptinfo
            .live_vars
            .iter()
            .map(|(a, l)| (a.clone(), l.into()))
            .collect();
        let callframes = deoptinfo.inlined_frames.clone();

        // Calculate the address inside the root trace we want side-traces to jump. Since the
        // side-trace finishes at the control point we need to re-enter via the trace header and
        // cannot jump back directly into the trace body.
        // FIXME: Check if RPython has found a solution to this (if there is any).
        let root_addr = unsafe { root_ctr.entry().add(root_ctr.prologue_offset) };

        // Pass along [GuardIdx]'s of previous guard failures and add this guard failure's
        // [GuardIdx] to the list.
        let guards = if let Some(v) = &self.prevguards {
            let mut v = v.clone();
            v.push(gidx);
            v
        } else {
            vec![gidx]
        };

        Arc::new(YkSideTraceInfo {
            bid: deoptinfo.bid.clone(),
            guards,
            lives,
            callframes,
            root_addr: RootTracePtr(root_addr),
            root_offset: root_ctr.sp_offset,
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

    fn disassemble(&self, with_addrs: bool) -> Result<String, Box<dyn Error>> {
        AsmPrinter::new(&self.buf, &self.comments, with_addrs).to_string()
    }
}

/// Disassembles emitted code for testing and debugging purposes.
struct AsmPrinter<'a> {
    buf: &'a ExecutableBuffer,
    comments: &'a IndexMap<usize, Vec<String>>,
    /// When true, instruction offset and address are included in the output.
    with_addrs: bool,
}

impl<'a> AsmPrinter<'a> {
    fn new(
        buf: &'a ExecutableBuffer,
        comments: &'a IndexMap<usize, Vec<String>>,
        with_addrs: bool,
    ) -> Self {
        Self {
            buf,
            comments,
            with_addrs,
        }
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
            if self.with_addrs {
                out.push(format!(
                    "{:016x} {:08x}: {}",
                    (bptr as u64) + off,
                    off,
                    istr
                ));
            } else {
                out.push(istr.to_string());
            }
        }
        Ok(out.join("\n"))
    }
}

/// A representation of an x64 immediate, suitable for use in x64 instructions.
///
/// Note that the integer values inside may be zero or sign-extended depending on the construction
/// of an instance of this enum.
enum Immediate {
    I8(i8),
    I16(i16),
    I32(i32),
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
        jitc_yk::jit_ir::{self, Inst, Module, ParamIdx, TraceKind},
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
    fn match_asm(cgo: Arc<X64CompiledTrace>, ptn: &str, full_asm: bool) {
        let dis = cgo.disassemble(full_asm).unwrap();

        // The disassembler alternates between upper- and lowercase hex, making matching addresses
        // difficult. So just lowercase both pattern and text to avoid tests randomly breaking when
        // addresses change.
        let ptn = ptn.to_lowercase();
        match fmatcher(&ptn).matches(&dis.to_lowercase()) {
            Ok(()) => (),
            Err(e) => panic!("{e}"),
        }
    }

    fn codegen_and_test(mod_str: &str, patt_lines: &str, full_asm: bool) {
        let m = Module::from_str(mod_str);
        let mt = MT::new().unwrap();
        let hl = HotLocation {
            kind: HotLocationKind::Tracing,
            tracecompilation_errors: 0,
        };
        match_asm(
            Assemble::new(&m, None, None)
                .unwrap()
                .codegen(mt, Arc::new(Mutex::new(hl)), None)
                .unwrap()
                .as_any()
                .downcast::<X64CompiledTrace>()
                .unwrap(),
            patt_lines,
            full_asm,
        );
    }

    #[test]
    fn cg_load_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: ptr = load %0
                black_box %1
            ",
            "
                ...
                ; %1: ptr = load %0
                mov r.64.x, [rbx]
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_load_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                %1: i8 = load %0
                black_box %1
            ",
            "
                ...
                ; %1: i8 = load %0
                movzx r.64.x, byte ptr [rbx]
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_load_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param 0
                %1: i32 = load %0
                black_box %1
            ",
            "
                ...
                ; %1: i32 = Load %0
                mov r.32.x, [r.64.x]
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_load_const_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                *%0 = 0x0
            ",
            "
                ...
                ; *%0 = 0x0
                mov r.64.x, 0x00
                mov [r.64.y], r.64.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_ptradd() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = ptr_add %0, 64
                black_box %1
            ",
            "
                ...
                ; %1: ptr = ptr_add %0, 64
                lea r.64.x, [r.64._+0x40]
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_ptradd_load() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: ptr = param 1
                %2: ptr = ptr_add %0, 64
                %3: i64 = load %2
                %4: ptr = ptr_add %1, 32
                %5: i64 = load %4
                %6: ptr = ptr_add %4, 1
                black_box %3
                black_box %5
                black_box %6
            ",
            "
                ...
                ; %1: ...
                ; %3: i64 = load %0 + 64
                mov r.64.x, [rbx+{{_}}]
                ; %4: ptr = ptr_add %1, 32
                lea r.64.y, [r.64.z+0x20]
                ; %5: i64 = load %1 + 32
                mov r.64._, [r.64.z+0x20]
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_ptradd_store() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: ptr = param 1
                %2: ptr = ptr_add %0, 64
                *%2 = 1i8
                %4: ptr = ptr_add %1, 32
                %5: i64 = load %4
                *%4 = 2i8
                black_box %5
            ",
            "
                ...
                ; *(%0 + 64) = 1i8
                mov byte ptr [rbx+{{_}}], 0x01
                ; %5: i64 = load %1 + 32
                mov r.64.y, [r.64.x+0x20]
                ; *(%1 + 32) = 2i8
                mov byte ptr [r.64.x+0x20], 0x02
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_dynptradd() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = param 1
                %2: ptr = dyn_ptr_add %0, %1, 1
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 1
                lea r.64.x, [r.64._+r.64.x*1]
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = param 1
                %2: ptr = dyn_ptr_add %0, %1, 2
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 2
                lea r.64.x, [r.64._+r.64.x*2]
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = param 1
                %2: ptr = dyn_ptr_add %0, %1, 4
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 4
                lea r.64.x, [r.64._+r.64.x*4]
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = param 1
                %2: ptr = dyn_ptr_add %0, %1, 5
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 5
                imul r.64.x, r.64.x, 0x05
                add r.64.x, r.64._
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = param 1
                %2: ptr = dyn_ptr_add %0, %1, 16
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 16
                shl r.64.x, 0x04
                add r.64.x, r.64._
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: i32 = param 1
                %2: ptr = dyn_ptr_add %0, %1, 77
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 77
                imul r.64.x, r.64.x, 0x4d
                add r.64.x, r.64._
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_store_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                %1: ptr = param 1
                *%1 = %0
            ",
            "
                ...
                ; %0: ptr = param ...
                ; %1: ptr = param ...
                ; *%1 = %0
                mov [r.64.x], r.64.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_store_consts() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param 0
                *%0 = 1i8
                *%0 = 2i16
                *%0 = 3i32
                *%0 = 4i64
            ",
            "
                ...
                ; %0: ptr = param ...
                ; *%0 = 1i8
                mov byte ptr [r.64.x], 0x01
                ; *%0 = 2i16
                mov word ptr [r.64.x], 0x02
                ; *%0 = 3i32
                mov dword ptr [r.64.x], 0x03
                ; *%0 = 4i64
                mov qword ptr [r.64.x], 0x04
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_add_i16() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i16 = add %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: i16 = add %0, %1
                add r.64.x, r.64.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_add_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param 0
                %1: i64 = param 1
                %2: i64 = add %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: i64 = add %0, %1
                add r.64.x, r.64.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_add_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param 0
                %1: i64 = add %0, 1i64
                black_box %1
            ",
            "
                ...
                ; %1: i64 = add %0, 1i64
                add r.64.x, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_add_minus_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param 0
                %1: i64 = add %0, 18446744073709551615i64
                black_box %1
            ",
            "
                ...
                ; %1: i64 = add %0, 18446744073709551615i64
                add r.64.x, 0xffffffffffffffff
                ...
                ",
            false,
        );
        // note: disassembler sign-extended the immediate when displaying it.
    }

    #[test]
    fn cg_const_add_i32max_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param 0
                %1: i64 = add %0, 2147483647i64
                black_box %1
            ",
            "
                ...
                ; %1: i64 = add %0, 2147483647i64
                add r.64.x, 0x7fffffff
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_add_i32max_plus_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param 0
                %1: i64 = add %0, 2147483648i64
                black_box %1
            ",
            "
                ...
                ; %1: i64 = add %0, 2147483648i64
                mov r.64.x, 0x80000000
                add r.64.y, r.64.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_add_one_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param 0
                %1: i32 = add %0, 1i32
                black_box %1
            ",
            "
                ...
                ; %1: i32 = add %0, 1i32
                add r.32.x, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_add_minus_one_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param 0
                %1: i32 = add %0, 4294967295i32
                black_box %1
            ",
            "
                ...
                ; %1: i32 = add %0, 4294967295i32
                add r.32.x, 0xffffffff
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_and() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i1 = param 6
                %7: i16 = and %0, %1
                %8: i32 = and %2, %3
                %9: i63 = and %4, %5
                %10: i1 = and %6, 1i1
                black_box %7
                black_box %8
                black_box %9
                black_box %10
            ",
            "
                ...
                ; %7: i16 = and %0, %1
                and r.64.a, r.64.b
                ; %8: i32 = and %2, %3
                and r.64.c, r.64.d
                ; %9: i63 = and %4, %5
                and r.64.e, r.64.f
                ; %10: i1 = and %6, 1i1
                and r.32._, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_ashr() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i32 = param 1
                %2: i63 = param 2
                %3: i16 = ashr %0, 1i16
                %4: i32 = ashr %1, 2i32
                %5: i63 = ashr %2, 3i63
                black_box %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %3: i16 = ashr %0, 1i16
                ...
                movsx r.64.a, r.16.a
                sar r.64.a, 0x01
                ; %4: i32 = ashr %1, 2i32
                ...
                sar r.32.c, 0x02
                ; %5: i63 = ashr %2, 3i63
                ...
                shl r.64.e, 0x01
                sar r.64.e, 0x01
                sar r.64.e, 0x03
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_lshr() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i32 = param 1
                %2: i63 = param 2
                %3: i16 = lshr %0, 1i16
                %4: i32 = lshr %1, 2i32
                %5: i63 = lshr %2, 3i63
                black_box %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %3: i16 = lshr %0, 1i16
                ...
                movzx r.64.a, r.16.a
                shr r.64.a, 0x01
                ; %4: i32 = lshr %1, 2i32
                ...
                shr r.32.c, 0x02
                ; %5: i63 = lshr %2, 3i63
                ...
                shl r.64.e, 0x01
                shr r.64.e, 0x01
                shr r.64.e, 0x03
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_shl() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i32 = param 1
                %2: i63 = param 2
                %3: i16 = shl %0, 1i16
                %4: i32 = shl %1, 2i32
                %5: i63 = shl %2, 3i63
                %6: i32 = shl %1, %4
                black_box %3
                black_box %4
                black_box %5
                black_box %6
            ",
            "
                ...
                ; %3: i16 = shl %0, 1i16
                ...
                shl r.64.a, 0x01
                ; %4: i32 = shl %1, 2i32
                ...
                shl r.32.a, 0x02
                ; %5: i63 = shl %2, 3i63
                shl r.64.b, 0x03
                ; %6: i32 = shl %1, %4
                shl r.32.b, cl
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_mul() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i16 = mul %0, %1
                %7: i32 = mul %2, %3
                %8: i63 = mul %4, %5
                black_box %6
                black_box %7
                black_box %8
            ",
            "
                ...
                ; %6: i16 = mul %0, %1
                mov rax, ...
                mul r.64.a
                ; %7: i32 = mul %2, %3
                ......
                mul r.64.b
                ; %8: i63 = mul %4, %5
                ......
                mul r.64.c
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_or() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i1 = param 6
                %7: i16 = or %0, %1
                %8: i32 = or %2, %3
                %9: i63 = or %4, %5
                %10: i1 = or %6, 1i1
                black_box %7
                black_box %8
                black_box %9
                black_box %10
            ",
            "
                ...
                ; %7: i16 = or %0, %1
                or r.64.a, r.64.b
                ; %8: i32 = or %2, %3
                or r.64.c, r.64.d
                ; %9: i63 = or %4, %5
                or r.64.e, r.64.f
                ; %10: i1 = or %6, 1i1
                or r.32._, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_sdiv() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i16 = sdiv %0, %1
                %7: i32 = sdiv %2, %3
                %8: i63 = sdiv %4, %5
                black_box %6
                black_box %7
                black_box %8
            ",
            "
                ...
                ; %6: i16 = sdiv %0, %1
                ...
                movsx rax, ax
                movsx r.64.a, r.16.a
                cqo
                idiv r.64.a
                ; %7: i32 = sdiv %2, %3
                ...
                cdq
                idiv r.32.b
                ; %8: i63 = sdiv %4, %5
                ...
                shl rax, 0x01
                sar rax, 0x01
                shl r.64.c, 0x01
                sar r.64.c, 0x01
                cqo
                idiv r.64.c
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_srem() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i16 = srem %0, %1
                %7: i32 = srem %2, %3
                %8: i63 = srem %4, %5
                black_box %6
                black_box %7
                black_box %8
            ",
            "
                ...
                ; %6: i16 = srem %0, %1
                ...
                movsx rax, ax
                movsx r.64.a, r.16.a
                cqo
                idiv r.64.a
                ; %7: i32 = srem %2, %3
                ...
                cdq
                idiv r.32.b
                ; %8: i63 = srem %4, %5
                ...
                shl rax, 0x01
                sar rax, 0x01
                shl r.64.c, 0x01
                sar r.64.c, 0x01
                cqo
                idiv r.64.c
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_sub() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i16 = sub %0, %1
                %7: i32 = sub %2, %3
                %8: i63 = sub %4, %5
                %9: i32 = sub 0i32, %7
                black_box %6
                black_box %7
                black_box %8
                black_box %9
            ",
            "
                ...
                ; %6: i16 = sub %0, %1
                movsx r.64.a, r.16.a
                movsx r.64.b, r.16.b
                sub r.64.a, r.64.b
                ; %7: i32 = sub %2, %3
                sub r.32.c, r.32.d
                ; %8: i63 = sub %4, %5
                shl r.64.e, 0x01
                sar r.64.e, 0x01
                shl r.64.f, 0x01
                sar r.64.f, 0x01
                sub r.64.e, r.64.f
                ; %9: i32 = sub 0i32, %7
                ......
                neg r.32.c
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_xor() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i1 = param 6
                %7: i16 = xor %0, %1
                %8: i32 = xor %2, %3
                %9: i63 = xor %4, %5
                %10: i1 = xor %6, 1i1
                black_box %7
                black_box %8
                black_box %9
                black_box %10
            ",
            "
                ...
                ; %7: i16 = xor %0, %1
                xor r.64.a, r.64.b
                ; %8: i32 = xor %2, %3
                xor r.64.c, r.64.d
                ; %9: i63 = xor %4, %5
                xor r.64.e, r.64.f
                ; %10: i1 = xor %6, 1i1
                xor r.32._, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_udiv() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i16 = param 1
                %2: i32 = param 2
                %3: i32 = param 3
                %4: i63 = param 4
                %5: i63 = param 5
                %6: i16 = udiv %0, %1
                %7: i32 = udiv %2, %3
                %8: i63 = udiv %4, %5
                black_box %6
                black_box %7
                black_box %8
            ",
            "
                ...
                ; %6: i16 = udiv %0, %1
                ...
                movzx rax, ax
                movzx r.64.a, r.16.a
                xor rdx, rdx
                div r.64.a
                ; %7: i32 = udiv %2, %3
                ...
                xor edx, edx
                div r.32.b
                ; %8: i63 = udiv %4, %5
                ...
                shl rax, 0x01
                shr rax, 0x01
                shl r.64.c, 0x01
                shr r.64.c, 0x01
                xor rdx, rdx
                div r.64.c
                ...
                ",
            false,
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
                mov rax, 0x{sym_addr:X}
                call rax
                ...
            "
            ),
            false,
        );
    }

    #[test]
    fn cg_call_with_args() {
        let sym_addr = symbol_to_ptr("puts").unwrap().addr();
        codegen_and_test(
            "
              func_decl puts (i32, i32, i32)

              entry:
                %0: i32 = param 0
                %1: i32 = param 1
                %2: i32 = param 2
                call @puts(%0, %1, %2)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2)
                ...
                mov rdx, r.64.x
                mov rdi, r.64.y
                mov r.64.tgt, 0x{sym_addr:X}
                call r.64.tgt
                ...
            "
            ),
            false,
        );
    }

    #[test]
    fn cg_call_with_different_args() {
        let sym_addr = symbol_to_ptr("puts").unwrap().addr();
        codegen_and_test(
            "
              func_decl puts (i8, i16, ptr)

              entry:
                %0: i8 = param 0
                %1: i16 = param 1
                %2: ptr = param 2
                call @puts(%0, %1, %2)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2)
                ...
                mov rdx, r.64.x
                mov rdi, r.64.y
                mov r.64.tgt, 0x{sym_addr:X}
                call r.64.tgt
                ...
            "
            ),
            false,
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
            false,
        );
    }

    #[test]
    fn cg_call_ret() {
        codegen_and_test(
            "
             func_decl puts() -> i32
             entry:
               %0: i32 = call @puts()
            ",
            "
                ...
                ; %0: i32 = call @puts()
                mov r.64.x, ...
                call r.64.x
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_eq() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i32 = param 1
                header_start [%0, %1]
                %3: i1 = eq %0, %0
                %4: i1 = eq %1, %1
                black_box %3
                black_box %4
                header_end [%0, %1]
            ",
            "
                ...
                ; %3: i1 = eq %0, %0
                ......
                ......
                movzx r.64.a, r.16.a
                movzx r.64.b, r.16.b
                cmp r.64.a, r.64.b
                setz r.8._
                ...
                ; %4: i1 = eq %1, %1
                ......
                ......
                cmp r.32.c, r.32.d
                setz r.8._
                ; black_box %3
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_sext() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i32 = param 1
                %2: i63 = param 2
                %3: i32 = sext %0
                %4: i64 = sext %1
                %5: i64 = sext %2
                black_box %3
                black_box %4
                black_box %5
                ",
            "
                ...
                ; %3: i32 = sext %0
                movsx r.64.a, r.16.a
                ; %4: i64 = sext %1
                movsxd r.64.b, r.32.b
                ; %5: i64 = sext %2
                shl r.64.c, 0x01
                sar r.64.c, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_zext() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param 0
                %1: i32 = param 1
                %2: i63 = param 2
                %3: i32 = zext %0
                %4: i64 = zext %1
                %5: i64 = zext %2
                black_box %3
                black_box %4
                black_box %5
                ",
            "
                ...
                ; %3: i32 = zext %0
                movzx r.64.a, r.16.a
                ; %4: i64 = zext %1
                ...
                ; %5: i64 = zext %2
                shl r.64.c, 0x01
                shr r.64.c, 0x01
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_bitcast() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param 0
                %1: i32 = param 1
                %2: double = bitcast %0
                %3: float = bitcast %1
                black_box %2
                black_box %3
                ",
            "
            ...
            ; %2: double = bitcast %0
            cvtsi2sd fp.128.x, r.64.x
            ; %3: float = bitcast %1
            cvtsi2ss fp.128.y, r.32.y
            ...
            ",
            false,
        );
    }

    #[test]
    fn cg_guard_true() {
        codegen_and_test(
            "
              entry:
                %0: i1 = param 0
                guard true, %0, []
            ",
            "
                ...
                ; guard true, %0, [] ; ...
                movzx r.64._, byte ptr ...
                bt r.64._, 0x00
                jnb 0x...
                ...
                ; deopt id for guard 0
                push rsi
                mov rsi, 0x00
                jmp ...
                ; call __yk_deopt
                ...
                mov rdi, rbp
                mov r9, 0x...
                mov rax, 0x...
                call rax
            ",
            false,
        );
    }

    #[test]
    fn cg_guard_false() {
        codegen_and_test(
            "
              entry:
                %0: i1 = param 0
                guard false, %0, []
            ",
            "
                ...
                ; guard false, %0, [] ; ...
                movzx r.64._, byte ptr ...
                bt r.64._, 0x00
                jb 0x...
                ...
                ; deopt id for guard 0
                push rsi
                mov rsi, 0x00
                jmp ...
                ; call __yk_deopt
                ...
                mov rdi, rbp
                mov r9, 0x...
                mov rax, 0x...
                call rax
            ",
            false,
        );
    }

    #[test]
    fn cg_guard_const() {
        codegen_and_test(
            "
              entry:
                %0: i1 = param 0
                %1: i8 = 10i8
                %2: i8 = 32i8
                %3: i8 = add %1, %2
                guard false, %0, [%0, 10i8, 32i8, 42i8]
            ",
            "
                ...
                ; guard false, %0, [0:%0_0: %0, 0:%0_1: 10i8, 0:%0_2: 32i8, 0:%0_3: 42i8] ; trace_gidx 0 safepoint_id 0
                movzx r.64._, byte ptr ...
                bt r.64._, 0x00
                jb 0x...
                ...
                ; deopt id for guard 0
                push rsi
                mov rsi, 0x00
                jmp ...
                ; call __yk_deopt
                ...
                mov rdi, rbp
                mov r9, 0x...
                mov rax, 0x...
                call rax
            ",
            false,
        );
    }

    #[test]
    fn cg_icmp_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                %1: i1 = eq %0, 3i8
                black_box %1
            ",
            "
                ...
                ; %1: i1 = eq %0, 3i8
                ...
                movzx r.64.x, r.8._
                cmp r.64.x, 0x03
                setz r.8.x
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_icmp_guard() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                %1: i1 = eq %0, 3i8
                guard true, %1, []
            ",
            "
                ...
                ; %1: i1 = eq %0, 3i8
                movzx r.64.x, r.8._
                cmp r.64.x, 0x03
                ; guard true, %1, [] ; ...
                jnz 0x...
                ...
            ",
            false,
        );
    }

    /// Check we don't optimise icmp+guard if the result of the icmp is needed later.
    #[test]
    fn cg_icmp_guard_reused() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                %1: i1 = eq %0, 3i8
                guard true, %1, []
                %3: i8 = sext %1
                black_box %3
            ",
            "
                ...
                ; %1: i1 = eq %0, 3i8
                movzx r.64.x, r.8._
                cmp r.64.x, 0x03
                setz bl
                ; guard true, %1, [] ; ...
                bt r.64._, 0x00
                jnb 0x...
                ; %3: i8 = sext %1
                shl rbx, 0x3f
                sar rbx, 0x3f
                ; black_box %3
                ...
            ",
            false,
        );
    }

    #[should_panic]
    #[test]
    fn unterminated_trace() {
        codegen_and_test(
            "
              entry:
                 header_end []
                ",
            "
                ...
                ud2
                ",
            false,
        );
    }

    #[test]
    fn looped_trace_smallest() {
        // FIXME: make the offset and disassembler format hex the same so we can match
        // easier (capitalisation of hex differs).
        codegen_and_test(
            "
              entry:
                header_start []
                header_end []
            ",
            "
                ...
                ; header_start []
                ; header_end []
                jmp {{target}}
            ",
            false,
        );
    }

    #[test]
    fn looped_trace_bigger() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                header_start [%0]
                %2: i8 = add %0, %0
                black_box %2
                header_end [%0]
            ",
            "
                ...
                ; %0: i8 = param ...
                ...
                ; header_start [%0]
                ; %2: i8 = add %0, %0
                {{_}} {{off}}: ...
                ...
                ; header_end [%0]
                ...
                {{_}} {{_}}: jmp 0x00000000{{off}}
            ",
            true,
        );
    }

    #[test]
    fn cg_srem_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                %1: i8 = param 1
                %2: i8 = srem %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: i8 = srem %0, %1
                mov rax, r.64.y
                movsx rax, al
                movsx rsi, sil
                cqo
                idiv r.64.x
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_srem_i56() {
        codegen_and_test(
            "
              entry:
                %0: i56 = param 0
                %1: i56 = param 1
                %2: i56 = srem %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: i56 = srem %0, %1
                mov rax, r.64.y
                shl rax, 0x08
                sar rax, 0x08
                shl rsi, 0x08
                sar rsi, 0x08
                cqo
                idiv r.64.x
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_trunc() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param 0
                %1: i8 = trunc %0
                %2: i8 = trunc %0
                %3: i8 = add %2, %1
                black_box %3
            ",
            "
                ...
                ; %0: i32 = param ...
                ...
                ; %1: i8 = trunc %0
                mov r.64.x, r.64.y
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_select() {
        codegen_and_test(
            "
              entry:
                %0: i1 = param 0
                %1: i32 = %0 ? 1i32 : 2i32
                black_box %1
            ",
            "
                ...
                ; %1: i32 = %0 ? 1i32 : 2i32
                mov r.64.x, 0x01
                mov r.64.y, 0x02
                bt r.64.z, 0x00
                cmovnb r.64.x, r.64.y
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                %1: i8 = 1i8
                %2: i8 = add %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: i8 = add %0, 1i8
                add r.32.y, 0x01
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_sitofp_float() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param 0
                %1: float = si_to_fp %0
                black_box %1
            ",
            "
                ...
                ; %1: float = si_to_fp %0
                cvtsi2ss fp.128.x, r.32.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_sitofp_double() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param 0
                %1: double = si_to_fp %0
                black_box %1
            ",
            "
                ...
                ; %1: double = si_to_fp %0
                cvtsi2sd fp.128.x, r.32.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fpext_float_double() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: double = fp_ext %0
                black_box %1
            ",
            "
                ...
                ; %0: float = param ...
                ; %1: double = fp_ext %0
                cvtss2sd fp.128.x, fp.128.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fptosi_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: i32 = fp_to_si %0
                black_box %1
            ",
            "
                ...
                ; %1: i32 = fp_to_si %0
                cvttss2si r.64.x, fp.128.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fptosi_double() {
        codegen_and_test(
            "
              entry:
                %0: double = param 0
                %1: i32 = fp_to_si %0
                black_box %1
            ",
            "
                ...
                ; %1: i32 = fp_to_si %0
                cvttsd2si r.64.x, fp.128.x
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fdiv_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: float = param 1
                %2: float = fdiv %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: float = fdiv %0, %1
                divss fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fdiv_double() {
        codegen_and_test(
            "
              entry:
                %0: double = param 0
                %1: double = param 1
                %2: double = fdiv %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: double = fdiv %0, %1
                divsd fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fadd_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: float = param 1
                %2: float = fadd %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: float = fadd %0, %1
                addss fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fadd_double() {
        codegen_and_test(
            "
              entry:
                %0: double = param 0
                %1: double = param 1
                %2: double = fadd %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: double = fadd %0, %1
                addsd fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fsub_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: float = param 1
                %2: float = fsub %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: float = fsub %0, %1
                subss fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fsub_double() {
        codegen_and_test(
            "
              entry:
                %0: double = param 0
                %1: double = param 1
                %2: double = fsub %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: double = fsub %0, %1
                subsd fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fmul_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: float = param 1
                %2: float = fmul %0, %1
                %3: float = fmul %1, %1
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: float = fmul %0, %1
                mulss fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fmul_double() {
        codegen_and_test(
            "
              entry:
                %0: double = param 0
                %1: double = param 1
                %2: double = fmul %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: double = fmul %0, %1
                mulsd fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fcmp_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: float = param 1
                %2: i1 = f_ueq %0, %1
                %3: i1 = f_ugt %0, %1
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: i1 = f_ueq %0, %1
                ucomiss fp.128.x, fp.128.y
                setz r.8.x
                setnp r.8.y
                and r.8.x, r.8.y
                ; %3: i1 = f_ugt %0, %1
                ucomiss fp.128.x, fp.128.y
                setnbe r.8._
                setnp r.8._
                and r.8._, r.8._
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_fcmp_double() {
        codegen_and_test(
            "
              entry:
                %0: double = param 0
                %1: double = param 1
                %2: i1 = f_ueq %0, %1
                %3: i1 = f_ugt %0, %1
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: i1 = f_ueq %0, %1
                ucomisd fp.128.x, fp.128.y
                setz r.8.x
                setnp r.8.y
                and r.8.x, r.8.y
                ; %3: i1 = f_ugt %0, %1
                ucomisd fp.128.x, fp.128.y
                setnbe r.8._
                setnp r.8._
                and r.8._, r.8._
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_float() {
        codegen_and_test(
            "
              entry:
                %0: float = fadd 1.2float, 3.4float
                black_box %0
            ",
            "
                ...
                ; %0: float = fadd 1.2float, 3.4float
                ...
                mov r.32.x, 0x3f99999a
                movd fp.128.x, r.32.x
                ...
                mov r.32.x, 0x4059999a
                movd fp.128.y, r.32.x
                ...
                addss fp.128.x, fp.128.y
                ...
                ",
            false,
        );
    }

    #[test]
    fn cg_const_double() {
        codegen_and_test(
            "
              entry:
                %0: double = fadd 1.2double, 3.4double
                black_box %0
            ",
            "
                ...
                ; %0: double = fadd 1.2double, 3.4double
                ...
                mov r.64.x, 0x3ff3333333333333
                ...
                mov r.64.x, 0x400b333333333333
                ...
                ",
            false,
        );
    }

    #[test]
    fn loop_jump_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param 0
                header_start [%0]
                %2: i8 = 42i8
                header_end [%2]
            ",
            "
                ...
                ; %0: i8 = param ...
                ...
                ; header_start [%0]
                ; header_end [42i8]
                mov byte ptr [rbp-0x01], 0x2a
                jmp ...
            ",
            false,
        );
    }

    #[test]
    fn cg_fneg() {
        codegen_and_test(
            "
              entry:
                %0: float = param 0
                %1: double = param 1
                %2: float = fneg %0
                %3: double = fneg %1
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: float = fneg %0
                mov r.32.x, 0x80000000
                movd fp.128.y, r.32.x
                xorps fp.128.z, fp.128.y
                ; %3: double = fneg %1
                mov r.64.x, 0x8000000000000000
                movq fp.128.y, r.64.x
                xorpd fp.128.a, fp.128.y
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_aliasing_params() {
        let mut m = jit_ir::Module::new(TraceKind::HeaderOnly, 0, 0).unwrap();

        // Create two trace paramaters whose locations alias.
        let loc = yksmp::Location::Register(13, 1, [].into());
        m.push_param(loc.clone());
        let pinst1: Inst =
            jit_ir::ParamInst::new(ParamIdx::try_from(0).unwrap(), m.int8_tyidx()).into();
        m.push(pinst1.clone()).unwrap();
        m.push_param(loc);
        let pinst2: Inst =
            jit_ir::ParamInst::new(ParamIdx::try_from(1).unwrap(), m.int8_tyidx()).into();
        m.push(pinst2.clone()).unwrap();
        let op1 = m.push_and_make_operand(pinst1).unwrap();
        let op2 = m.push_and_make_operand(pinst2).unwrap();

        let add_inst = jit_ir::BinOpInst::new(op1, jit_ir::BinOp::Add, op2);
        m.push(add_inst.into()).unwrap();

        let mt = MT::new().unwrap();
        let hl = HotLocation {
            kind: HotLocationKind::Tracing,
            tracecompilation_errors: 0,
        };

        Assemble::new(&m, None, None)
            .unwrap()
            .codegen(mt, Arc::new(Mutex::new(hl)), None)
            .unwrap()
            .as_any()
            .downcast::<X64CompiledTrace>()
            .unwrap();
    }
}
