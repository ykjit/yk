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
//!   * When a value is in a register, we make no guarantees about what the upper bits are set to.
//!     You must sign or zero extend at all points that these values are important.

#[cfg(any(debug_assertions, test))]
use crate::compile::jitc_yk::gdb::{self, GdbCtx};
use crate::{
    aotsmp::AOT_STACKMAPS,
    compile::{
        CompilationError, CompiledTrace, Guard, GuardId,
        jitc_yk::{
            CodeGen, YkSideTraceInfo,
            aot_ir::{self, DeoptSafepoint},
            arbbitint::ArbBitInt,
            jit_ir::{
                self, BinOp, Const, FloatTy, GuardInfoIdx, HasGuardInfo, IndirectCallIdx,
                InlinedFrame, Inst, InstIdx, Module, Operand, TraceKind, Ty,
            },
        },
    },
    location::HotLocation,
    mt::{MT, TraceId},
};
use dynasmrt::{
    AssemblyOffset, DynamicLabel, DynasmApi, DynasmError, DynasmLabelApi, ExecutableBuffer,
    Register as dynasmrtRegister,
    components::StaticLabel,
    dynasm,
    x64::{Rq, Rx},
};
use indexmap::IndexMap;
use page_size;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    cell::Cell,
    debug_assert_matches,
    error::Error,
    slice,
    sync::{Arc, Weak, atomic::fence},
};
use ykaddr::addr::symbol_to_ptr;

mod deopt;
pub(super) mod lsregalloc;
mod rev_analyse;

use deopt::{__yk_deopt, __yk_ret_from_trace};
use lsregalloc::{GPConstraint, GuardSnapshot, LSRegAlloc, RegConstraint, RegExtension};

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
static REG64_BITSIZE: u32 = 64;
static RBP_DWARF_NUM: u16 = 6;

/// The x64 SysV ABI requires a 16-byte aligned stack prior to any call.
const SYSV_CALL_STACK_ALIGN: usize = 16;

/// To stop us having to say `VarLocation<Register>` everywhere, we use this type alias so that
/// within this `x64` module and its descendants we can just say `VarLocation`.
pub(crate) type VarLocation = super::reg_alloc::VarLocation<Register>;

/// The lock used when patching a side-trace into a parent trace.
static LK_PATCH: Mutex<()> = Mutex::new(());

/// Returns the offset of the given thread local in relation to the segment register `fs`. At JIT
/// runtime we use this offset to calculate the absolute address of the thread local for each
/// thread.
fn get_tls_offset(name: &std::ffi::CString) -> i32 {
    let gaddr = unsafe { libc::dlsym(std::ptr::null_mut(), name.as_ptr()) } as usize;
    if gaddr == 0 {
        panic!(
            "Unable to find global address: {}",
            name.clone().into_string().unwrap()
        )
    }
    let mut fsaddr: usize;
    unsafe {
        std::arch::asm!(
            "mov {fsaddr}, fs:0",
            fsaddr = out(reg) fsaddr,
        );
    }
    let off = (fsaddr - gaddr).try_into().unwrap();
    // The segment register pointer should always be bigger than any of the thread local pointers.
    // For simplicity and since that what dynasmrt expects later we return an `i32` here.
    assert!(off > 0);
    off
}

/// A function that we can put a debugger breakpoint on.
/// FIXME: gross hack.
#[cfg(debug_assertions)]
#[unsafe(no_mangle)]
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
                let bitw = m.inst(iidx).def_bitw(m);
                assert!(bitw <= 32);
                VarLocation::ConstInt {
                    bits: bitw,
                    v: u64::from(*v),
                }
            }
            yksmp::Location::LargeConstant(v) => {
                let bitw = m.inst(iidx).def_bitw(m);
                assert!(bitw <= 64);
                VarLocation::ConstInt { bits: bitw, v: *v }
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
                yksmp::Location::Register(dwarf, 8, SmallVec::new())
            }
            VarLocation::ConstInt { bits, v } => {
                if *bits <= 32 {
                    yksmp::Location::Constant(u32::try_from(*v).unwrap())
                } else if *bits <= 64 {
                    yksmp::Location::LargeConstant(*v)
                } else {
                    todo!(">32 bit constant")
                }
            }
            VarLocation::ConstPtr(v) => yksmp::Location::LargeConstant(u64::try_from(*v).unwrap()),
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
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        Assemble::new(&m)?.codegen(mt, hl)
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
    guards: Vec<CompilingGuard>,
    /// Maps assembly offsets to comments.
    ///
    /// Comments used by the trace printer for debugging and testing only.
    ///
    /// Each assembly offset can have zero or more comment lines.
    comments: Cell<IndexMap<usize, Vec<String>>>,
    /// Stack pointer offset from the base pointer of the interpreter frame:
    ///   * For a root trace, this will be the size of the interpreter frame.
    ///   * For side traces, it will be the parent frame's stack offset.
    sp_offset: usize,
    /// The offset after the trace's prologue. This is the re-entry point when returning from
    /// side-traces.
    prologue_offset: AssemblyOffset,
}

impl<'a> Assemble<'a> {
    fn new(m: &'a jit_ir::Module) -> Result<Box<Assemble<'a>>, CompilationError> {
        #[cfg(debug_assertions)]
        m.assert_well_formed();

        let asm = dynasmrt::x64::Assembler::new()
            .map_err(|e| CompilationError::ResourceExhausted(Box::new(e)))?;
        // Since we are executing the trace in the main interpreter frame we need this to
        // initialise the trace's register allocator in order to access local variables.
        let sp_offset = match m.tracekind() {
            TraceKind::HeaderOnly
            | TraceKind::HeaderAndBody
            | TraceKind::Connector(_)
            | TraceKind::DifferentFrames => {
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
            }
            TraceKind::Sidetrace(sti) => {
                // This is a side-trace. Use the passed in stack size to initialise the register
                // allocator.
                sti.sp_offset
            }
        };

        Ok(Box::new(Self {
            m,
            ra: LSRegAlloc::new(m, sp_offset),
            asm,
            header_start_locs: Vec::new(),
            body_start_locs: Vec::new(),
            guards: Vec::new(),
            comments: Cell::new(IndexMap::new()),
            sp_offset,
            prologue_offset: AssemblyOffset(0),
        }))
    }

    fn codegen(
        mut self: Box<Self>,
        mt: Arc<MT>,
        hl: Arc<Mutex<HotLocation>>,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let alloc_off = self.emit_prologue();
        self.cg_insts()?;
        let body_stack_size = self.ra.stack_size();
        let (compiled_guards, patch_deopts, max_guard_body_stack_size) =
            self.codegen_guard_bodies()?;
        let max_stack_size = std::cmp::max(max_guard_body_stack_size, body_stack_size)
            .next_multiple_of(SYSV_CALL_STACK_ALIGN);

        // Now we know the size of the stack frame (i.e. self.asp), patch the allocation with the
        // correct amount.
        self.patch_frame_allocation(alloc_off, max_stack_size);

        // If an error happens here, we've made a mistake in the assembly we generate.
        self.asm
            .commit()
            .map_err(|e| CompilationError::InternalError(format!("When committing: {e}")))?;
        // This unwrap cannot fail if `commit` (above) succeeded.
        let buf = self.asm.finalize().unwrap();

        // Patch deopt addresses into the mov instruction.
        patch_addresses(
            patch_deopts
                .into_iter()
                .map(|(mov, deopt)| (buf.ptr(AssemblyOffset(mov.0 + 2)), buf.ptr(deopt) as u64))
                .collect::<Vec<_>>()
                .as_slice(),
        );

        #[cfg(any(debug_assertions, test))]
        let gdb_ctx = gdb::register_jitted_code(
            self.m.ctrid(),
            buf.ptr(AssemblyOffset(0)),
            buf.size(),
            self.comments.get_mut(),
        )?;

        Ok(Arc::new(X64CompiledTrace {
            ctrid: self.m.ctrid(),
            safepoint: self.m.safepoint.as_ref().cloned(),
            mt,
            buf,
            compiled_guards,
            sp_offset: max_stack_size,
            prologue_offset: self.prologue_offset.0,
            entry_vars: self.header_start_locs.clone(),
            hl: Arc::downgrade(&hl),
            comments: self.comments.take(),
            #[cfg(any(debug_assertions, test))]
            gdb_ctx,
        }))
    }

    /// Push `n` bytes of NOP-equivalent instructions. This may or may not be literal `NOP`s:
    /// higher values will lead to different sequences. In all cases, the generated code will have
    /// no runtime effect.
    fn push_nops(&mut self, mut n: usize) {
        // From https://en.wikipedia.org/wiki/NOP_(code)
        while n > 0 {
            match n {
                1 => self.asm.push(0x90),
                2 => self.asm.extend([0x66, 0x90]),
                3 => self.asm.extend([0x0F, 0x1F, 0x00]),
                4 => self.asm.extend([0x0F, 0x1F, 0x40, 0x00]),
                5 => self.asm.extend([0x0F, 0x1F, 0x44, 0x00, 0x00]),
                6 => self.asm.extend([0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]),
                7 => self.asm.extend([0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00]),
                8 => self
                    .asm
                    .extend([0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]),
                _ => {
                    self.asm
                        .extend([0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
                    n -= 9;
                    continue;
                }
            }
            break;
        }
    }

    /// Codegen an instruction.
    fn cg_insts(&mut self) -> Result<(), CompilationError> {
        let mut iter = self.m.iter_skipping_insts().peekable();
        let mut next = iter.next();
        let mut in_header = true;
        while let Some((iidx, inst)) = next {
            if self.ra.rev_an.is_inst_tombstone(iidx) {
                next = iter.next();
                continue;
            }
            if !inst.is_internal_inst()
                && !inst.has_load_effect(self.m)
                && !inst.has_store_effect(self.m)
                && !inst.is_guard()
                && self.ra.rev_an.used_only_by_guards(iidx)
            {
                next = iter.next();
                continue;
            }
            self.comment_inst(iidx, inst);
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
                jit_ir::Inst::Load(i) => {
                    next = iter.next();
                    // We can in some situations generate better code for:
                    //   %1 = load ...
                    //   %2 = add %1, <constant>
                    //   *%1 = %2
                    if let Some((b_iidx, Inst::BinOp(b_inst))) = next
                        && b_inst.binop() == BinOp::Add
                        && self.op_to_sign_ext_i32(&b_inst.rhs(self.m)).is_some()
                        && let Some((s_iidx, Inst::Store(s_inst))) = iter.peek()
                    {
                        // Now we have to check that -- taking into account ptr offsets -- the two
                        // instructions really are using the same pointer.
                        let (l_ptr_op, l_off) = match self.ra.ptradd(iidx) {
                            Some(x) => (x.ptr(self.m), x.off()),
                            None => (i.ptr(self.m), 0),
                        };
                        let (s_ptr_op, s_off) = match self.ra.ptradd(*s_iidx) {
                            Some(x) => (x.ptr(self.m), x.off()),
                            None => (s_inst.ptr(self.m), 0),
                        };
                        if l_ptr_op == s_ptr_op && l_off == s_off {
                            // We now need to check that no-one else needs the result of the load
                            // or the result of the binop.
                            if !self.ra.rev_an.is_inst_var_still_used_after(b_iidx, iidx)
                                && !self.ra.rev_an.is_inst_var_still_used_after(*s_iidx, b_iidx)
                            {
                                self.cg_load_bin_store(iidx, *i, b_iidx, b_inst, *s_iidx, *s_inst);
                                let _ = iter.next();
                                next = iter.next();
                                continue;
                            }
                        }
                    }
                    self.cg_load(iidx, i);
                    continue;
                }
                jit_ir::Inst::PtrAdd(pa_inst) => self.cg_ptradd(iidx, pa_inst),
                jit_ir::Inst::DynPtrAdd(i) => self.cg_dynptradd(iidx, i),
                jit_ir::Inst::Store(i) => self.cg_store(iidx, i),
                jit_ir::Inst::LookupGlobal(i) => self.cg_lookupglobal(iidx, i),
                jit_ir::Inst::Call(inst) => {
                    let func_decl_idx = inst.target();
                    match self.m.func_decl(func_decl_idx).name() {
                        // This is a workaround for dealing with struct return values, which our
                        // register allocator cannot handle. When we see this particular call we
                        // look forwards for the matching `extractvalue` instructions and then
                        // codegen all three instructions in one go.
                        x if x.starts_with("llvm.umul.with.overflow") => {
                            let (mut overflow, overflow_off) = if let Some((
                                iidx,
                                jit_ir::Inst::ExtractValue(einst),
                            )) = iter.next()
                            {
                                (iidx, einst.index())
                            } else {
                                panic!("expected extractvalue instruction");
                            };
                            let (mut result, result_off) = if let Some((
                                iidx,
                                jit_ir::Inst::ExtractValue(einst),
                            )) = iter.next()
                            {
                                (iidx, einst.index())
                            } else {
                                panic!("expected extractvalue instruction");
                            };
                            if overflow_off != 1 {
                                // The extractvalue instructions aren't in the expected order, so
                                // swap them.
                                assert_eq!(result_off, 1);
                                (overflow, result) = (result, overflow);
                            }
                            let args = (0..(inst.num_args()))
                                .map(|i| inst.operand(self.m, i))
                                .collect::<Vec<_>>();
                            let [op_a, op_b] = args.try_into().unwrap();
                            self.cg_umul_overflow(iidx, overflow, result, op_a, op_b);
                        }
                        _ => self.cg_call(iidx, inst)?,
                    }
                }
                jit_ir::Inst::IndirectCall(i) => self.cg_indirectcall(iidx, i)?,
                jit_ir::Inst::ICmp(ic_inst) => {
                    next = iter.next();
                    // We have a special optimisation for `ICmp`s iff they're immediately followed
                    // by a `Guard`.
                    if let Some((next_iidx, Inst::Guard(g_inst))) = next
                        && let Operand::Var(cond_idx) = g_inst.cond(self.m)
                    {
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
                    self.cg_icmp(iidx, ic_inst);
                    continue;
                }
                jit_ir::Inst::Guard(i) => self.cg_guard(iidx, i),
                jit_ir::Inst::TraceHeaderStart => self.cg_header_start(),
                jit_ir::Inst::TraceHeaderEnd(is_connector) => {
                    self.cg_header_end(iidx, *is_connector);
                    in_header = false;
                }
                jit_ir::Inst::TraceBodyStart => self.cg_body_start(),
                jit_ir::Inst::TraceBodyEnd => self.cg_body_end(iidx),
                jit_ir::Inst::SidetraceEnd => self.cg_sidetrace_end(iidx),
                jit_ir::Inst::Deopt(gidx) => self.cg_deopt(iidx, *gidx),
                jit_ir::Inst::Return(id) => self.cg_return(iidx, *id),
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
                jit_ir::Inst::DebugStr(..) => (),
                jit_ir::Inst::PtrToInt(i) => self.cg_ptrtoint(iidx, i),
                jit_ir::Inst::IntToPtr(i) => self.cg_inttoptr(iidx, i),
                jit_ir::Inst::UIToFP(i) => self.cg_uitofp(iidx, i),
                jit_ir::Inst::ExtractValue(_) => todo!(),
            }

            next = iter.next();
        }
        Ok(())
    }

    /// Add a comment to the trace. Note: for instructions, use [Self::comment_inst] which formats
    /// things more appropriately for instructions.
    fn comment(&mut self, line: String) {
        self.comments
            .get_mut()
            .entry(self.asm.offset().0)
            .or_default()
            .push(line);
    }

    /// Add a comment to the trace for a "JIT IR" instruction. This function will format some
    /// instructions differently to the normal trace IR, because this x64 backend has some
    /// non-generic optimisations / modifications.
    fn comment_inst(&mut self, iidx: InstIdx, inst: Inst) {
        match inst {
            Inst::Guard(x) => {
                let gi = x.guard_info(self.m);
                let live_vars = gi
                    .live_vars()
                    .iter()
                    .map(|(x, y)| {
                        format!(
                            "{}:%{}_{}: {}",
                            usize::from(x.funcidx()),
                            usize::from(x.bbidx()),
                            usize::from(x.iidx()),
                            y.unpack(self.m).display(self.m),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                self.comment(format!(
                    "guard {}, {}, [{live_vars}] ; trace_gid {} safepoint_id {}",
                    if x.expect() { "true" } else { "false" },
                    x.cond(self.m).display(self.m),
                    self.guards.len(),
                    gi.safepoint_id()
                ));
                return;
            }
            Inst::Load(_) => {
                if let Some(painst) = self.ra.ptradd(iidx) {
                    self.comment(format!(
                        "%{iidx}: {} = load {} + {}",
                        self.m.type_(inst.tyidx(self.m)).display(self.m),
                        painst.ptr(self.m).display(self.m),
                        painst.off()
                    ));
                    return;
                }
            }
            Inst::Store(sinst) => {
                if let Some(painst) = self.ra.ptradd(iidx) {
                    self.comment(format!(
                        "*({} + {}) = {}",
                        painst.ptr(self.m).display(self.m),
                        painst.off(),
                        sinst.val(self.m).display(self.m)
                    ));
                    return;
                }
            }
            _ => (),
        }
        self.comment(inst.display(self.m, iidx).to_string())
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
        self.comment(format!("prologue for trace ID #{}", self.m.ctrid()));

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
            self.comment("Breakpoint hack".into());
            // Clippy points out that `__yk_depot as i64` isn't portable, but since this entire
            // module is x86 only, we don't need to worry about portability.
            #[allow(clippy::fn_to_numeric_cast)]
            {
                dynasm!(self.asm
                    ; push r11
                    ; mov r11, QWORD __yk_break as *const () as i64
                    ; call r11
                    ; pop r11
                );
            }
        }

        alloc_off
    }

    /// Generate code for all guard bodies in this trace. Note: this will set
    /// `self.compiling_guards` to the empty list.
    fn codegen_guard_bodies(
        &mut self,
    ) -> Result<
        (
            Vec<CompiledGuard>,
            Vec<(AssemblyOffset, AssemblyOffset)>,
            usize,
        ),
        CompilationError,
    > {
        let mut compiled_guards = Vec::with_capacity(self.guards.len());
        let mut patch_deopts = Vec::new();
        let mut max_stack_size = 0;

        if self.guards.is_empty() {
            return Ok((compiled_guards, patch_deopts, max_stack_size));
        }
        // We now have to construct the "full" deopt points. Inside the trace itself, are just
        // a pair of instructions: a `cmp` followed by a `jnz` to a `fail_label` that has not
        // yet been defined. We now have to construct a full call to `__yk_deopt` for each of
        // those labels. Since, in general, we'll have multiple guards, we construct a simple
        // stub which puts an ID in a register then JMPs to (shared amongst all guards) code
        // which does the full call to __yk_deopt.
        let deopt_label = self.asm.new_dynamic_label();
        let guardcheck_label = self.asm.new_dynamic_label();
        for (i, mut gd) in std::mem::take(&mut self.guards).into_iter().enumerate() {
            let off = self.asm.offset().0;
            let align = off.next_multiple_of(16);
            self.push_nops(align - off);
            let fail_label = gd.fail_label;
            self.comment(format!("Deopt ID and patch point for guard {i:?}"));
            dynasm!(self.asm;=> fail_label);

            self.ra.restore_guard_snapshot(gd.guard_snapshot);
            let ginfo = gd.ginfo.guard_info(self.m);
            let mut body_iidxs = Vec::new();
            let mut todos = ginfo
                .live_vars()
                .iter()
                .map(|(_, pop)| pop.unpack(self.m))
                .filter_map(|x| {
                    if let Operand::Var(y) = x {
                        Some(y)
                    } else {
                        None
                    }
                })
                .filter(|x| self.ra.rev_an.used_only_by_guards(*x))
                .collect::<Vec<_>>();
            while let Some(todo_iidx) = todos.pop() {
                let todo_inst = self.m.inst(todo_iidx);
                if !todo_inst.is_internal_inst()
                    && !todo_inst.has_load_effect(self.m)
                    && !todo_inst.has_store_effect(self.m)
                    && self.ra.rev_an.used_only_by_guards(todo_iidx)
                {
                    body_iidxs.push(todo_iidx);
                    self.m
                        .inst(todo_iidx)
                        .map_operand_vars(self.m, &mut |x| todos.push(x));
                }
            }
            body_iidxs.sort();
            body_iidxs.dedup();
            for body_iidx in body_iidxs {
                let inst = self.m.inst(body_iidx);
                if inst.is_internal_inst()
                    || inst.has_load_effect(self.m)
                    || inst.has_store_effect(self.m)
                {
                    continue;
                }
                self.comment_inst(body_iidx, inst);
                match inst {
                    Inst::BinOp(x) => self.cg_binop(body_iidx, &x),
                    Inst::ICmp(x) => self.cg_icmp(body_iidx, &x),
                    Inst::LookupGlobal(x) => self.cg_lookupglobal(body_iidx, &x),
                    Inst::PtrAdd(x) => self.cg_ptradd(body_iidx, &x),
                    Inst::PtrToInt(x) => self.cg_ptrtoint(body_iidx, &x),
                    Inst::Trunc(x) => self.cg_trunc(body_iidx, &x),
                    Inst::Select(x) => self.cg_select(body_iidx, &x),
                    Inst::SExt(x) => self.cg_sext(body_iidx, &x),
                    Inst::ZExt(x) => self.cg_zext(body_iidx, &x),
                    x => todo!("{x:?}"),
                }
            }

            let (jumpreg, live_vars) = self.ra.get_ready_for_deopt(&mut self.asm, gd.ginfo);
            // FIXME: Why are `deoptid`s 64 bit? We're not going to have that many guards!

            // Align this location in such a way that the operand of the below `mov`
            // instruction is aligned to 8 bytes, and the entirety of the `mov` instruction (10
            // bytes) fits into the cache-line.
            let clsize =
                cache_size::cache_line_size(1, cache_size::CacheType::Instruction).unwrap();
            let off = self.asm.offset().0;
            let mut align = (off + 2).next_multiple_of(8);
            if align.next_multiple_of(clsize) == align {
                align += 8
            }
            self.push_nops(align - off - 2);
            // Store the future patch offset for this guard.
            let mov_off = self.asm.offset();
            gd.fail_offset = mov_off;
            // Emit the guard failure code.
            let deoptid = i32::try_from(i).unwrap();
            dynasm!(self.asm
                // After compiling a side-trace for this location, we want to patch in a jump
                // to its address here later, so we can jump to side-traces directly. We use an
                // available register that the register allocator has reserved for us to store
                // the jump location. After finalizing this trace we patch the `mov`
                // instruction below with the absolute address of the deopt routine. Once a
                // side-trace is compiled, we patch the same `mov` with a side-trace address to
                // allow us to directly jump to the side-trace.
                // FIXME: If the side-trace offset to its parent-trace is < 32-bit we can emit
                // a relative jump here that doesn't require a register and is likely faster.
                // FIXME: Ideally, instead of patching this place, we could patch the guards
                // directly which gets rid of an extra jump. But since `cg_icmp_guard` makes
                // use of various types of jumps (e.g. jne, jl, etc), this is an optimisation
                // for another time.
                ; mov Rq(jumpreg.code()), QWORD 0x0
                ; jmp Rq(jumpreg.code())
            );
            let deopt_off = self.asm.offset();
            patch_deopts.push((mov_off, deopt_off));
            dynasm!(self.asm
                ; push rsi // FIXME: We push RSI now so we can fish it back out in
                           // `deopt_label`. This misaligns the stack which we align
                           // below.
                ; mov rsi, deoptid
                ; jmp => guardcheck_label
            );
            compiled_guards.push(CompiledGuard {
                bid: gd.bid,
                fail_offset: gd.fail_offset,
                live_vars,
                inlined_frames: gd.inlined_frames,
                guard: Guard::new(),
            });

            max_stack_size = max_stack_size.max(self.ra.stack_size());
        }

        self.comment("Call __yk_deopt".to_string());
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
            // Correct the alignment for the `push rsi` above so the stack is on a 16 byte
            // boundary.
            dynasm!(self.asm; sub rsp, 8);

            // Deoptimise.
            dynasm!(self.asm; => deopt_label);
            dynasm!(self.asm
                ; mov rdi, rbp
                ; mov r8, QWORD self.m.ctrid().as_u64().cast_signed()
                ; mov rax, QWORD __yk_deopt as *const () as  i64
                ; call rax
            );
        }
        Ok((compiled_guards, patch_deopts, max_stack_size))
    }

    /// Patch the frame for a stack size -- which must be aligned to `SYSV_CALL_STACK_ALIGN` by the
    /// caller of this function!
    fn patch_frame_allocation(&mut self, asm_off: AssemblyOffset, stack_size: usize) {
        // The stack should be 16-byte aligned after allocation. This ensures that calls in the
        // trace also get a 16-byte aligned stack, as per the SysV ABI.
        // Since we initialise the register allocator with interpreter frame and parent trace
        // frames, the actual size we need to substract from RSP is the difference between the
        // current stack size and the base size we inherited.
        match i32::try_from(stack_size - self.sp_offset) {
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
                let bitw = lhs.bitw(self.m);
                // We only optimise the canonicalised case.
                if let Some(v) = self.op_to_sign_ext_i32(&rhs) {
                    match bitw {
                        32 | 64 => {
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [GPConstraint::InputOutput {
                                    op: lhs,
                                    in_ext: RegExtension::Undefined,
                                    out_ext: RegExtension::ZeroExtended,
                                    force_reg: None,
                                }],
                            );
                            match bitw {
                                32 => dynasm!(self.asm; add Rd(lhs_reg.code()), v),
                                64 => dynasm!(self.asm; add Rq(lhs_reg.code()), v),
                                _ => unreachable!(),
                            }
                        }
                        8 | 16 => {
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [GPConstraint::InputOutput {
                                    op: lhs,
                                    in_ext: RegExtension::Undefined,
                                    out_ext: RegExtension::Undefined,
                                    force_reg: None,
                                }],
                            );
                            dynasm!(self.asm; add Rd(lhs_reg.code()), v)
                        }
                        x => todo!("{x}"),
                    }
                } else {
                    match bitw {
                        32 | 64 => {
                            let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [
                                    GPConstraint::InputOutput {
                                        op: lhs,
                                        in_ext: RegExtension::Undefined,
                                        out_ext: RegExtension::ZeroExtended,
                                        force_reg: None,
                                    },
                                    GPConstraint::Input {
                                        op: rhs,
                                        in_ext: RegExtension::Undefined,
                                        force_reg: None,
                                        clobber_reg: false,
                                    },
                                ],
                            );
                            match bitw {
                                32 => dynasm!(self.asm; add Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                                64 => dynasm!(self.asm; add Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                                _ => unreachable!(),
                            }
                        }
                        8 | 16 => {
                            let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [
                                    GPConstraint::InputOutput {
                                        op: lhs,
                                        in_ext: RegExtension::Undefined,
                                        out_ext: RegExtension::Undefined,
                                        force_reg: None,
                                    },
                                    GPConstraint::Input {
                                        op: rhs,
                                        in_ext: RegExtension::Undefined,
                                        force_reg: None,
                                        clobber_reg: false,
                                    },
                                ],
                            );
                            dynasm!(self.asm; add Rd(lhs_reg.code()), Rd(rhs_reg.code()))
                        }
                        x => todo!("{x}"),
                    }
                }
            }
            BinOp::And | BinOp::Or | BinOp::Xor => {
                let bitw = lhs.bitw(self.m);
                // We only optimise the canonicalised case.
                if let Some(v) = self.op_to_imm64(&rhs) {
                    let [lhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [GPConstraint::InputOutput {
                            op: lhs,
                            in_ext: RegExtension::Undefined,
                            out_ext: if inst.binop() == BinOp::And {
                                RegExtension::ZeroExtended
                            } else {
                                RegExtension::Undefined
                            },
                            force_reg: None,
                        }],
                    );
                    match inst.binop() {
                        BinOp::And => match bitw {
                            1..=32 => dynasm!(self.asm; and Rd(lhs_reg.code()), v),
                            // OPT: We could `and` with a 32 bit register.
                            64 => dynasm!(self.asm; and Rq(lhs_reg.code()), v),
                            _ => unreachable!(),
                        },
                        BinOp::Or => match bitw {
                            1..=32 => dynasm!(self.asm; or Rd(lhs_reg.code()), v),
                            64 => dynasm!(self.asm; or Rq(lhs_reg.code()), v),
                            _ => unreachable!(),
                        },
                        BinOp::Xor => match bitw {
                            1..=32 => dynasm!(self.asm; xor Rd(lhs_reg.code()), v),
                            64 => dynasm!(self.asm; xor Rq(lhs_reg.code()), v),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    }
                } else {
                    let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [
                            GPConstraint::InputOutput {
                                op: lhs,
                                in_ext: RegExtension::Undefined,
                                out_ext: RegExtension::Undefined,
                                force_reg: None,
                            },
                            GPConstraint::Input {
                                op: rhs,
                                in_ext: RegExtension::Undefined,
                                force_reg: None,
                                clobber_reg: false,
                            },
                        ],
                    );
                    match inst.binop() {
                        BinOp::And => match bitw {
                            1..=32 => dynasm!(self.asm; and Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                            64 => dynasm!(self.asm; and Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                            x => todo!("{x}"),
                        },
                        BinOp::Or => match bitw {
                            1..=32 => dynasm!(self.asm; or Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                            64 => dynasm!(self.asm; or Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                            x => todo!("{x}"),
                        },
                        BinOp::Xor => match bitw {
                            1..=32 => dynasm!(self.asm; xor Rd(lhs_reg.code()), Rd(rhs_reg.code())),
                            64 => dynasm!(self.asm; xor Rq(lhs_reg.code()), Rq(rhs_reg.code())),
                            x => todo!("{x}"),
                        },
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
                if let Some(v) = self.op_to_zero_ext_i8(&rhs) {
                    match inst.binop() {
                        BinOp::AShr => {
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [GPConstraint::InputOutput {
                                    op: lhs,
                                    in_ext: RegExtension::SignExtended,
                                    out_ext: RegExtension::SignExtended,
                                    force_reg: None,
                                }],
                            );
                            dynasm!(self.asm; sar Rq(lhs_reg.code()), v);
                        }
                        BinOp::LShr => {
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [GPConstraint::InputOutput {
                                    op: lhs,
                                    in_ext: RegExtension::ZeroExtended,
                                    out_ext: RegExtension::ZeroExtended,
                                    force_reg: None,
                                }],
                            );
                            dynasm!(self.asm; shr Rq(lhs_reg.code()), v);
                        }
                        BinOp::Shl => {
                            let [lhs_reg] = self.ra.assign_gp_regs(
                                &mut self.asm,
                                iidx,
                                [GPConstraint::InputOutput {
                                    op: lhs,
                                    in_ext: RegExtension::Undefined,
                                    out_ext: RegExtension::Undefined,
                                    force_reg: None,
                                }],
                            );
                            dynasm!(self.asm; shl Rq(lhs_reg.code()), v);
                        }
                        x => todo!("{x}"),
                    }
                } else {
                    let (in_ext, out_ext) = match inst.binop() {
                        BinOp::AShr => (RegExtension::SignExtended, RegExtension::SignExtended),
                        BinOp::LShr => (RegExtension::ZeroExtended, RegExtension::ZeroExtended),
                        BinOp::Shl => (RegExtension::Undefined, RegExtension::Undefined),
                        _ => unreachable!(),
                    };
                    let [lhs_reg, _rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [
                            GPConstraint::InputOutput {
                                op: lhs,
                                in_ext,
                                out_ext,
                                force_reg: None,
                            },
                            GPConstraint::Input {
                                op: rhs,
                                in_ext: RegExtension::ZeroExtended,
                                force_reg: Some(Rq::RCX),
                                clobber_reg: false,
                            },
                        ],
                    );
                    match inst.binop() {
                        BinOp::AShr => dynasm!(self.asm; sar Rq(lhs_reg.code()), cl),
                        BinOp::LShr => dynasm!(self.asm; shr Rq(lhs_reg.code()), cl),
                        BinOp::Shl => dynasm!(self.asm; shl Rq(lhs_reg.code()), cl),
                        _ => unreachable!(),
                    }
                }
            }
            BinOp::Mul => {
                let [_lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        GPConstraint::InputOutput {
                            op: lhs,
                            in_ext: RegExtension::ZeroExtended,
                            out_ext: RegExtension::Undefined,
                            force_reg: Some(Rq::RAX),
                        },
                        GPConstraint::Input {
                            op: rhs,
                            in_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        // Because we're dealing with unchecked multiply, the higher-order part of
                        // the result in RDX is ignored.
                        GPConstraint::Clobber { force_reg: Rq::RDX },
                    ],
                );
                assert!(rhs_reg != Rq::RAX && rhs_reg != Rq::RDX);
                dynasm!(self.asm; mul Rq(rhs_reg.code()));
            }
            BinOp::SDiv => {
                assert_eq!(lhs.bitw(self.m), rhs.bitw(self.m));
                let [_lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        // 64-bit (or 32-bit) signed division with idiv operates on RDX:RAX
                        // (EDX:EAX) and stores the quotient in RAX (EAX). We ignore the remainder
                        // stored into RDX (EDX).
                        GPConstraint::InputOutput {
                            op: lhs,
                            in_ext: RegExtension::SignExtended,
                            out_ext: RegExtension::SignExtended,
                            force_reg: Some(Rq::RAX),
                        },
                        GPConstraint::Input {
                            op: rhs,
                            in_ext: RegExtension::SignExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        GPConstraint::Clobber { force_reg: Rq::RDX },
                    ],
                );
                assert!(rhs_reg != Rq::RAX && rhs_reg != Rq::RDX);
                dynasm!(self.asm
                    ; cqo // Sign extend RAX up to RDX:RAX.
                    ; idiv Rq(rhs_reg.code())
                );
            }
            BinOp::SRem => {
                let bitw = lhs.bitw(self.m);
                let [lhs_reg, rhs_reg, _rem_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    // 64-bit (or 32-bit) signed division with idiv operates on RDX:RAX
                    // (EDX:EAX) and stores the remainder in RDX (EDX). We ignore the
                    // quotient stored into RAX (EAX).
                    [
                        GPConstraint::Input {
                            op: lhs,
                            in_ext: RegExtension::SignExtended,
                            force_reg: Some(Rq::RAX),
                            clobber_reg: true,
                        },
                        GPConstraint::Input {
                            op: rhs,
                            in_ext: RegExtension::SignExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        GPConstraint::Output {
                            out_ext: RegExtension::SignExtended,
                            force_reg: Some(Rq::RDX),
                            can_be_same_as_input: false,
                        },
                    ],
                );
                assert_eq!(lhs_reg, Rq::RAX);
                assert_eq!(_rem_reg, Rq::RDX);
                assert!(rhs_reg != Rq::RAX && rhs_reg != Rq::RDX);
                assert!(bitw > 0 && bitw <= 64);
                dynasm!(self.asm
                    ; cqo // Sign extend RAX up to RDX:RAX.
                    ; idiv Rq(rhs_reg.code())
                );
            }
            BinOp::Sub => {
                let bitw = lhs.bitw(self.m);
                if let Some(0) = self.op_to_sign_ext_i32(&lhs) {
                    assert!(bitw <= 64);
                    let [lhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [GPConstraint::InputOutput {
                            op: rhs,
                            in_ext: RegExtension::SignExtended,
                            out_ext: RegExtension::SignExtended,
                            force_reg: None,
                        }],
                    );
                    dynasm!(self.asm; neg Rq(lhs_reg.code()));
                } else {
                    let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [
                            GPConstraint::InputOutput {
                                op: lhs,
                                in_ext: RegExtension::SignExtended,
                                out_ext: RegExtension::SignExtended,
                                force_reg: None,
                            },
                            GPConstraint::Input {
                                op: rhs,
                                in_ext: RegExtension::SignExtended,
                                force_reg: None,
                                clobber_reg: false,
                            },
                        ],
                    );
                    assert!(bitw > 0 && bitw <= 64);
                    dynasm!(self.asm; sub Rq(lhs_reg.code()), Rq(rhs_reg.code()));
                }
            }
            BinOp::UDiv => {
                let bitw = lhs.bitw(self.m);
                assert_eq!(lhs.bitw(self.m), rhs.bitw(self.m));
                let [_lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        // 64-bit (or 32-bit) unsigned division with idiv operates on RDX:RAX
                        // (EDX:EAX) and stores the quotient in RAX (EAX). We ignore the remainder
                        // put into RDX (EDX).
                        GPConstraint::InputOutput {
                            op: lhs,
                            in_ext: RegExtension::ZeroExtended,
                            out_ext: RegExtension::ZeroExtended,
                            force_reg: Some(Rq::RAX),
                        },
                        GPConstraint::Input {
                            op: rhs,
                            in_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        GPConstraint::Clobber { force_reg: Rq::RDX },
                    ],
                );
                assert!(rhs_reg != Rq::RAX && rhs_reg != Rq::RDX);
                assert!(bitw > 0 && bitw <= 64);
                dynasm!(self.asm
                    ; xor rdx, rdx
                    ; div Rq(rhs_reg.code())
                );
            }
            BinOp::FDiv => {
                let bitw = lhs.bitw(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match bitw {
                    32 => dynasm!(self.asm; divss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    64 => dynasm!(self.asm; divsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FAdd => {
                let bitw = lhs.bitw(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match bitw {
                    32 => dynasm!(self.asm; addss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    64 => dynasm!(self.asm; addsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FMul => {
                let bitw = lhs.bitw(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match bitw {
                    32 => dynasm!(self.asm; mulss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    64 => dynasm!(self.asm; mulsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    _ => todo!(),
                }
            }
            BinOp::FSub => {
                let bitw = lhs.bitw(self.m);
                let [lhs_reg, rhs_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::InputOutput(lhs), RegConstraint::Input(rhs)],
                );
                match bitw {
                    32 => dynasm!(self.asm; subss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                    64 => dynasm!(self.asm; subsd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
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
        debug_assert!(self.m.inst(iidx).def_bitw(self.m) <= 64);
        match m {
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
                self.ra
                    .force_assign_inst_indirect(iidx, i32::try_from(frame_off).unwrap());
            }
            VarLocation::ConstInt { bits, v } => {
                self.ra.assign_const_int(iidx, bits, v);
            }
            e => panic!("{e:?}"),
        }
    }

    /// Generate code for a [LoadInst], loading from a `register + off`. `off` should only be
    /// non-zero if the [LoadInst] references a [PtrAddInst].
    fn cg_load(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LoadInst) {
        let (ptr_op, off) = match self.ra.ptradd(iidx) {
            Some(x) => (x.ptr(self.m), x.off()),
            None => (inst.ptr(self.m), 0),
        };

        match self.m.type_(inst.tyidx()) {
            Ty::Integer(_) | Ty::Ptr => {
                let [in_reg, out_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        GPConstraint::Input {
                            op: ptr_op,
                            in_ext: RegExtension::Undefined,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        GPConstraint::Output {
                            out_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            can_be_same_as_input: true,
                        },
                    ],
                );
                match self.m.inst(iidx).def_bitw(self.m) {
                    8 => {
                        dynasm!(self.asm ; movzx Rd(out_reg.code()), BYTE [Rq(in_reg.code()) + off])
                    }
                    16 => {
                        dynasm!(self.asm ; movzx Rd(out_reg.code()), WORD [Rq(in_reg.code()) + off])
                    }
                    32 => dynasm!(self.asm ; mov Rd(out_reg.code()), [Rq(in_reg.code()) + off]),
                    64 => dynasm!(self.asm ; mov Rq(out_reg.code()), [Rq(in_reg.code()) + off]),
                    x => todo!("{x}"),
                };
            }
            Ty::Float(fty) => {
                let ([src_reg], [tgt_reg]) = self.ra.assign_regs(
                    &mut self.asm,
                    iidx,
                    [GPConstraint::Input {
                        op: ptr_op,
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    }],
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

    fn cg_load_bin_store(
        &mut self,
        l_iidx: InstIdx,
        l_inst: jit_ir::LoadInst,
        b_iidx: InstIdx,
        b_inst: jit_ir::BinOpInst,
        s_iidx: InstIdx,
        s_inst: jit_ir::StoreInst,
    ) {
        let (ptr_op, off) = match self.ra.ptradd(l_iidx) {
            Some(x) => (x.ptr(self.m), x.off()),
            None => (l_inst.ptr(self.m), 0),
        };
        if let Some(c) = self.op_to_sign_ext_i32(&b_inst.rhs(self.m)) {
            self.comment(Inst::BinOp(b_inst).display(self.m, b_iidx).to_string());
            self.comment(Inst::Store(s_inst).display(self.m, s_iidx).to_string());
            let [reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                s_iidx,
                [GPConstraint::Input {
                    op: ptr_op,
                    in_ext: RegExtension::Undefined,
                    force_reg: None,
                    clobber_reg: false,
                }],
            );
            match b_inst.binop() {
                BinOp::Add => match Inst::Load(l_inst).def_bitw(self.m) {
                    64 => dynasm!(&mut self.asm; add QWORD [Rq(reg.code()) + off], c),
                    32 => dynasm!(&mut self.asm; add DWORD [Rq(reg.code()) + off], c),
                    16 => dynasm!(&mut self.asm; add WORD [Rq(reg.code()) + off], c as i16),
                    8 => dynasm!(&mut self.asm; add BYTE [Rq(reg.code()) + off], c as i8),
                    x => todo!("{x}"),
                },
                x => todo!("{x}"),
            }
        } else {
            todo!();
        }
    }

    fn cg_ptradd(&mut self, iidx: InstIdx, inst: &jit_ir::PtrAddInst) {
        // LLVM semantics dictate that the offset should be sign-extended/truncated up/down to the
        // size of the LLVM pointer index type. For address space zero on x86, truncation can't
        // happen, and when an immediate second operand is used for x86_64 `add`, it is implicitly
        // sign extended.
        let ptr_op = inst.ptr(self.m);
        let [in_reg, out_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::Input {
                    op: ptr_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: None,
                    clobber_reg: false,
                },
                GPConstraint::Output {
                    out_ext: RegExtension::ZeroExtended,
                    force_reg: None,
                    can_be_same_as_input: true,
                },
            ],
        );

        dynasm!(self.asm ; lea Rq(out_reg.code()), [Rq(in_reg.code()) + inst.off()]);
    }

    fn cg_dynptradd(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::DynPtrAddInst) {
        // LLVM semantics dictate that the element size and number of elements should be
        // sign-extended/truncated up/down to the size of the LLVM pointer index type. For address
        // space zero on x86_64, truncation can't happen, and when an immediate third operand is
        // used, that also isn't a worry.
        match inst.elem_size() {
            1 | 2 | 4 | 8 => {
                let [num_elems_reg, ptr_reg, out_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        GPConstraint::Input {
                            op: inst.num_elems(self.m),
                            in_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        GPConstraint::Input {
                            op: inst.ptr(self.m),
                            in_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                        GPConstraint::Output {
                            out_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            can_be_same_as_input: true,
                        },
                    ],
                );

                match inst.elem_size() {
                    1 => {
                        dynasm!(self.asm; lea Rq(out_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code())])
                    }
                    2 => {
                        dynasm!(self.asm; lea Rq(out_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code()) * 2])
                    }
                    4 => {
                        dynasm!(self.asm; lea Rq(out_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code()) * 4])
                    }
                    8 => {
                        dynasm!(self.asm; lea Rq(out_reg.code()), [Rq(ptr_reg.code()) + Rq(num_elems_reg.code()) * 8])
                    }
                    _ => unreachable!(),
                }
            }
            _ => {
                let [num_elems_reg, ptr_reg] = self.ra.assign_gp_regs(
                    &mut self.asm,
                    iidx,
                    [
                        GPConstraint::InputOutput {
                            op: inst.num_elems(self.m),
                            in_ext: RegExtension::ZeroExtended,
                            out_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                        },
                        GPConstraint::Input {
                            op: inst.ptr(self.m),
                            in_ext: RegExtension::ZeroExtended,
                            force_reg: None,
                            clobber_reg: false,
                        },
                    ],
                );
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
            None => (inst.ptr(self.m), 0),
        };

        let val = inst.val(self.m);
        match self.m.type_(val.tyidx(self.m)) {
            Ty::Integer(_) | Ty::Ptr => {
                let bitw = val.bitw(self.m);
                if bitw == 8
                    && let Some(v) = self.op_to_zero_ext_i8(&val)
                {
                    let [tgt_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [GPConstraint::Input {
                            op: tgt_op,
                            in_ext: RegExtension::Undefined,
                            force_reg: None,
                            clobber_reg: false,
                        }],
                    );
                    dynasm!(self.asm ; mov BYTE [Rq(tgt_reg.code()) + off], v);
                } else if bitw == 16
                    && let Some(v) = self.op_to_zero_ext_i16(&val)
                {
                    let [tgt_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [GPConstraint::Input {
                            op: tgt_op,
                            in_ext: RegExtension::Undefined,
                            force_reg: None,
                            clobber_reg: false,
                        }],
                    );
                    dynasm!(self.asm ; mov WORD [Rq(tgt_reg.code()) + off], v);
                } else if bitw == 32
                    && let Some(v) = self.op_to_zero_ext_i32(&val)
                {
                    let [tgt_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [GPConstraint::Input {
                            op: tgt_op,
                            in_ext: RegExtension::Undefined,
                            force_reg: None,
                            clobber_reg: false,
                        }],
                    );
                    dynasm!(self.asm ; mov DWORD [Rq(tgt_reg.code()) + off], v);
                } else if bitw == 64
                    && let Some(v) = self.op_to_imm64(&val)
                {
                    let [tgt_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [GPConstraint::Input {
                            op: tgt_op,
                            in_ext: RegExtension::Undefined,
                            force_reg: None,
                            clobber_reg: false,
                        }],
                    );
                    dynasm!(self.asm ; mov QWORD [Rq(tgt_reg.code()) + off], v);
                } else {
                    let [tgt_reg, val_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [
                            GPConstraint::Input {
                                op: tgt_op,
                                in_ext: RegExtension::Undefined,
                                force_reg: None,
                                clobber_reg: false,
                            },
                            GPConstraint::Input {
                                op: val,
                                in_ext: RegExtension::ZeroExtended,
                                force_reg: None,
                                clobber_reg: false,
                            },
                        ],
                    );
                    match bitw {
                        1 | 8 => {
                            dynasm!(self.asm ; mov BYTE [Rq(tgt_reg.code()) + off], Rb(val_reg.code()))
                        }
                        16 => {
                            dynasm!(self.asm ; mov WORD [Rq(tgt_reg.code()) + off], Rw(val_reg.code()))
                        }
                        32 => {
                            dynasm!(self.asm ; mov DWORD [Rq(tgt_reg.code()) + off], Rd(val_reg.code()))
                        }
                        64 => {
                            dynasm!(self.asm ; mov QWORD [Rq(tgt_reg.code()) + off], Rq(val_reg.code()))
                        }
                        _ => todo!(),
                    }
                }
            }
            Ty::Float(fty) => {
                let ([tgt_reg], [val_reg]) = self.ra.assign_regs(
                    &mut self.asm,
                    iidx,
                    [GPConstraint::Input {
                        op: tgt_op,
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    }],
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
            Ty::Struct(_) => todo!(),
            Ty::Void | Ty::Func(_) => todo!(),
            Ty::Unimplemented(_) => todo!(),
        }
    }

    #[cfg(not(test))]
    fn cg_lookupglobal(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::LookupGlobalInst) {
        let decl = inst.decl(self.m);
        let [tgt_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::Output {
                out_ext: RegExtension::Undefined,
                force_reg: None,
                can_be_same_as_input: false,
            }],
        );
        if decl.is_threadlocal() {
            let off = get_tls_offset(decl.name());
            dynasm!(self.asm
                ; fs mov Rq(tgt_reg.code()), [0]
                ; sub Rq(tgt_reg.code()), off
            );
            return;
        }
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

        match self.m.func_decl(func_decl_idx).name() {
            "llvm.assume" => Ok(()),
            "llvm.lifetime.start.p0" => Ok(()),
            "llvm.lifetime.end.p0" => Ok(()),
            x if x.starts_with("llvm.abs.") => {
                let [op, is_int_min] = args.try_into().unwrap();
                self.cg_abs(iidx, op, is_int_min);
                Ok(())
            }
            x if x.starts_with("llvm.ctpop.") => {
                let [op] = args.try_into().unwrap();
                self.cg_ctpop(iidx, op);
                Ok(())
            }
            x if x.starts_with("llvm.floor.") => {
                let [op] = args.try_into().unwrap();
                self.cg_floor(iidx, op);
                Ok(())
            }
            x if x.starts_with("llvm.fshl.i") => {
                let [op_a, op_b, op_c] = args.try_into().unwrap();
                self.cg_fshl(iidx, op_a, op_b, op_c);
                Ok(())
            }
            x if x.starts_with("llvm.memcpy.") => {
                let [dst, src, len, is_volatile] = args.try_into().unwrap();
                self.cg_memcpy(iidx, dst, src, len, is_volatile);
                Ok(())
            }
            "llvm.memset.p0.i64" => {
                let [dst, val, len, is_volatile] = args.try_into().unwrap();
                self.cg_memset(iidx, dst, val, len, is_volatile);
                Ok(())
            }
            x if x.starts_with("llvm.smax.") => {
                let [lhs_op, rhs_op] = args.try_into().unwrap();
                self.cg_smax(iidx, lhs_op, rhs_op);
                Ok(())
            }
            x if x.starts_with("llvm.smin.") => {
                let [lhs_op, rhs_op] = args.try_into().unwrap();
                self.cg_smin(iidx, lhs_op, rhs_op);
                Ok(())
            }
            x => {
                // If we have an optimised clone of the function, call it.
                let va = symbol_to_ptr(&format!("__yk_opt_{x}"))
                    .or_else(|_| symbol_to_ptr(x))
                    .map_err(|e| CompilationError::General(e.to_string()))?;
                self.emit_call(iidx, fty, Some(va), None, &args)
            }
        }
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
            .map(|reg| GPConstraint::Clobber { force_reg: *reg })
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
                    gp_cnstrs[gp_i] = GPConstraint::Input {
                        op: arg.clone(),
                        in_ext: RegExtension::ZeroExtended,
                        force_reg: Some(*reg),
                        clobber_reg: true,
                    };
                }
                Ty::Struct(_) => todo!(),
                Ty::Void => unreachable!(),
                Ty::Unimplemented(_) => todo!(),
            }
        }

        // Deal with outputs.
        let ret_ty = fty.ret_type(self.m);
        // FIXME: We only support up to register-sized return values at the moment.
        #[cfg(debug_assertions)]
        if !matches!(ret_ty, Ty::Void) {
            debug_assert!(ret_ty.bitw().unwrap() <= 64);
        }
        let rax_i = CALLER_CLOBBER_REGS
            .iter()
            .position(|x| *x == Rq::RAX)
            .unwrap();
        match ret_ty {
            Ty::Void => {
                if let Some(op) = callee_op.clone() {
                    // Indirect call
                    if !fty.is_vararg() {
                        gp_cnstrs[rax_i] = GPConstraint::Input {
                            op,
                            in_ext: RegExtension::ZeroExtended,
                            force_reg: Some(Rq::RAX),
                            clobber_reg: true,
                        };
                    } else {
                        // We won't be able to use rax in this case!
                        todo!();
                    }
                }
            }
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
                if callee.is_some() {
                    // Direct call
                    gp_cnstrs[rax_i] = GPConstraint::Output {
                        out_ext: RegExtension::ZeroExtended,
                        force_reg: Some(Rq::RAX),
                        can_be_same_as_input: false,
                    };
                } else if let Some(op) = callee_op.clone() {
                    // Indirect call
                    if !fty.is_vararg() {
                        gp_cnstrs[rax_i] = GPConstraint::InputOutput {
                            op,
                            in_ext: RegExtension::ZeroExtended,
                            out_ext: RegExtension::ZeroExtended,
                            force_reg: Some(Rq::RAX),
                        };
                    }
                }
            }
            Ty::Struct(_) => todo!(),
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
                        gp_cnstrs.clone().try_into().unwrap(),
                        fp_cnstrs,
                    );
                    // rax is considered clobbered, but isn't used to pass an argument, so we can
                    // safely use it for the function pointer.
                    dynasm!(self.asm
                        ; mov rax, QWORD p as i64
                        ; call rax
                    );
                } else {
                    gp_cnstrs.push(GPConstraint::Temporary);
                    let ([.., tmp_reg], _): ([Rq; CALLER_CLOBBER_REGS.len() + 1], [Rx; 16]) =
                        self.ra.assign_regs(
                            &mut self.asm,
                            iidx,
                            gp_cnstrs.clone().try_into().unwrap(),
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
                if !fty.is_vararg() {
                    let ([..], _): ([Rq; CALLER_CLOBBER_REGS.len()], [Rx; 16]) =
                        self.ra.assign_regs(
                            &mut self.asm,
                            iidx,
                            gp_cnstrs.clone().try_into().unwrap(),
                            fp_cnstrs,
                        );
                    dynasm!(self.asm; call rax);
                } else {
                    gp_cnstrs.push(GPConstraint::Input {
                        op,
                        in_ext: RegExtension::ZeroExtended,
                        force_reg: None,
                        clobber_reg: false,
                    });
                    let ([.., op_reg], _): ([Rq; CALLER_CLOBBER_REGS.len() + 1], [Rx; 16]) =
                        self.ra.assign_regs(
                            &mut self.asm,
                            iidx,
                            gp_cnstrs.clone().try_into().unwrap(),
                            fp_cnstrs,
                        );
                    dynasm!(self.asm
                        ; mov rax, num_float_args
                        ; call Rq(op_reg.code()));
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn cg_abs(&mut self, iidx: InstIdx, op: Operand, _is_int_min: Operand) {
        let bitw = op.bitw(self.m);
        let [io_reg, tmp_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::InputOutput {
                    op: op.clone(),
                    in_ext: RegExtension::SignExtended,
                    out_ext: RegExtension::SignExtended,
                    force_reg: None,
                },
                GPConstraint::Temporary,
            ],
        );
        match bitw {
            64 => {
                // This returns INT_MIN for INT_MIN, which is correct whether or not `is_int_min`
                // is 0 or 1.
                dynasm!(self.asm
                    ; mov Rq(tmp_reg.code()), Rq(io_reg.code())
                    ; neg Rq(io_reg.code())
                    ; cmovl Rq(io_reg.code()), Rq(tmp_reg.code())
                );
            }
            x => todo!("{x}"),
        }
    }

    fn cg_ctpop(&mut self, iidx: InstIdx, op: Operand) {
        let bitw = op.bitw(self.m);
        let [in_reg, out_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::Input {
                    op: op.clone(),
                    in_ext: if bitw == 32 || bitw == 64 {
                        RegExtension::Undefined
                    } else {
                        RegExtension::ZeroExtended
                    },
                    force_reg: None,
                    clobber_reg: false,
                },
                GPConstraint::Output {
                    out_ext: RegExtension::ZeroExtended,
                    force_reg: None,
                    can_be_same_as_input: true,
                },
            ],
        );
        assert!(bitw > 1 && bitw <= 64);
        if bitw <= 32 {
            dynasm!(self.asm; popcnt Rd(out_reg.code()), Rd(in_reg.code()));
        } else if bitw == 64 {
            dynasm!(self.asm; popcnt Rq(out_reg.code()), Rq(in_reg.code()));
        } else {
            todo!("{bitw}");
        }
    }

    fn cg_floor(&mut self, iidx: InstIdx, op: Operand) {
        match self.m.type_(op.tyidx(self.m)) {
            Ty::Void => todo!(),
            Ty::Integer(_) => todo!(),
            Ty::Ptr => todo!(),
            Ty::Func(_) => todo!(),
            Ty::Struct(_) => todo!(),
            Ty::Float(fty) => {
                let [in_reg, out_reg] = self.ra.assign_fp_regs(
                    &mut self.asm,
                    iidx,
                    [RegConstraint::Input(op.clone()), RegConstraint::Output],
                );
                match fty {
                    FloatTy::Float => todo!(),
                    FloatTy::Double => {
                        dynasm!(self.asm; roundsd Rx(out_reg.code()), Rx(in_reg.code()), 1)
                    }
                }
            }
            Ty::Unimplemented(_) => todo!(),
        }
    }

    fn cg_umul_overflow(
        &mut self,
        iidx: InstIdx,
        overflow: InstIdx,
        result: InstIdx,
        op_a: Operand,
        op_b: Operand,
    ) {
        let bitw = op_a.bitw(self.m);
        let [_lhs_reg, rhs_reg, _] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::Input {
                    op: op_a,
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RAX),
                    clobber_reg: true,
                },
                GPConstraint::Input {
                    op: op_b,
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: None,
                    clobber_reg: false,
                },
                // Because we're dealing with unchecked multiply, the higher-order part of
                // the result in RDX is ignored.
                GPConstraint::Clobber { force_reg: Rq::RDX },
            ],
        );

        // Find a register to store the overflow flag, which is stored in the first extractvalue
        // instruction.
        let [overflow_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            overflow,
            [GPConstraint::Output {
                out_ext: RegExtension::Undefined,
                force_reg: None,
                can_be_same_as_input: false,
            }],
        );
        // The 2nd extractvalue instruction stores the multiplication result which is in RAX.
        let [_] = self.ra.assign_gp_regs(
            &mut self.asm,
            result,
            [GPConstraint::Output {
                out_ext: RegExtension::Undefined,
                force_reg: Some(Rq::RAX),
                can_be_same_as_input: false,
            }],
        );
        assert!(rhs_reg != Rq::RAX && rhs_reg != Rq::RDX);

        match bitw {
            32 => dynasm!(self.asm; mul Rd(rhs_reg.code())),
            64 => dynasm!(self.asm; mul Rq(rhs_reg.code())),
            _ => todo!(),
        }
        dynasm!(self.asm
            ; seto Rb(overflow_reg.code())
            ; and Rb(overflow_reg.code()), 1
        );
    }

    fn cg_fshl(&mut self, iidx: InstIdx, op_a: Operand, op_b: Operand, op_c: Operand) {
        let bitw = op_a.bitw(self.m);
        match bitw {
            64 => {
                if let Some(c) = self.op_to_zero_ext_i8(&op_c) {
                    let [a_reg, b_reg] = self.ra.assign_gp_regs(
                        &mut self.asm,
                        iidx,
                        [
                            GPConstraint::InputOutput {
                                op: op_a,
                                in_ext: RegExtension::SignExtended,
                                out_ext: RegExtension::SignExtended,
                                force_reg: None,
                            },
                            GPConstraint::Input {
                                op: op_b,
                                in_ext: RegExtension::SignExtended,
                                force_reg: None,
                                clobber_reg: false,
                            },
                        ],
                    );
                    dynasm!(self.asm; shld Rq(a_reg.code()), Rq(b_reg.code()), c);
                } else {
                    todo!();
                }
            }
            x => todo!("{x}"),
        }
    }

    fn cg_memcpy(
        &mut self,
        iidx: InstIdx,
        dst_op: Operand,
        src_op: Operand,
        len_op: Operand,
        _is_volatile_op: Operand,
    ) {
        let [_, _, _] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::Input {
                    op: dst_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RDI),
                    clobber_reg: true,
                },
                GPConstraint::Input {
                    op: src_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RSI),
                    clobber_reg: true,
                },
                GPConstraint::Input {
                    op: len_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RCX),
                    clobber_reg: true,
                },
            ],
        );
        dynasm!(self.asm; rep movsb);
    }

    fn cg_memset(
        &mut self,
        iidx: InstIdx,
        dst_op: Operand,
        val_op: Operand,
        len_op: Operand,
        _is_volatile_op: Operand,
    ) {
        let [_, _, _] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::Input {
                    op: dst_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RDI),
                    clobber_reg: true,
                },
                GPConstraint::Input {
                    op: val_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RAX),
                    clobber_reg: true,
                },
                GPConstraint::Input {
                    op: len_op.clone(),
                    in_ext: RegExtension::ZeroExtended,
                    force_reg: Some(Rq::RCX),
                    clobber_reg: true,
                },
            ],
        );
        dynasm!(self.asm; rep stosb);
    }

    fn cg_smax(&mut self, iidx: InstIdx, lhs: Operand, rhs: Operand) {
        assert_eq!(lhs.bitw(self.m), rhs.bitw(self.m));
        let bitw = lhs.bitw(self.m);
        let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::InputOutput {
                    op: lhs,
                    in_ext: RegExtension::SignExtended,
                    out_ext: RegExtension::SignExtended,
                    force_reg: None,
                },
                GPConstraint::Input {
                    op: rhs,
                    in_ext: RegExtension::SignExtended,
                    force_reg: None,
                    clobber_reg: false,
                },
            ],
        );
        match bitw {
            64 => {
                dynasm!(self.asm
                    ; cmp Rq(lhs_reg.code()), Rq(rhs_reg.code())
                    ; cmovl Rq(lhs_reg.code()), Rq(rhs_reg.code())
                );
            }
            32 => {
                dynasm!(self.asm
                    ; cmp Rd(lhs_reg.code()), Rd(rhs_reg.code())
                    ; cmovl Rd(lhs_reg.code()), Rd(rhs_reg.code())
                );
            }
            x => todo!("{x}"),
        }
    }

    fn cg_smin(&mut self, iidx: InstIdx, lhs: Operand, rhs: Operand) {
        assert_eq!(lhs.bitw(self.m), rhs.bitw(self.m));
        let bitw = lhs.bitw(self.m);
        let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [
                GPConstraint::InputOutput {
                    op: lhs,
                    in_ext: RegExtension::SignExtended,
                    out_ext: RegExtension::SignExtended,
                    force_reg: None,
                },
                GPConstraint::Input {
                    op: rhs,
                    in_ext: RegExtension::SignExtended,
                    force_reg: None,
                    clobber_reg: false,
                },
            ],
        );
        match bitw {
            64 | 32 => {
                dynasm!(self.asm
                    ; cmp Rq(lhs_reg.code()), Rq(rhs_reg.code())
                    ; cmovg Rq(lhs_reg.code()), Rq(rhs_reg.code())
                );
            }
            x => todo!("{x}"),
        }
    }

    /// Return the [VarLocation] an [Operand] relates to.
    fn op_to_var_location(&self, op: Operand) -> VarLocation {
        match op {
            Operand::Var(iidx) => self.ra.var_location(iidx),
            Operand::Const(cidx) => match self.m.const_(cidx) {
                Const::Float(_, v) => VarLocation::ConstFloat(*v),
                Const::Int(_, x) => VarLocation::ConstInt {
                    bits: x.bitw(),
                    v: x.to_zero_ext_u64().unwrap(),
                },
                Const::Ptr(v) => VarLocation::ConstPtr(*v),
            },
        }
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i32`, return it
    /// sign-extended to 32 bits, otherwise return `None`.
    fn op_to_sign_ext_i32(&self, op: &Operand) -> Option<i32> {
        if let Operand::Const(cidx) = op
            && let Const::Int(_, x) = self.m.const_(*cidx)
        {
            return x.to_sign_ext_i32();
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as a sign-extended
    /// imm64, return an `i32`, otherwise return `None`.
    fn op_to_imm64(&self, op: &Operand) -> Option<i32> {
        if let Operand::Const(cidx) = op {
            match self.m.const_(*cidx) {
                Const::Float(_, _) => todo!(),
                Const::Int(_, v) => v
                    .to_zero_ext_u32()
                    .filter(|x| *x <= i32::MAX.cast_unsigned())
                    .map(|x| x.cast_signed()),
                Const::Ptr(v) => ArbBitInt::from_u64(64, u64::try_from(*v).unwrap())
                    .to_zero_ext_u32()
                    .filter(|x| *x <= i32::MAX.cast_unsigned())
                    .map(|x| x.cast_signed()),
            }
        } else {
            None
        }
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i8`, return
    /// it zero-extended to 8 bits, otherwise return `None`.
    fn op_to_zero_ext_i8(&self, op: &Operand) -> Option<i8> {
        if let Operand::Const(cidx) = op
            && let Const::Int(_, x) = self.m.const_(*cidx)
        {
            return x.to_zero_ext_u8().map(|x| x as i8);
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i8`, return
    /// it zero-extended to 8 bits, otherwise return `None`.
    fn op_to_zero_ext_i16(&self, op: &Operand) -> Option<i16> {
        if let Operand::Const(cidx) = op
            && let Const::Int(_, x) = self.m.const_(*cidx)
        {
            return x.to_zero_ext_u16().map(|x| x as i16);
        }
        None
    }

    /// If an `Operand` refers to a constant integer that can be represented as an `i32`, return it
    /// zero-extended to 32 bits, otherwise return `None`.
    fn op_to_zero_ext_i32(&self, op: &Operand) -> Option<i32> {
        if let Operand::Const(cidx) = op {
            match self.m.const_(*cidx) {
                Const::Float(_, _) => todo!(),
                Const::Int(_, v) => v.to_zero_ext_u32().map(|x| x.cast_signed()),
                Const::Ptr(v) => ArbBitInt::from_u64(64, u64::try_from(*v).unwrap())
                    .to_zero_ext_u32()
                    .map(|x| x.cast_signed()),
            }
        } else {
            None
        }
    }

    fn cg_cmp_const(&mut self, bitw: u32, lhs_reg: Rq, rhs: i32) {
        if rhs == 0 {
            match bitw {
                8 | 16 | 32 => dynasm!(self.asm; test Rd(lhs_reg.code()), Rd(lhs_reg.code())),
                64 => dynasm!(self.asm; test Rq(lhs_reg.code()), Rq(lhs_reg.code())),
                _ => todo!("{bitw}"),
            }
        } else {
            match bitw {
                8 | 16 | 32 => dynasm!(self.asm; cmp Rd(lhs_reg.code()), rhs),
                64 => dynasm!(self.asm; cmp Rq(lhs_reg.code()), rhs),
                _ => todo!("{bitw}"),
            }
        }
    }

    fn cg_cmp_regs(&mut self, bitw: u32, lhs_reg: Rq, rhs_reg: Rq) {
        match bitw {
            8 | 16 | 32 => dynasm!(self.asm; cmp Rd(lhs_reg.code()), Rd(rhs_reg.code())),
            64 => dynasm!(self.asm; cmp Rq(lhs_reg.code()), Rq(rhs_reg.code())),
            _ => todo!("{bitw}"),
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
        assert!(!self.ra.rev_an.is_inst_var_still_used_after(g_iidx, ic_iidx));

        // Codegen ICmp
        let (lhs, pred, rhs) = (
            ic_inst.lhs(self.m),
            ic_inst.predicate(),
            ic_inst.rhs(self.m),
        );
        let bitw = self.m.type_(lhs.tyidx(self.m)).bitw().unwrap();
        let (imm, mut in_ext) = if pred.signed() {
            (self.op_to_sign_ext_i32(&rhs), RegExtension::SignExtended)
        } else if bitw == 64 {
            (self.op_to_imm64(&rhs), RegExtension::ZeroExtended)
        } else {
            (self.op_to_zero_ext_i32(&rhs), RegExtension::ZeroExtended)
        };
        if bitw == 32 || bitw == 64 {
            in_ext = RegExtension::Undefined;
        }
        if let Some(v) = imm {
            let [lhs_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                ic_iidx,
                [GPConstraint::Input {
                    op: lhs,
                    in_ext,
                    force_reg: None,
                    clobber_reg: false,
                }],
            );
            self.cg_cmp_const(bitw, lhs_reg, v);
        } else {
            let [lhs_reg, rhs_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                ic_iidx,
                [
                    GPConstraint::Input {
                        op: lhs,
                        in_ext,
                        force_reg: None,
                        clobber_reg: false,
                    },
                    GPConstraint::Input {
                        op: rhs,
                        in_ext,
                        force_reg: None,
                        clobber_reg: false,
                    },
                ],
            );
            self.cg_cmp_regs(bitw, lhs_reg, rhs_reg);
        };

        // Codegen guard
        self.ra.expire_regs(g_iidx);
        self.comment_inst(g_iidx, g_inst.into());
        let fail_label = self.guard_to_deopt(HasGuardInfo::Guard(g_inst));

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
        let bitw = self.m.type_(lhs.tyidx(self.m)).bitw().unwrap();
        let (imm, mut in_ext) = if pred.signed() {
            (self.op_to_sign_ext_i32(&rhs), RegExtension::SignExtended)
        } else if bitw == 64 {
            (self.op_to_imm64(&rhs), RegExtension::ZeroExtended)
        } else {
            (self.op_to_zero_ext_i32(&rhs), RegExtension::ZeroExtended)
        };
        if bitw == 32 || bitw == 64 {
            in_ext = RegExtension::Undefined;
        }
        let out_reg = if let Some(v) = imm {
            let [lhs_reg, out_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [
                    GPConstraint::Input {
                        op: lhs,
                        in_ext,
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
            self.cg_cmp_const(bitw, lhs_reg, v);
            out_reg
        } else {
            let [lhs_reg, rhs_reg, out_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [
                    GPConstraint::Input {
                        op: lhs,
                        in_ext,
                        force_reg: None,
                        clobber_reg: false,
                    },
                    GPConstraint::Input {
                        op: rhs,
                        in_ext,
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
            self.cg_cmp_regs(bitw, lhs_reg, rhs_reg);
            out_reg
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
            jit_ir::Predicate::Equal => dynasm!(self.asm; sete Rb(out_reg.code())),
            jit_ir::Predicate::NotEqual => dynasm!(self.asm; setne Rb(out_reg.code())),
            jit_ir::Predicate::UnsignedGreater => dynasm!(self.asm; seta Rb(out_reg.code())),
            jit_ir::Predicate::UnsignedGreaterEqual => dynasm!(self.asm; setae Rb(out_reg.code())),
            jit_ir::Predicate::UnsignedLess => dynasm!(self.asm; setb Rb(out_reg.code())),
            jit_ir::Predicate::UnsignedLessEqual => dynasm!(self.asm; setbe Rb(out_reg.code())),
            jit_ir::Predicate::SignedGreater => dynasm!(self.asm; setg Rb(out_reg.code())),
            jit_ir::Predicate::SignedGreaterEqual => dynasm!(self.asm; setge Rb(out_reg.code())),
            jit_ir::Predicate::SignedLess => dynasm!(self.asm; setl Rb(out_reg.code())),
            jit_ir::Predicate::SignedLessEqual => dynasm!(self.asm; setle Rb(out_reg.code())),
        }
    }

    fn cg_fcmp(&mut self, iidx: InstIdx, inst: &jit_ir::FCmpInst) {
        // For some predicates we do as LLVM does and rewrite the operation into an equivalent one
        // that can be codegenned more efficiently.
        //
        // For example, suppose we want to codegen this 32-bit float comparison:
        //
        //   %3: i1 = f_ugt %1, %2
        //
        // This means "set %3 to 1 if %1 > %2 or if the comparison's result unordered (i.e. one or
        // both of %1 and %2 were NaN), otherwise set %3 to 0".
        //
        // Assume that %1 is in xmm1 and %2 is in xmm2. We'd use `ucomis{s,d}` (depending on if we
        // are operating on floats or doubles) to set the flags register and then interpret the
        // flags to know which relation(s) held.
        //
        // Here's the truth table:
        //
        //               | ZF | PF | CF
        //     ----------+----+----+---
        //     UNORDERED | 1  | 1  | 1
        //         >     | 0  | 0  | 0
        //         <     | 0  | 0  | 1
        //         =     | 1  | 0  | 0
        //
        // A naiave code-gen for this does `ucomiss xmm1, xmm2` to set the flags register, then
        // checks for the unordered result (PF=1) and if xmm1 > xmm2 (CF=0 and ZF=0), i.e.:
        //
        //     ucomiss xmm1, xmm2
        //     setp al                  ; unordered result?
        //     seta bl                  ; xmm1 > xmm2?
        //     or al, bl                ; either of the above true? result in al
        //
        // That's a lot of work for one comparison and what's more it requires a temporary
        // register.
        //
        // A more clever codegen converts `%3: i1 = f_ugt %1, %2` into the equivalent
        // `%3: i1 = f_ult $2, %1` by inverting the predicate and swapping the operands. By doing
        // this we can capture the correct outcome in one flag check: CF=1. This allows much more
        // efficient codegen:
        //
        //     ucomiss xmm2, xmm1       ; operands swapped!
        //     setb al                  ; predicate inverted, result in al
        //
        // For evience of this optimisation in LLVM, see:
        // https://github.com/llvm/llvm-project/blob/2b340c10a611d929fee25e6222909c8915e3d6b6/llvm/lib/Target/X86/X86InstrInfo.cpp#L3388
        let (lhs, pred, rhs) = match (inst.lhs(self.m), inst.predicate(), inst.rhs(self.m)) {
            (lhs, jit_ir::FloatPredicate::UnorderedGreater, rhs) => {
                (rhs, jit_ir::FloatPredicate::UnorderedLess, lhs)
            }
            (lhs, jit_ir::FloatPredicate::UnorderedGreaterEqual, rhs) => {
                (rhs, jit_ir::FloatPredicate::UnorderedLessEqual, lhs)
            }
            (lhs, jit_ir::FloatPredicate::OrderedLess, rhs) => {
                (rhs, jit_ir::FloatPredicate::OrderedGreater, lhs)
            }
            (lhs, jit_ir::FloatPredicate::OrderedLessEqual, rhs) => {
                (rhs, jit_ir::FloatPredicate::OrderedGreaterEqual, lhs)
            }
            (lhs, pred, rhs) => (lhs, pred, rhs),
        };
        let bitw = lhs.bitw(self.m);

        // Set the EFLAGS register with the result of a FP comparison with `ucomis{s,d}`.
        //
        // Doing so requires us to assign registers, so if interpreting the flags afterwards will
        // require a temporary register, pass `needs_tmp=true`.
        //
        // Returns a tuple containing the register assignment for the registers you'd need to
        // interpret the flags later: `(target_reg, tmp_reg)` where `tmp_reg` is `Option`.
        let set_eflags = |bitw, needs_tmp| {
            let fp_cstrs = [RegConstraint::Input(lhs), RegConstraint::Input(rhs)];
            let target_reg_cstr = GPConstraint::Output {
                out_ext: RegExtension::Undefined,
                force_reg: None,
                can_be_same_as_input: false,
            };
            let (tgt_reg, lhs_reg, rhs_reg, tmp_reg) = if needs_tmp {
                let ([tgt_reg, tmp_reg], [lhs_reg, rhs_reg]) = self.ra.assign_regs(
                    &mut self.asm,
                    iidx,
                    [target_reg_cstr, GPConstraint::Temporary], // request additional temp reg.
                    fp_cstrs,
                );
                (tgt_reg, lhs_reg, rhs_reg, Some(tmp_reg))
            } else {
                let ([tgt_reg], [lhs_reg, rhs_reg]) =
                    self.ra
                        .assign_regs(&mut self.asm, iidx, [target_reg_cstr], fp_cstrs);
                (tgt_reg, lhs_reg, rhs_reg, None)
            };

            // We use `ucomis{s,d}` instead of `comis{s,d}` because our IR semantics are such that
            // a float comparison involving a qNaN operand shouldn't cause a floating point
            // exception.
            match bitw {
                32 => dynasm!(self.asm; ucomiss Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                64 => dynasm!(self.asm; ucomisd Rx(lhs_reg.code()), Rx(rhs_reg.code())),
                _ => panic!(),
            }
            (tgt_reg, tmp_reg)
        };

        // Interpret the flags assignment WRT the predicate.
        //
        // Note that although floats are signed values, `ucomis{s,d}` sets CF (not SF and OF, as
        // you might expect). So when checking the outcome you have to use the "above" and "below"
        // variants of `setcc`, as if you were comparing unsigned integers.
        match pred {
            jit_ir::FloatPredicate::OrderedLess
            | jit_ir::FloatPredicate::OrderedLessEqual
            | jit_ir::FloatPredicate::UnorderedGreater
            | jit_ir::FloatPredicate::UnorderedGreaterEqual => {
                // All of these cases were re-written to their inverse above.
                unreachable!();
            }
            jit_ir::FloatPredicate::OrderedNotEqual => {
                let (tgt_reg, _) = set_eflags(bitw, false);
                dynasm!(self.asm; setne Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::OrderedEqual => {
                // This case requires two flag checks (and thus a temp reg).
                let (tgt_reg, tmp_reg) = set_eflags(bitw, true);
                let tmp_reg = tmp_reg.unwrap(); // cannot fail. we passed true to set_eflags().
                dynasm!(self.asm
                    ; sete Rb(tmp_reg.code())
                    ; setnp Rb(tgt_reg.code())
                    ; and Rb(tgt_reg.code()), Rb(tmp_reg.code())
                );
            }
            jit_ir::FloatPredicate::OrderedGreaterEqual => {
                let (tgt_reg, _) = set_eflags(bitw, false);
                dynasm!(self.asm; setae Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::OrderedGreater => {
                let (tgt_reg, _) = set_eflags(bitw, false);
                dynasm!(self.asm; seta Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::UnorderedEqual => {
                let (tgt_reg, _) = set_eflags(bitw, false);
                dynasm!(self.asm; sete Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::UnorderedNotEqual => {
                // This case requires two flag checks (and thus a temp reg).
                let (tgt_reg, tmp_reg) = set_eflags(bitw, true);
                let tmp_reg = tmp_reg.unwrap(); // cannot fail. we passed true to set_eflags().
                dynasm!(self.asm
                    ; setne Rb(tmp_reg.code())
                    ; setp Rb(tgt_reg.code())
                    ; or Rb(tgt_reg.code()), Rb(tmp_reg.code())
                )
            }
            jit_ir::FloatPredicate::UnorderedLess => {
                let (tgt_reg, _) = set_eflags(bitw, false);
                dynasm!(self.asm; setb Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::UnorderedLessEqual => {
                let (tgt_reg, _) = set_eflags(bitw, false);
                dynasm!(self.asm; setbe Rb(tgt_reg.code()))
            }
            jit_ir::FloatPredicate::False
            | jit_ir::FloatPredicate::Ordered
            | jit_ir::FloatPredicate::Unordered
            | jit_ir::FloatPredicate::True => todo!("{}", pred),
        }
    }

    /// Move live values from their source location into the target location when doing a jump back
    /// to the beginning of a trace (or a jump from a side-trace to the beginning of its root
    /// trace).
    fn write_jump_vars(&mut self, iidx: InstIdx) {
        let (tgt_vars, src_ops) = match self.m.tracekind() {
            TraceKind::HeaderOnly => (self.header_start_locs.clone(), self.m.trace_header_end()),
            TraceKind::HeaderAndBody => (self.body_start_locs.clone(), self.m.trace_body_end()),
            TraceKind::Connector(ctr) => {
                let ctr = Arc::clone(ctr)
                    .as_any()
                    .downcast::<X64CompiledTrace>()
                    .unwrap();
                (ctr.entry_vars().to_vec(), self.m.trace_header_end())
            }
            TraceKind::Sidetrace(sti) => {
                assert_eq!(sti.sp_offset, self.sp_offset);
                (sti.entry_vars.clone(), self.m.trace_header_end())
            }
            TraceKind::DifferentFrames => panic!(),
        };

        // First of all we work out what to do with registers.
        let mut gp_regs = lsregalloc::GP_REGS
            .iter()
            .map(|_| GPConstraint::None)
            .collect::<Vec<_>>();
        let mut fp_regs = lsregalloc::FP_REGS
            .iter()
            .map(|_| RegConstraint::None)
            .collect::<Vec<_>>();
        for (i, op) in src_ops.iter().enumerate() {
            let op = op.unpack(self.m);
            if let VarLocation::Register(reg) = tgt_vars[i] {
                match reg {
                    Register::GP(r) => {
                        if let GPConstraint::Input { op: cur_op, .. } =
                            &gp_regs[usize::from(r.code())]
                        {
                            // Two operands both have to end up in the same register: if our
                            // existing candidate is already in a register (i.e. is cheaper than
                            // unspilling), we prefer that, otherwise we hope that the "new" one
                            // we've seen might be in a register.
                            if self.ra.find_op_in_gp_reg(cur_op).is_some() {
                                continue;
                            }
                        }
                        gp_regs[usize::from(r.code())] = GPConstraint::Input {
                            op: op.clone(),
                            in_ext: RegExtension::Undefined,
                            force_reg: Some(r),
                            clobber_reg: false,
                        };
                    }
                    Register::FP(r) => {
                        fp_regs[usize::from(r.code())] = RegConstraint::InputIntoReg(op.clone(), r);
                    }
                }
            }
        }

        // Second we handle moving spill locations around.

        // We may need a temporary register to move values around, but obtaining this might force a
        // spill. We thus put off obtaining a temporary register unless we know we really need it.
        let mut tmp_reg = None;
        for (i, op) in src_ops.iter().enumerate() {
            let op = op.unpack(self.m);
            let mut src = self.op_to_var_location(op.clone());
            let dst = tgt_vars[i];
            if dst == src {
                // The value is already in the correct place.
                continue;
            }
            match dst {
                VarLocation::Stack {
                    frame_off: off_dst,
                    size: size_dst,
                } => {
                    let off_dst = i32::try_from(off_dst).unwrap();
                    // Deal with everything that doesn't need a temporary register first.
                    match src {
                        VarLocation::Register(Register::GP(reg)) => {
                            match size_dst {
                                8 => dynasm!(self.asm;
                                    mov QWORD [rbp - off_dst], Rq(reg.code())
                                ),
                                4 => dynasm!(self.asm;
                                    mov DWORD [rbp - off_dst], Rd(reg.code())
                                ),
                                _ => todo!(),
                            }
                            continue;
                        }
                        VarLocation::ConstInt { bits, v } => match bits {
                            32 => {
                                dynasm!(self.asm;
                                    mov DWORD [rbp - off_dst], v as i32
                                );
                                continue;
                            }
                            8 => {
                                dynasm!(self.asm;
                                mov BYTE [rbp - off_dst], v as i8);
                                continue;
                            }
                            _ => (),
                        },
                        VarLocation::ConstPtr(_) => (),
                        VarLocation::Stack { .. } => (),
                        e => todo!("{:?}", e),
                    }

                    // We really have to have a temporary register. Oh well.
                    if tmp_reg.is_none() {
                        tmp_reg = Some(self.ra.tmp_register_for_write_vars(&mut self.asm));
                        // The temporary register could have caused a spill which causes `src` to
                        // change its location, so recalculate.
                        src = self.op_to_var_location(op.clone());
                    }
                    let spare_reg = tmp_reg.unwrap();
                    match src {
                        // Handled in the earlier `match`.
                        VarLocation::Register(Register::GP(_)) => unreachable!(),
                        VarLocation::ConstInt { bits, v } => match bits {
                            64 => dynasm!(self.asm
                                ; mov Rq(spare_reg.code()), QWORD v.cast_signed()
                                ; mov QWORD [rbp - off_dst], Rq(spare_reg.code())
                            ),
                            // Handled in the earlier `match`.
                            _ => unreachable!(),
                        },
                        VarLocation::ConstPtr(v) => {
                            dynasm!(self.asm
                                ; mov Rq(spare_reg.code()), QWORD v as i64
                                ; mov QWORD [rbp - off_dst], Rq(spare_reg.code())
                            );
                        }
                        VarLocation::Stack {
                            frame_off: off_src,
                            size: size_src,
                        } => match size_src {
                            8 => dynasm!(self.asm
                                ; mov Rq(spare_reg.code()), QWORD [rbp - i32::try_from(off_src).unwrap()]
                                ; mov QWORD [rbp - off_dst], Rq(spare_reg.code())
                            ),
                            4 => dynasm!(self.asm
                                ; mov Rd(spare_reg.code()), DWORD [rbp - i32::try_from(off_src).unwrap()]
                                ; mov DWORD [rbp - off_dst], Rd(spare_reg.code())
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
                VarLocation::Register(_reg) => (), // Handled in the earlier loop
                _ => todo!("{dst:?}"),
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
                    self.comment("Unterminated trace".to_owned());
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
                // The end of a side-trace. Map live variables of this side-trace to the entry variables of
                // the root parent trace, then jump to it.
                self.write_jump_vars(iidx);
                self.ra.align_stack(SYSV_CALL_STACK_ALIGN);
                dynasm!(self.asm
                    // Reset rsp to the root trace's frame.
                    ; mov rsp, rbp
                    ; sub rsp, i32::try_from(sti.target_ctr.entry_sp_off()).unwrap()
                    ; mov rdi, QWORD sti.target_ctr.entry() as i64
                    // We can safely use RDI here, since the root trace won't expect live variables in this
                    // register since it's being used as an argument to the control point.
                    ; jmp rdi);
            }
            TraceKind::HeaderOnly | TraceKind::HeaderAndBody | TraceKind::Connector(_) => panic!(),
            TraceKind::DifferentFrames => panic!(),
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
        // 16-byte align this jump target to improve performance.
        let off = self.asm.offset().0;
        self.push_nops(off.next_multiple_of(16) - off);
        match self.m.tracekind() {
            TraceKind::HeaderOnly => {
                dynasm!(self.asm; ->tloop_start:);
            }
            TraceKind::HeaderAndBody => {
                dynasm!(self.asm; ->reentry:);
            }
            TraceKind::Sidetrace(_) => todo!(),
            TraceKind::Connector(_) => (),
            TraceKind::DifferentFrames => (),
        }
        self.prologue_offset = self.asm.offset();
    }

    fn cg_header_end(&mut self, iidx: InstIdx, is_connector: bool) {
        match self.m.tracekind() {
            TraceKind::HeaderOnly => {
                assert!(!is_connector);
                self.write_jump_vars(iidx);
                dynasm!(self.asm; jmp ->tloop_start);
            }
            TraceKind::HeaderAndBody => {
                assert!(!is_connector);
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
                            self.ra.assign_const_int(iidx, bits, v);
                        }
                        VarLocation::ConstPtr(v) => {
                            self.ra.assign_const_ptr(iidx, v);
                        }
                        e => panic!("{e:?}"),
                    }
                }
            }
            TraceKind::Connector(ctr) => {
                let ctr = Arc::clone(ctr)
                    .as_any()
                    .downcast::<X64CompiledTrace>()
                    .unwrap();

                self.write_jump_vars(iidx);
                self.ra.align_stack(SYSV_CALL_STACK_ALIGN);

                self.comment(format!("Jump to root trace #{}", ctr.ctrid()));
                dynasm!(self.asm
                    // Reset rsp to the root trace's frame.
                    ; mov rsp, rbp
                    ; sub rsp, i32::try_from(ctr.sp_offset).unwrap()
                    ; mov rdi, QWORD unsafe { ctr.entry().add(ctr.prologue_offset) } as i64
                    // We can safely use RDI here, since the root trace won't expect live variables in this
                    // register since it's being used as an argument to the control point.
                    ; jmp rdi);
            }
            TraceKind::Sidetrace(_) => panic!(),
            TraceKind::DifferentFrames => panic!(),
        }
    }

    fn cg_deopt(&mut self, _iidx: InstIdx, gidx: GuardInfoIdx) {
        let fail_label = self.guard_to_deopt(HasGuardInfo::Deopt(gidx));
        // Until this place is patched with a side-trace, we always forcibly deopt at
        // this point.
        dynasm!(self.asm ; jmp =>fail_label);
    }

    fn cg_return(&mut self, _iidx: InstIdx, safepoint: u64) {
        // Return from the trace naturally back into the interpreter.
        let aot_smaps = AOT_STACKMAPS.as_ref().unwrap();
        let (_, pinfo) = aot_smaps.get(usize::try_from(safepoint).unwrap());
        if !pinfo.hasfp {
            todo!();
        }
        let size = i32::try_from(pinfo.csrs.len()).unwrap() * 8;
        #[allow(clippy::fn_to_numeric_cast)]
        {
            dynasm!(self.asm
                ; mov rdi, QWORD self.m.ctrid().as_u64().cast_signed()
                ; mov rax, QWORD __yk_ret_from_trace as *const () as i64
                ; call rax
                ; mov rsp, rbp
                ; sub rsp, size
            );
        }
        // Restore callee-saved registers.
        let mut csrs = pinfo.csrs.clone();
        csrs.sort_by_key(|v| v.1);
        for (reg, _) in csrs {
            let rq = match reg {
                0 => Rq::RAX,
                15 => Rq::R15,
                14 => Rq::R14,
                13 => Rq::R13,
                12 => Rq::R12,
                3 => Rq::RBX,
                _ => panic!("Not a callee-saved register"),
            };
            dynasm!(self.asm
                ; pop Rq(rq.code())
            );
        }
        dynasm!(self.asm
            ; pop rbp
            ; ret
        );
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
        // 16-byte align this jump target to improve performance.
        let off = self.asm.offset().0;
        self.push_nops(off.next_multiple_of(16) - off);
        dynasm!(self.asm; ->tloop_start:);
    }

    fn cg_sext(&mut self, iidx: InstIdx, sinst: &jit_ir::SExtInst) {
        let [_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::AlignExtension {
                op: sinst.val(self.m),
                out_ext: RegExtension::SignExtended,
            }],
        );
    }

    fn cg_zext(&mut self, iidx: InstIdx, zinst: &jit_ir::ZExtInst) {
        let [_reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::AlignExtension {
                op: zinst.val(self.m),
                out_ext: RegExtension::ZeroExtended,
            }],
        );
    }

    fn cg_ptrtoint(&mut self, iidx: InstIdx, inst: &jit_ir::PtrToIntInst) {
        let src = inst.val(self.m);
        let src_bitw = self.m.type_(self.m.ptr_tyidx()).bitw();
        let dest_bitw = self.m.type_(inst.dest_tyidx()).bitw();

        if dest_bitw <= src_bitw {
            // A pointer is being converted to an integer the same size as, or smaller than the
            // pointer.
            let [_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [GPConstraint::InputOutput {
                    op: src,
                    in_ext: RegExtension::Undefined,
                    out_ext: RegExtension::Undefined,
                    force_reg: None,
                }],
            );
        } else {
            // A pointer is being converted to an integer larger than the pointer.
            todo!();
        }
    }

    fn cg_inttoptr(&mut self, iidx: InstIdx, inst: &jit_ir::IntToPtrInst) {
        let src = inst.val(self.m);
        let src_bitw = self.m.type_(src.tyidx(self.m)).bitw();
        let dest_bitw = self.m.type_(self.m.ptr_tyidx()).bitw();

        if src_bitw <= dest_bitw {
            // An integer the same size as a pointer, or smaller than a pointer, is being converted
            // to a pointer.
            let [_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [GPConstraint::InputOutput {
                    op: src,
                    in_ext: RegExtension::ZeroExtended,
                    out_ext: RegExtension::ZeroExtended,
                    force_reg: None,
                }],
            );
        } else {
            // An integer larger than a pointer being truncated into a pointer.
            todo!();
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
                    [GPConstraint::Input {
                        op: inst.val(self.m),
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    }],
                    [RegConstraint::Output],
                );
                // unwrap safe: IR would be invalid otherwise.
                match gp_ty.bitw().unwrap() {
                    32 => dynasm!(self.asm; cvtsi2ss Rx(tgt_reg.code()), Rd(src_reg.code())),
                    64 => dynasm!(self.asm; cvtsi2sd Rx(tgt_reg.code()), Rq(src_reg.code())),
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
            [GPConstraint::Input {
                op: inst.val(self.m),
                // 64 bit values are already sign extended, and we read 32 bit values from 32 bit
                // registers, thus we don't need to sign extend them to 64 bits.
                in_ext: RegExtension::Undefined,
                force_reg: None,
                clobber_reg: false,
            }],
            [RegConstraint::Output],
        );

        let src_bitw = inst.val(self.m).bitw(self.m);
        match self.m.type_(inst.dest_tyidx()) {
            jit_ir::Ty::Float(jit_ir::FloatTy::Float) => {
                assert_eq!(src_bitw, 32);
                dynasm!(self.asm; cvtsi2ss Rx(tgt_reg.code()), Rd(src_reg.code()));
            }
            jit_ir::Ty::Float(jit_ir::FloatTy::Double) => match src_bitw {
                32 => dynasm!(self.asm; cvtsi2sd Rx(tgt_reg.code()), Rd(src_reg.code())),
                64 => dynasm!(self.asm; cvtsi2sd Rx(tgt_reg.code()), Rq(src_reg.code())),
                _ => todo!(),
            },
            _ => panic!(),
        }
    }

    fn cg_uitofp(&mut self, iidx: InstIdx, inst: &jit_ir::UIToFPInst) {
        let ([src_reg], [tgt_reg, tmp_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::Input {
                op: inst.val(self.m),
                // 64 bit values are already sign extended, and we read 32 bit values from 32 bit
                // registers, thus we don't need to sign extend them to 64 bits.
                in_ext: RegExtension::Undefined,
                force_reg: None,
                clobber_reg: false,
            }],
            [RegConstraint::Output, RegConstraint::Temporary],
        );

        let src_bitw = inst.val(self.m).bitw(self.m);
        match self.m.type_(inst.dest_tyidx()) {
            jit_ir::Ty::Float(jit_ir::FloatTy::Float) => todo!(),
            jit_ir::Ty::Float(jit_ir::FloatTy::Double) => match src_bitw {
                64 => {
                    // This is a port of what clang does when you cast a `uint64_t` to a `double`.
                    // It relies on loading magic constants from memory.
                    //
                    // FIXME: There's no need to repeatedly emit this data for every conversion.
                    let const0: [u32; 4] = [1127219200, 1160773632, 0, 0];
                    let const0_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            const0.as_ptr() as *const u8,
                            const0.len() * std::mem::size_of::<u32>(),
                        )
                    };
                    let const1: [u64; 2] = [0x4330000000000000, 0x4530000000000000];
                    let const1_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            const1.as_ptr() as *const u8,
                            const1.len() * std::mem::size_of::<u64>(),
                        )
                    };
                    dynasm!(self.asm
                        ; jmp >body
                        ; .align 16
                        ; lcpi0_0:
                        ;  .bytes const0_bytes
                        ; lcpi0_1:
                        ;  .bytes const1_bytes
                        ; body:
                        ; movq Rx(tmp_reg.code()), Rq(src_reg.code())
                        ; punpckldq Rx(tmp_reg.code()), [<lcpi0_0]
                        ; subpd Rx(tmp_reg.code()), [<lcpi0_1]
                        ; movapd  Rx(tgt_reg.code()), Rx(tmp_reg.code())
                        ; unpckhpd Rx(tgt_reg.code()), Rx(tmp_reg.code())
                        ; addsd Rx(tgt_reg.code()), Rx(tmp_reg.code())
                    );
                }
                _ => todo!(),
            },
            _ => panic!(),
        }
    }

    fn cg_fptosi(&mut self, iidx: InstIdx, inst: &jit_ir::FPToSIInst) {
        let from_val = inst.val(self.m);
        let to_ty = self.m.type_(inst.dest_tyidx());
        // Unwrap cannot fail: floats and integers are sized.
        let from_bitw = from_val.bitw(self.m);
        let to_bitw = to_ty.bitw().unwrap();

        // FIXME: If we cast to a larger-sized integer type, we will need to sign extend the value
        // to keep the same numeric value.
        assert!(to_bitw <= from_bitw);

        let ([tgt_reg], [src_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::Output {
                out_ext: RegExtension::SignExtended,
                force_reg: None,
                can_be_same_as_input: false,
            }],
            [RegConstraint::Input(from_val)],
        );

        match from_bitw {
            32 => dynasm!(self.asm; cvttss2si Rq(tgt_reg.code()), Rx(src_reg.code())),
            64 => dynasm!(self.asm; cvttsd2si Rq(tgt_reg.code()), Rx(src_reg.code())),
            _ => panic!(),
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
        let [reg] = self.ra.assign_gp_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::InputOutput {
                op: i.val(self.m),
                in_ext: RegExtension::Undefined,
                out_ext: RegExtension::ZeroExtended,
                force_reg: None,
            }],
        );
        self.ra.force_zero_extend_to_reg64(
            &mut self.asm,
            reg,
            self.m.type_(i.dest_tyidx()).bitw().unwrap(),
        );
    }

    fn cg_select(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::SelectInst) {
        // First load the true case. We then immediately follow this up with a conditional move,
        // overwriting the value with the false case, if the condition was false.
        match self.m.type_(inst.trueval(self.m).tyidx(self.m)) {
            Ty::Void => todo!(),
            Ty::Integer(_) | Ty::Ptr => self.cg_select_int_ptr(iidx, inst),
            Ty::Func(_) => todo!(),
            Ty::Float(_) => self.cg_select_float(iidx, inst),
            Ty::Struct(_) => todo!(),
            Ty::Unimplemented(_) => unreachable!(),
        }
    }

    fn cg_select_int_ptr(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::SelectInst) {
        let condval = inst.cond(self.m);
        assert_eq!(condval.bitw(self.m), 1);
        let trueval = inst.trueval(self.m);
        let falseval = inst.falseval(self.m);
        assert_eq!(trueval.bitw(self.m), falseval.bitw(self.m));
        if trueval.bitw(self.m) == 1
            && let Some(c) = self.op_to_zero_ext_i8(&trueval)
        {
            let [cond_reg, val_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [
                    GPConstraint::InputOutput {
                        op: condval,
                        in_ext: RegExtension::Undefined,
                        out_ext: RegExtension::Undefined,
                        force_reg: None,
                    },
                    GPConstraint::Input {
                        op: falseval,
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    },
                ],
            );
            match c {
                0 => dynasm!(self.asm
                    ; not Rd(cond_reg.code())
                    ; and Rd(cond_reg.code()), Rd(val_reg.code())
                ),
                1 => dynasm!(self.asm; or Rd(cond_reg.code()), Rd(val_reg.code())),
                _ => unreachable!(),
            }
        } else if trueval.bitw(self.m) == 1
            && let Some(c) = self.op_to_zero_ext_i8(&falseval)
        {
            let [cond_reg, val_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [
                    GPConstraint::InputOutput {
                        op: condval,
                        in_ext: RegExtension::Undefined,
                        out_ext: RegExtension::Undefined,
                        force_reg: None,
                    },
                    GPConstraint::Input {
                        op: trueval,
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    },
                ],
            );
            match c {
                0 => dynasm!(self.asm; and Rd(cond_reg.code()), Rd(val_reg.code())),
                1 => dynasm!(self.asm
                    ; not Rd(cond_reg.code())
                    ; or Rd(cond_reg.code()), Rd(val_reg.code())
                ),
                _ => unreachable!(),
            }
        } else {
            let [cond_reg, true_reg, false_reg] = self.ra.assign_gp_regs(
                &mut self.asm,
                iidx,
                [
                    GPConstraint::Input {
                        op: condval,
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    },
                    GPConstraint::InputOutput {
                        op: trueval,
                        in_ext: RegExtension::Undefined,
                        out_ext: RegExtension::Undefined,
                        force_reg: None,
                    },
                    GPConstraint::Input {
                        op: inst.falseval(self.m),
                        in_ext: RegExtension::Undefined,
                        force_reg: None,
                        clobber_reg: false,
                    },
                ],
            );
            dynasm!(self.asm
                ; bt Rd(cond_reg.code()), 0
                ; cmovnc Rq(true_reg.code()), Rq(false_reg.code())
            );
        }
    }

    fn cg_select_float(&mut self, iidx: jit_ir::InstIdx, inst: &jit_ir::SelectInst) {
        let ([cond_reg], [true_reg, false_reg, out_reg]) = self.ra.assign_regs(
            &mut self.asm,
            iidx,
            [GPConstraint::Input {
                op: inst.cond(self.m),
                in_ext: RegExtension::Undefined,
                force_reg: None,
                clobber_reg: false,
            }],
            [
                RegConstraint::Input(inst.trueval(self.m)),
                RegConstraint::Input(inst.falseval(self.m)),
                RegConstraint::Output,
            ],
        );
        debug_assert_eq!(
            self.m
                .type_(inst.cond(self.m).tyidx(self.m))
                .bitw()
                .unwrap(),
            1
        );

        assert_eq!(
            inst.trueval(self.m).bitw(self.m),
            inst.falseval(self.m).bitw(self.m)
        );
        match inst.trueval(self.m).bitw(self.m) {
            32 => {
                dynasm!(self.asm
                    ;   bt Rd(cond_reg.code()), 0
                    ;   jc >equal
                    ;   movss Rx(out_reg.code()), Rx(false_reg.code())
                    ;   jmp >done
                    ; equal:
                    ;   movss Rx(out_reg.code()), Rx(true_reg.code())
                    ; done:
                );
            }
            64 => {
                dynasm!(self.asm
                    ;   bt Rd(cond_reg.code()), 0
                    ;   jc >equal
                    ;   movsd Rx(out_reg.code()), Rx(false_reg.code())
                    ;   jmp >done
                    ; equal:
                    ;   movsd Rx(out_reg.code()), Rx(true_reg.code())
                    ; done:
                );
            }
            x => todo!("{x}"),
        }
    }

    fn guard_to_deopt(&mut self, hgi: HasGuardInfo) -> DynamicLabel {
        let fail_label = self.asm.new_dynamic_label();
        let ginfo = hgi.guard_info(self.m);
        let gd = CompilingGuard {
            ginfo: hgi,
            guard_snapshot: self.ra.guard_snapshot(),
            bid: *ginfo.bid(),
            fail_label,
            // We don't know the offset yet but will fill this in later.
            fail_offset: AssemblyOffset(0),
            inlined_frames: ginfo.inlined_frames().to_vec(),
        };
        self.guards.push(gd);
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
            [GPConstraint::Temporary],
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
        let cond = inst.cond(self.m);
        let reg = self.ra.tmp_register_for_guard(&mut self.asm, iidx, cond);
        let fail_label = self.guard_to_deopt(HasGuardInfo::Guard(*inst));
        dynasm!(self.asm ; bt Rd(reg.code()), 0);
        if inst.expect() {
            dynasm!(self.asm ; jnb =>fail_label);
        } else {
            dynasm!(self.asm ; jb =>fail_label);
        }
    }
}

/// Information required by guards while we're compiling them.
#[derive(Debug)]
struct CompilingGuard {
    ginfo: HasGuardInfo,
    guard_snapshot: GuardSnapshot,
    /// The AOT block that the failing guard originated from.
    bid: aot_ir::BBlockId,
    fail_label: DynamicLabel,
    fail_offset: AssemblyOffset,
    inlined_frames: Vec<InlinedFrame>,
}

/// A compiled guard: contains information required by deoptimisation.
#[derive(Debug)]
struct CompiledGuard {
    /// The AOT block that the failing guard originated from.
    bid: aot_ir::BBlockId,
    fail_offset: AssemblyOffset,
    /// Live variables, mapping AOT vars to JIT vars.
    live_vars: Vec<(aot_ir::InstId, VarLocation)>,
    inlined_frames: Vec<InlinedFrame>,
    /// Keeps track of deopt amount and compiled side-trace.
    guard: Guard,
}

#[derive(Debug)]
pub(crate) struct X64CompiledTrace {
    /// This trace's [TraceId].
    ctrid: TraceId,
    // Reference to the meta-tracer required for side tracing.
    mt: Arc<MT>,
    /// For connector traces, the matching [DeoptSafePoint].
    safepoint: Option<DeoptSafepoint>,
    /// The executable code itself.
    buf: ExecutableBuffer,
    /// Information about compiled guards; mostly used for deoptimisation.
    compiled_guards: Vec<CompiledGuard>,
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

    /// Return the [CompiledGuard] at `gid`.
    fn compiled_guard(&self, gid: GuardId) -> &CompiledGuard {
        &self.compiled_guards[usize::from(gid)]
    }

    pub(crate) fn sidetraceinfo(
        &self,
        gid: GuardId,
        target_ctr: Arc<dyn CompiledTrace>,
    ) -> Arc<YkSideTraceInfo<Register>> {
        let target_ctr = target_ctr.as_any().downcast::<X64CompiledTrace>().unwrap();
        // FIXME: Can we reference these instead of copying them, e.g. by passing in a reference to
        // the `CompiledTrace` and `gid` or better a reference to `DeoptInfo`?
        let gd = &self.compiled_guards[usize::from(gid)];
        let lives = gd
            .live_vars
            .iter()
            .map(|(a, l)| (a.clone(), l.into()))
            .collect();
        let callframes = gd.inlined_frames.clone();

        Arc::new(YkSideTraceInfo {
            bid: gd.bid,
            lives,
            callframes,
            entry_vars: target_ctr.entry_vars().to_vec(),
            sp_offset: self.sp_offset,
            target_ctr,
        })
    }
}

impl CompiledTrace for X64CompiledTrace {
    fn ctrid(&self) -> TraceId {
        self.ctrid
    }

    fn safepoint(&self) -> &Option<DeoptSafepoint> {
        &self.safepoint
    }

    fn entry(&self) -> *const libc::c_void {
        self.buf.ptr(AssemblyOffset(0)) as *const libc::c_void
    }

    fn entry_sp_off(&self) -> usize {
        self.sp_offset
    }

    fn guard(&self, gid: GuardId) -> &crate::compile::Guard {
        &self.compiled_guards[usize::from(gid)].guard
    }

    /// Patch the address of a side-trace directly into the parent trace.
    /// * `gid`: The guard to be patched.
    /// * `staddr`: The address of the side-trace.
    fn patch_guard(&self, gid: GuardId, staddr: *const std::ffi::c_void) {
        // Since we have to temporarily make the parent trace writable, another thread trying
        // to patch this trace could interfere with that. Having this lock prevents this.
        let _lock = LK_PATCH.lock();

        // Calculate a pointer to the address we want to patch.
        let patch_offset = self.compiled_guards[usize::from(gid)].fail_offset;
        // Add 2 bytes to get to the address operand of the mov instruction.
        let patch_addr = unsafe { self.buf.ptr(patch_offset).offset(2) };
        // FIXME: Is it better/faster to protect the entire buffer in one go and then patch each
        // address, rather than mark single pages writeable, patch, and mark readable again?
        patch_addresses(&[(patch_addr, staddr as u64)]);
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

    fn code(&self) -> &[u8] {
        &self.buf
    }

    fn name(&self) -> String {
        format!("__yk_trace_{}", self.ctrid().as_u64())
    }
}

/// Patch multiple addresses in contiguous memory. The slice is a sequence of pairs `(tgt, val)`:
/// - `tgt`: The address we want to patch. Needs to be 8-byte aligned and must not cross a
///   cache-line boundary.
/// - `val`: The value to be written to `tgt`.
///
/// This function `mprotect`s a single chunk of memory, potentially spanning multiple pages: all
/// the addresses passed in the slice *must* be in contiguously allocated pages. Failing to do so
/// will lead to undefined behaviour (including poor performance and outright crashes).
///
/// This function does not perform any locking but is not itself thread safe: if you call this in a
/// context where other threads may also try to call this function, you must use an external lock
/// to ensure that two instances of this function cannot run simultaneously doing so. Failing to do
/// so will lead to undefined behaviour.
fn patch_addresses(patches: &[(*const u8, u64)]) {
    if patches.is_empty() {
        return;
    }

    // Calculate the lowest page address in the patches as `mprotect` requires a page-aligned
    // address.
    let page_size = page_size::get();
    let low = patches
        .iter()
        .map(|(x, _)| ((*x as usize) / page_size) * page_size)
        .min()
        .unwrap();
    // Calculate the number of bytes from the lowest page address we will need to `mprotect`.
    let high = patches.iter().map(|(x, _)| *x as usize).max().unwrap();
    let len = high - low;

    // Mark the range of memory as writeable.
    if unsafe {
        libc::mprotect(
            low as *mut libc::c_void,
            len,
            libc::PROT_EXEC | libc::PROT_READ | libc::PROT_WRITE,
        )
    } != 0
    {
        todo!();
    }

    // Patch addresses.
    for (tgt, val) in patches {
        // The target address must be 8-byte aligned so that we do a single write. By definition,
        // this means that we also cannot span a cache-line.
        assert!((*tgt as usize).is_multiple_of(8));
        unsafe { *(*tgt as *mut u64) = *val };
    }
    // This `fence` serves two purposes:
    //   1. In all cases, it makes sure that the compiler doesn't remove any of the writes in the
    //      `for` loop.
    //   2. In multi-threaded contexts, it will ensure that other threads using the accompanying
    //      lock have the same view of memory.
    //
    // Note: this `fence` does not, and cannot, force other threads to immediately see the same
    // view of memory as this thread. In other words, other threads executing the same machine code
    // may go arbitrarily long without observing the writes performed in this thread.
    fence(std::sync::atomic::Ordering::Release);

    // Mark the range of memory as unwriteable.
    if unsafe {
        libc::mprotect(
            low as *mut libc::c_void,
            len,
            libc::PROT_EXEC | libc::PROT_READ,
        )
    } != 0
    {
        todo!();
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
        let start_ip = u64::try_from(bptr.addr()).unwrap();
        let code = unsafe { slice::from_raw_parts(bptr, len) };
        let fmt = zydis::Formatter::intel();
        let dec = zydis::Decoder::new64();
        for insn_info in dec.decode_all::<zydis::VisibleOperands>(code, start_ip) {
            let (ip, _raw_bytes, insn) = insn_info.unwrap();
            let off = ip - start_ip;
            if let Some(lines) = self.comments.get(&usize::try_from(off).unwrap()) {
                for line in lines {
                    out.push(format!("; {line}"));
                }
            }
            let istr = fmt.format(Some(ip), &insn).unwrap();
            if self.with_addrs {
                out.push(format!("{ip:016x} {off:08x}: {istr}"));
            } else {
                out.push(istr.to_string());
            }
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
    use crate::{
        compile::{
            CompiledTrace,
            jitc_yk::jit_ir::{self, Inst, Module, ParamIdx, TraceKind},
        },
        location::{HotLocation, HotLocationKind},
        mt::{MT, TraceId},
    };
    use fm::{FMBuilder, FMatcher};
    use lazy_static::lazy_static;
    use parking_lot::Mutex;
    use regex::{Regex, RegexBuilder};
    use smallvec::smallvec;
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
                    if !r.is_empty() {
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

        static ref FP_REG_IGNORE_RE: Regex = {
            Regex::new(r"fp\.128\._").unwrap()
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

    fn fmatcher(ptn: &str) -> FMatcher<'_> {
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
            .name_matcher_ignore(FP_REG_IGNORE_RE.clone(), FP_REG_TEXT_RE.clone())
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

        let fmm = fmatcher("fp.128.x fp.128.y fp.128._");
        assert!(fmm.matches("xmm0 xmm1 xmm0").is_ok());
        assert!(fmm.matches("xmm0 xmm0 xmm0").is_err());
        assert!(fmm.matches("xmm0 xmm1 xmm2").is_ok());
        assert!(fmm.matches("xmm0 xmm1 xmm0").is_ok());
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
            kind: HotLocationKind::Tracing(mt.next_trace_id()),
            tracecompilation_errors: 0,
            #[cfg(feature = "ykd")]
            debug_str: None,
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
            full_asm,
        );
    }

    #[test]
    fn cg_load_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: ptr = load %0
                black_box %1
            ",
            "
                ...
                ; %1: ptr = load %0
                mov r.64.x, [rax]
                ",
            false,
        );
    }

    #[test]
    fn cg_load_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i8 = load %0
                black_box %1
            ",
            "
                ...
                ; %1: i8 = load %0
                movzx r.32.x, byte ptr [rax]
                ",
            false,
        );
    }

    #[test]
    fn cg_load_i32() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param reg
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
    fn cg_load_bin_store() {
        // Check it optimises when it should
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i8 = load %0
                %2: i8 = add %1, 1i8
                *%0 = %2
            ",
            "
                ...
                ; %1: i8 = load %0
                ; %2: i8 = add %1, 1i8
                ; *%0 = %2
                add byte ptr [r.64._], 0x01
                ",
            false,
        );
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: ptr = param reg
                %2: i64 = load %0
                %3: i64 = add %2, 2147483647i64
                *%0 = %3
                %5: i64 = load %1
                %6: i64 = add %5, 18446744073709551615i64
                *%1 = %6
            ",
            "
                ...
                ; %2: i64 = load %0
                ; %3: i64 = add %2, 2147483647i64
                ; *%0 = %3
                add qword ptr [r.64._], 0x7FFFFFFF
                ; %5: i64 = load %1
                ; %6: i64 = add %5, 18446744073709551615i64
                ; *%1 = %6
                add qword ptr [r.64._], 0xffffffffffffffff
                ",
            false,
        );

        // Check it doesn't optimise when it shouldn't
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i8 = load %0
                %2: i8 = add %1, 1i8
                *%0 = %2
                black_box %1
            ",
            "
                ...
                ; %1: i8 = load %0
                movzx r.32.x, byte ptr [r.64.y]
                ; %2: i8 = add %1, 1i8
                mov r.64._, r.64.x
                add r.32.x, 0x01
                ; *%0 = %2
                and r.32.x, 0xff
                mov [r.64.y], r.8.x
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i8 = load %0
                %2: i8 = add %1, 1i8
                *%0 = %2
                black_box %2
            ",
            "
                ...
                ; %1: i8 = load %0
                movzx r.32.x, byte ptr [r.64.y]
                ; %2: i8 = add %1, 1i8
                add r.32.x, 0x01
                ; *%0 = %2
                and r.32.x, 0xff
                mov [r.64.y], r.8.x
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i8 = param reg
                %2: i8 = load %0
                %3: i8 = add %1, %2
                *%0 = %3
            ",
            "
                ...
                ; %2: i8 = load %0
                movzx r.32.x, byte ptr [rax]
                ; %3: i8 = add %1, %2
                add r.32.y, r.32.x
                ; *%0 = %3
                and r.32.y, 0xff
                mov [rax], r.8.y
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i64 = load %0
                %2: i64 = add %1, 2147483648i64
                *%0 = %2
                black_box %2
            ",
            "
                ...
                ; %1: i64 = load %0
                mov r.64.a, [r.64._]
                ; %2: i64 = add %1, 2147483648i64
                mov r.64.b, 0x80000000
                add r.64.a, r.64.b
                ; *%0 = %2
                mov [rax], r.64.a
                ",
            false,
        );
    }

    #[test]
    fn cg_store_const_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                *%0 = 0x0
            ",
            "
                ...
                ; *%0 = 0x0
                mov qword ptr [rax], 0x00
                ",
            false,
        );
    }

    #[test]
    fn cg_const_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i1 = eq %0, 0x1234
                black_box %1
            ",
            "
                ...
                ; %1: i1 = eq %0, 0x1234
                cmp rax, 0x1234
                setz r.8._
            ",
            false,
        );
    }

    #[test]
    fn cg_ptradd() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: ptr = ptr_add %0, 64
                %2: ptr = ptr_add %0, -1
                %3: ptr = ptr_add %0, 2147483647
                black_box %1
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %1: ptr = ptr_add %0, 64
                lea r.64.x, [r.64.p+0x40]
                ; %2: ptr = ptr_add %0, -1
                lea r.64.y, [r.64.p-0x01]
                ; %3: ptr = ptr_add %0, 2147483647
                lea r.64.p, [r.64.p+0x7fffffff]
                ",
            false,
        );
    }

    #[test]
    fn cg_ptradd_load() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: ptr = param reg
                %2: ptr = ptr_add %0, 64
                %3: i64 = load %2
                %4: ptr = ptr_add %1, 32
                %5: i64 = load %4
                %6: ptr = ptr_add %4, 1
                %7: ptr = ptr_add %4, -1
                %8: i64 = load %7
                %9: ptr = ptr_add %4, 2147483647
                %10: i64 = load %9
                black_box %3
                black_box %5
                black_box %6
                black_box %8
                black_box %10
            ",
            "
                ...
                ; %1: ...
                ; %3: i64 = load %0 + 64
                mov r.64.x, [r.64._+{{_}}]
                ; %4: ptr = ptr_add %1, 32
                lea r.64.y, [r.64.z+0x20]
                ; %5: i64 = load %1 + 32
                mov r.64._, [r.64.z+0x20]
                ; %6: ptr = ptr_add %4, 1
                lea r.64._, [r.64.y+0x01]
                ; %8: i64 = load %4 + -1
                mov r.64._, [r.64.y-0x01]
                ; %10: i64 = load %4 + 2147483647
                mov r.64._, [r.64.y+0x7fffffff]
                ",
            false,
        );
    }

    #[test]
    fn cg_ptradd_store() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: ptr = param reg
                %2: ptr = ptr_add %0, 64
                *%2 = 1i8
                %4: ptr = ptr_add %1, 32
                %5: i64 = load %4
                *%4 = 2i8
                %7: ptr = ptr_add %0, -1
                *%7 = 255i8
                %9: ptr = ptr_add %0, 2147483647
                *%9 = 65535i16
                black_box %5
            ",
            "
                ...
                ; *(%0 + 64) = 1i8
                mov byte ptr [rax+{{_}}], 0x01
                ; %5: i64 = load %1 + 32
                mov r.64.y, [r.64.x+0x20]
                ; *(%1 + 32) = 2i8
                mov byte ptr [r.64.x+0x20], 0x02
                ; *(%0 + -1) = 255i8
                mov byte ptr [r.64._-0x01], 0xff
                ; *(%0 + 2147483647) = 65535i16
                mov word ptr [r.64._+0x7fffffff], 0xffff
                ",
            false,
        );
    }

    #[test]
    fn cg_dynptradd() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i32 = param reg
                %2: ptr = dyn_ptr_add %0, %1, 1
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 1
                mov r.32.x, r.32.x
                lea r.64._, [r.64._+r.64.x*1]
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i32 = param reg
                %2: ptr = dyn_ptr_add %0, %1, 2
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 2
                mov r.32.x, r.32.x
                lea r.64._, [r.64._+r.64.x*2]
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i32 = param reg
                %2: ptr = dyn_ptr_add %0, %1, 4
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 4
                mov r.32.x, r.32.x
                lea r.64._, [r.64._+r.64.x*4]
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i32 = param reg
                %2: ptr = dyn_ptr_add %0, %1, 5
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 5
                mov r.32.x, r.32.x
                imul r.64.x, r.64.x, 0x05
                add r.64.x, r.64._
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i32 = param reg
                %2: ptr = dyn_ptr_add %0, %1, 16
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 16
                mov r.32.x, r.32.x
                shl r.64.x, 0x04
                add r.64.x, r.64._
                ...
                ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i32 = param reg
                %2: ptr = dyn_ptr_add %0, %1, 77
                black_box %2
            ",
            "
                ...
                ; %2: ptr = dyn_ptr_add %0, %1, 77
                mov r.32.x, r.32.x
                imul r.64.x, r.64.x, 0x4d
                add r.64.x, r.64._
                ",
            false,
        );
    }

    #[test]
    fn cg_store_ptr() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: ptr = param reg
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
                %0: ptr = param reg
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
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i16 = add %0, %1
                %3: i16 = add %2, 1i16
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: i16 = add %0, %1
                add r.32.x, r.32.y
                ; %3: i16 = add %2, 1i16
                ......
                add r.32.x, 0x01
                ",
            false,
        );
    }

    #[test]
    fn cg_add_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param reg
                %1: i64 = param reg
                %2: i64 = add %0, %1
                %3: i64 = add %2, 1i64
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: i64 = add %0, %1
                add r.64.x, r.64.y
                ; %3: i64 = add %2, 1i64
                ......
                add r.64.x, 0x01
                ",
            false,
        );
    }

    #[test]
    fn cg_const_add_one_i64() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param reg
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
                %0: i64 = param reg
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
                %0: i64 = param reg
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
                %0: i64 = param reg
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
                %0: i32 = param reg
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
                %0: i32 = param reg
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
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i1 = param reg
                %5: i16 = and %0, %1
                %6: i32 = and %2, %3
                %7: i1 = and %4, 1i1
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %5: i16 = and %0, %1
                and r.32.a, r.32.b
                ; %6: i32 = and %2, %3
                and r.32.c, r.32.d
                ; %7: i1 = and %4, 1i1
                and r.32._, 0x01
                ...
                ",
            false,
        );

        // Check that AND implicitly zero extends
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i8 = param reg
                %2: i64 = param reg
                %3: i1 = eq %0, 2i8
                %4: i8 = and %1, 3i8
                %5: i1 = eq %4, 4i8
                %6: i64 = and %2, 2147483647i64
                %7: i64 = and %2, 2147483648i64
                black_box %3
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %3: i1 = eq %0, 2i8
                and r.32.a, 0xff
                cmp r.32.a, 0x02
                setz ...
                ; %4: i8 = and %1, 3i8
                and r.32.b, 0x03
                ; %5: i1 = eq %4, 4i8
                cmp r.32.b, 0x04
                setz ...
                ; %6: i64 = and %2, 2147483647i64
                mov r.64.x, r.64.y
                and r.64.y, 0x7fffffff
                ; %7: i64 = and %2, 2147483648i64
                mov r.64.z, 0x80000000
                and r.64.x, r.64.z
                ",
            false,
        );
    }

    #[test]
    fn cg_ashr() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i16 = ashr %0, 1i16
                %3: i32 = ashr %1, 2i32
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: i16 = ashr %0, 1i16
                movsx r.64.a, r.16.a
                sar r.64.a, 0x01
                ; %3: i32 = ashr %1, 2i32
                movsxd r.64.b, r.32.b
                sar r.64.b, 0x02
                ",
            false,
        );
    }

    #[test]
    fn cg_lshr() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i16 = lshr %0, 1i16
                %3: i32 = lshr %1, 2i32
                black_box %2
                black_box %3
            ",
            "
                ...
                ; %2: i16 = lshr %0, 1i16
                and r.32.a, 0xffff
                shr r.64.a, 0x01
                ; %3: i32 = lshr %1, 2i32
                mov r.32.b, r.32.b
                shr r.64.b, 0x02
                ",
            false,
        );
    }

    #[test]
    fn cg_shl() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i16 = shl %0, 1i16
                %3: i32 = shl %1, 2i32
                %4: i32 = shl %1, %3
                black_box %2
                black_box %3
                black_box %4
            ",
            "
                ...
                ; %2: i16 = shl %0, 1i16
                shl rax, 0x01
                ; %3: i32 = shl %1, 2i32
                ......
                shl rcx, 0x02
                ; %4: i32 = shl %1, %3
                ...
                shl r.64._, cl
                ",
            false,
        );
    }

    #[test]
    fn cg_mul() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i16 = mul %0, %1
                %5: i32 = mul %2, %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %4: i16 = mul %0, %1
                mov r.64.a, rdx
                and eax, 0xffff
                and ecx, 0xffff
                mul rcx
                ; %5: i32 = mul %2, %3
                ...
                mul r.64.b
                ",
            false,
        );
    }

    #[test]
    fn cg_or() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i1 = param reg
                %5: i64 = param reg
                %6: i64 = param reg
                %7: i16 = or %0, %1
                %8: i32 = or %2, %3
                %9: i1 = or %4, 1i1
                %10: i64 = or %5, %6
                black_box %7
                black_box %8
                black_box %9
                black_box %10
            ",
            "
                ...
                ; %7: i16 = or %0, %1
                or r.32.a, r.32.b
                ; %8: i32 = or %2, %3
                or r.32.c, r.32.d
                ; %9: i1 = or %4, 1i1
                or r.32.e, 0x01
                ; %10: i64 = or %5, %6
                or r.64.f, r.64.g
                ",
            false,
        );

        // Check that OR implicitly zero extends
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i8 = param reg
                %2: i64 = param reg
                %3: i1 = eq %0, 2i8
                %4: i8 = or %1, 3i8
                %5: i1 = eq %4, 4i8
                %6: i64 = or %2, 2147483647i64
                %7: i64 = or %2, 2147483648i64
                black_box %3
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %3: i1 = eq %0, 2i8
                and r.32.a, 0xff
                cmp r.32.a, 0x02
                setz ...
                ; %4: i8 = or %1, 3i8
                or r.32.b, 0x03
                ; %5: i1 = eq %4, 4i8
                and r.32.b, 0xff
                cmp r.32.b, 0x04
                setz ...
                ; %6: i64 = or %2, 2147483647i64
                mov r.64.x, r.64.y
                or r.64.y, 0x7fffffff
                ; %7: i64 = or %2, 2147483648i64
                mov r.64.z, 0x80000000
                or r.64.x, r.64.z
                ",
            false,
        );
    }

    #[test]
    fn cg_sdiv() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i16 = sdiv %0, %1
                %5: i32 = sdiv %2, %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %4: i16 = sdiv %0, %1
                ...
                movsx rax, ax
                movsx r.64.a, r.16.a
                cqo
                idiv r.64.a
                ; %5: i32 = sdiv %2, %3
                ...
                cqo
                idiv r.64.b
                ",
            false,
        );
    }

    #[test]
    fn cg_srem() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i16 = srem %0, %1
                %5: i32 = srem %2, %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %4: i16 = srem %0, %1
                ......
                movsx rax, ax
                movsx r.64.a, r.16.a
                cqo
                idiv r.64.a
                ; %5: i32 = srem %2, %3
                ...
                movsxd rax, eax
                movsxd r.64.b, r.32.b
                cqo
                idiv r.64.b
                ",
            false,
        );
    }

    #[test]
    fn cg_sub() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i16 = sub %0, %1
                %5: i32 = sub %2, %3
                %6: i32 = sub 0i32, %5
                black_box %4
                black_box %5
                black_box %6
            ",
            "
                ...
                ; %4: i16 = sub %0, %1
                movsx r.64.a, r.16.a
                movsx r.64.b, r.16.b
                sub r.64.a, r.64.b
                ; %5: i32 = sub %2, %3
                movsxd r.64.c, r.32.c
                movsxd r.64.d, r.32.d
                sub r.64.c, r.64.d
                ; %6: i32 = sub 0i32, %5
                ......
                neg r.64.c
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
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i1 = param reg
                %5: i16 = xor %0, %1
                %6: i32 = xor %2, %3
                %7: i1 = xor %4, 1i1
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %5: i16 = xor %0, %1
                xor r.32.a, r.32.b
                ; %6: i32 = xor %2, %3
                xor r.32.c, r.32.d
                ; %7: i1 = xor %4, 1i1
                xor r.32.e, 0x01
                ",
            false,
        );

        // Check that XOR implicitly zero extends
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i8 = param reg
                %2: i64 = param reg
                %3: i1 = eq %0, 2i8
                %4: i8 = xor %1, 3i8
                %5: i1 = eq %4, 4i8
                %6: i64 = xor %2, 2147483647i64
                %7: i64 = xor %2, 2147483648i64
                black_box %3
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %3: i1 = eq %0, 2i8
                and r.32.a, 0xff
                cmp r.32.a, 0x02
                setz ...
                ; %4: i8 = xor %1, 3i8
                xor r.32.b, 0x03
                ; %5: i1 = eq %4, 4i8
                and r.32.b, 0xff
                cmp r.32.b, 0x04
                setz ...
                ; %6: i64 = xor %2, 2147483647i64
                mov r.64.x, r.64.y
                xor r.64.y, 0x7fffffff
                ; %7: i64 = xor %2, 2147483648i64
                mov r.64.z, 0x80000000
                xor r.64.x, r.64.z
                ",
            false,
        );
    }

    #[test]
    fn cg_udiv() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i16 = param reg
                %2: i32 = param reg
                %3: i32 = param reg
                %4: i16 = udiv %0, %1
                %5: i32 = udiv %2, %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %4: i16 = udiv %0, %1
                mov r.64.a, r.64._
                and eax, 0xffff
                and r.32.b, 0xffff
                xor rdx, rdx
                div r.64.b
                ; %5: i32 = udiv %2, %3
                ...
                xor rdx, rdx
                div r.64.c
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
              func_decl puts (i64, i64, i64)

              entry:
                %0: i64 = param reg
                %1: i64 = param reg
                %2: i64 = param reg
                call @puts(%0, %1, %2)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2)
                mov rsi, rcx
                mov rdi, rax
                mov r.64.tgt, 0x{sym_addr:X}
                call r.64.tgt
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
              func_decl puts(i8, i16, ptr, i1)

              entry:
                %0: i8 = param reg
                %1: i16 = param reg
                %2: ptr = param reg
                %3: i1 = param reg
                call @puts(%0, %1, %2, %3)
            ",
            &format!(
                "
                ...
                ; call @puts(%0, %1, %2, %3)
                mov rsi, rcx
                mov rcx, rbx
                mov rdi, rax
                and ecx, 0x01
                and esi, 0xffff
                and edi, 0xff
                mov r.64.tgt, 0x{sym_addr:X}
                call r.64.tgt
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
    fn cg_call_hints() {
        codegen_and_test(
            "
             func_decl llvm.assume (i1)
             func_decl llvm.lifetime.start.p0 (i64, ptr)
             func_decl llvm.lifetime.end.p0 (i64, ptr)
             entry:
               %0: i1 = param reg
               %1: ptr = param reg
               call @llvm.assume(%0)
               call @llvm.lifetime.start.p0(16i64, %1)
               call @llvm.lifetime.end.p0(16i64, %1)
               %5: ptr = ptr_add %1, 1
               black_box %5
            ",
            "
               ...
               ; call @llvm.assume(%0)
               ; call @llvm.lifetime.start.p0(16i64, %1)
               ; call @llvm.lifetime.end.p0(16i64, %1)
               ; %5: ...
               ...
            ",
            false,
        );
    }

    #[test]
    fn cg_call_abs() {
        codegen_and_test(
            "
             func_decl llvm.abs.i64 (i64, i1) -> i64
             entry:
               %0: i64 = param reg
               %1: i64 = call @llvm.abs.i64(%0, 0i1)
               %2: i64 = call @llvm.abs.i64(%0, 1i1)
               black_box %1
               black_box %2
            ",
            "
               ...
               ; %1: i64 = call @llvm.abs.i64(%0, 0i1)
               mov r.64.y, rax
               mov r.64.x, rax
               neg rax
               cmovl rax, r.64.x
               ; %2: i64 = call @llvm.abs.i64(%0, 1i1)
               mov r.64.x, r.64.y
               neg r.64.y
               cmovl r.64.y, r.64.x
            ",
            false,
        );
    }

    #[test]
    fn cg_call_ctpop() {
        codegen_and_test(
            "
             func_decl llvm.ctpop.i32 (i32) -> i32
             entry:
               %0: i32 = param reg
               %1: i32 = call @llvm.ctpop.i32(%0)
               black_box %1
            ",
            "
               ...
               ; %1: i32 = call @llvm.ctpop.i32(%0)
               popcnt r.32._, r.32.a
            ",
            false,
        );

        codegen_and_test(
            "
             func_decl llvm.ctpop.i64 (i64) -> i64
             entry:
               %0: i64 = param reg
               %1: i64 = call @llvm.ctpop.i64(%0)
               black_box %1
            ",
            "
               ...
               ; %1: i64 = call @llvm.ctpop.i64(%0)
               popcnt r.64._, r.64.a
            ",
            false,
        );
    }

    #[test]
    fn cg_call_floor() {
        codegen_and_test(
            "
             func_decl llvm.floor.f64 (double) -> double
             entry:
               %0: double = param reg
               %1: double = call @llvm.floor.f64(%0)
               black_box %1
            ",
            "
               ...
               ; %1: double = call @llvm.floor.f64(%0)
               roundsd fp.128._, fp.128._, 0x01
            ",
            false,
        );
    }

    #[test]
    fn cg_call_fshl() {
        codegen_and_test(
            "
             func_decl llvm.fshl.i64 (i64, i64, i64) -> i64
             entry:
               %0: i64 = param reg
               %1: i64 = param reg
               %2: i64 = call @llvm.fshl.i64(%0, %1, 17i64)
               black_box %2
            ",
            "
               ...
               ; %2: i64 = call @llvm.fshl.i64(%0, %1, 17i64)
               shld r.64.x, r.64.y, 0x11
            ",
            false,
        );
    }

    #[test]
    fn cg_call_memcpy() {
        codegen_and_test(
            "
             func_decl llvm.memcpy.p0.p0.i64 (ptr, ptr, i64, i1)
             entry:
               %0: ptr = param reg
               %1: ptr = param reg
               %2: i64 = param reg
               call @llvm.memcpy.p0.p0.i64(%0, %1, %2, 0i1)
            ",
            "
               ...
               ; call @llvm.memcpy.p0.p0.i64(%0, %1, %2, 0i1)
               mov rsi, rcx
               mov rcx, rdx
               mov rdi, rax
               rep movsb
            ",
            false,
        );
    }

    #[test]
    fn cg_call_memset() {
        codegen_and_test(
            "
             func_decl llvm.memset.p0.i64 (ptr, i8, i64, i1)
             entry:
               %0: ptr = param reg
               %1: i8 = param reg
               %2: i64 = param reg
               call @llvm.memset.p0.i64(%0, %1, %2, 0i1)
            ",
            "
               ...
               ; call @llvm.memset.p0.i64(%0, %1, %2, 0i1)
               mov rdi, rax
               mov rax, rcx
               mov rcx, rdx
               and eax, 0xff
               rep stosb
            ",
            false,
        );
    }

    #[test]
    fn cg_call_smax() {
        codegen_and_test(
            "
             func_decl llvm.smax.i64 (i64, i64) -> i64
             entry:
               %0: i64 = param reg
               %1: i64 = param reg
               %2: i64 = call @llvm.smax.i64(%0, %1)
               black_box %2
            ",
            "
               ...
               ; %2: i64 = call @llvm.smax.i64(%0, %1)
               cmp r.64.a, r.64.b
               cmovl r.64.a, r.64.b
            ",
            false,
        );
    }

    #[test]
    fn cg_call_smin() {
        codegen_and_test(
            "
             func_decl llvm.smin.i64 (i64, i64) -> i64
             entry:
               %0: i64 = param reg
               %1: i64 = param reg
               %2: i64 = call @llvm.smin.i64(%0, %1)
               black_box %2
            ",
            "
               ...
               ; %2: i64 = call @llvm.smin.i64(%0, %1)
               ...
               cmp r.64.a, r.64.b
               cmovnle r.64.a, r.64.b
            ",
            false,
        );

        codegen_and_test(
            "
             func_decl llvm.smin.i32 (i32, i32) -> i32
             entry:
               %0: i32 = param reg
               %1: i32 = param reg
               %2: i32 = call @llvm.smin.i32(%0, %1)
               black_box %2
            ",
            "
               ...
               ; %2: i32 = call @llvm.smin.i32(%0, %1)
               ...
               cmp r.64.a, r.64.b
               cmovnle r.64.a, r.64.b
            ",
            false,
        );
    }

    #[test]
    fn cg_icall() {
        codegen_and_test(
            "
              func_type f(i8) -> i16

              entry:
                %0: i8 = param reg
                %1: ptr = param reg
                %2: i16 = icall<f> %1(%0)
                black_box %2
            ",
            "
                ...
                ; %2: i16 = icall %1(%0)
                mov rdi, rax
                mov rax, rcx
                and edi, 0xff
                call rax
            ",
            false,
        );

        codegen_and_test(
            "
              func_type f(i8, ...) -> i16

              entry:
                %0: i8 = param reg
                %1: ptr = param reg
                %2: double = param reg
                %3: i16 = icall<f> %1(%0, %0, %2)
                black_box %3
            ",
            "
                ...
                ; %3: i16 = icall %1(%0, %0, %2)
                mov rsi, rax
                mov rdi, rax
                mov r15, rcx
                and esi, 0xff
                and edi, 0xff
                mov rax, 0x01
                call r15
            ",
            false,
        );

        codegen_and_test(
            "
              func_type f(i8)

              entry:
                %0: i8 = param reg
                %1: ptr = param reg
                icall<f> %1(%0)
            ",
            "
                ...
                ; icall %1(%0)
                mov rdi, rax
                mov rax, rcx
                and edi, 0xff
                call rax
            ",
            false,
        );
    }

    #[test]
    fn cg_eq() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i32 = param reg
                %3: i1 = eq %0, %0
                %4: i1 = eq %1, %1
                %5: i1 = eq %2, %1
                black_box %3
                black_box %4
                black_box %5
            ",
            "
                ...
                ; %3: i1 = eq %0, %0
                and r.32.a, 0xffff
                cmp r.32.a, r.32.a
                setz r.8._
                ; %4: i1 = eq %1, %1
                cmp r.32.b, r.32.b
                setz r.8._
                ; %5: i1 = eq %2, %1
                cmp r.32.c, r.32.b
                setz r.8._
            ",
            false,
        );
    }

    #[test]
    fn cg_sext() {
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i32 = sext %0
                %3: i64 = sext %1
                black_box %2
                black_box %3
                ",
            "
                ...
                ; %2: i32 = sext %0
                movsx r.64.a, r.16.a
                ; %3: i64 = sext %1
                movsxd r.64.b, r.32.b
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
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i32 = zext %0
                %3: i64 = zext %1
                black_box %2
                black_box %3
                ",
            "
                ...
                ; %2: i32 = zext %0
                and r.32.a, 0xffff
                ; %3: i64 = zext %1
                mov r.32.b, r.32.b
                ",
            false,
        );

        // Check that `zext`ing zero extended things doesn't allocate a new register.
        codegen_and_test(
            "
              entry:
                %0: i16 = param reg
                %1: i32 = param reg
                %2: i64 = param reg
                %3: i32 = zext %0
                %4: i64 = zext %0
                %5: i64 = zext %3
                %6: i64 = add %4, %5
                black_box %0
                black_box %1
                black_box %2
                black_box %3
                black_box %4
                black_box %5
                black_box %6
                ",
            "
                ...
                ; %0: i16 = param ...
                ; %1: i32 = param ...
                ; %2: i64 = param ...
                ; %3: i32 = zext %0
                and eax, 0xffff
                ; %4: i64 = zext %0
                ; %5: i64 = zext %3
                ; %6: i64 = add %4, %5
                mov r.64._, rax
                add rax, rax
                ",
            false,
        );
    }

    #[test]
    fn cg_bitcast() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param reg
                %1: i32 = param reg
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
                %0: i1 = param reg
                guard true, %0, []
            ",
            "
                ...
                ; guard true, %0, [] ; ...
                bt r.32._, 0x00
                jnb 0x...
                ...
                ; deopt id and patch point for guard 0
                nop ...
                ...
                push rsi
                mov rsi, 0x00
                jmp ...
                ; call __yk_deopt
                ...
                mov rdi, rbp
                mov r8, 0x...
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
                %0: i1 = param reg
                guard false, %0, []
            ",
            "
                ...
                ; guard false, %0, [] ; ...
                bt r.32._, 0x00
                jb 0x...
                ...
                ; deopt id and patch point for guard 0
                nop ...
                ...
                push rsi
                mov rsi, 0x00
                jmp ...
                ; call __yk_deopt
                ...
                mov rdi, rbp
                mov r8, 0x...
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
                %0: i1 = param reg
                %1: i8 = 10i8
                %2: i8 = 32i8
                %3: i8 = add %1, %2
                guard false, %0, [%0, 10i8, 32i8, 42i8]
            ",
            "
                ...
                ; guard false, %0, [0:%0_0: %0, 0:%0_1: 10i8, 0:%0_2: 32i8, 0:%0_3: 42i8] ; trace_gid 0 safepoint_id 0
                bt r.32._, 0x00
                jb 0x...
                ...
                ; deopt id and patch point for guard 0
                and r.32._, 0x01
                nop ...
                ...
                push rsi
                mov rsi, 0x00
                jmp ...
                ; call __yk_deopt
                ...
                mov rdi, rbp
                mov r8, 0x...
                mov rax, 0x...
                call rax
            ",
            false,
        );
    }

    #[test]
    fn cg_guard_spill() {
        // Check that spilling a live register in a guard spills in the deopt, not the "main",
        // branch.
        codegen_and_test(
            "
              entry:
                %0: i1 = param reg
                %1: i1 = param reg
                %2: i1 = param reg
                %3: i1 = param reg
                %4: i1 = param reg
                %5: i1 = param reg
                %6: i1 = param reg
                %7: i1 = param reg
                %8: i1 = param reg
                %9: i1 = param reg
                %10: i1 = param reg
                %11: i1 = param reg
                %12: i1 = param reg
                %13: i1 = param reg
                guard false, %0, [%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13]
            ",
            "
                ...
                ; guard false, %0, ...
                bt r.32._, 0x00
                jb ...
                ...
                ; deopt id and patch point for guard 0
                and r.32.x, 0x01
                mov [rbp-0x01], r.8.x
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_guard_otherwise_unused_in_deopt() {
        // Check that spilling a live register in a guard spills in the deopt, not the "main",
        // branch.
        codegen_and_test(
            "
              entry:
                %0: i1 = param reg
                %1: i8 = param reg
                %2: i8 = add %1, 7i8
                %3: i8 = add %1, 8i8
                guard false, %0, [%2, %3]
                black_box %3
            ",
            "
                ...
                ; %0: i1 = param register(0, 1, [])
                ; %1: i8 = param register(2, 1, [])
                ; %3: i8 = add %1, 8i8
                ...
                ; guard false, %0, ...
                ...
                ; black_box %3
                ...
                ; deopt id and patch point for guard 0
                ; %2: i8 = add %1, 7i8
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_deopt_reg_exts() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i1 = param reg
                %2: i8 = shl %0, 1i8
                guard true, %1, [%2]
            ",
            "
                ...
                ; guard true, %1, ...
                ...
                ; deopt id and patch point for guard 0
                ; %2: i8 = shl %0, 1i8
                mov ...
                shl r.64.x, 0x01
                and r.32.x, 0xff
                ...
            ",
            false,
        );
    }

    #[test]
    fn cg_icmp_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i1 = eq %0, 0i8
                black_box %1
            ",
            "
                ...
                ; %1: i1 = eq %0, 0i8
                and r.32.x, 0xff
                test r.32.x, r.32.x
                setz r.8.x
                ...
            ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i1 = eq %0, 3i8
                black_box %1
            ",
            "
                ...
                ; %1: i1 = eq %0, 3i8
                and r.32.x, 0xff
                cmp r.32.x, 0x03
                setz r.8.x
                ...
            ",
            false,
        );

        codegen_and_test(
            "
              entry:
                %0: i64 = param reg
                %1: i1 = eq %0, 2147483647i64
                %2: i1 = eq %0, 2147483648i64
                %3: i1 = slt %0, 4294967296i64
                %4: i1 = slt %0, 18446744073709551615i64
                black_box %1
                black_box %2
                black_box %3
                black_box %4
            ",
            "
                ...
                ; %1: i1 = eq %0, 2147483647i64
                cmp r.64.x, 0x7fffffff
                setz r.8._
                ; %2: i1 = eq %0, 2147483648i64
                mov r.64.y, 0x80000000
                cmp r.64.x, r.64.y
                setz r.8._
                ; %3: i1 = slt %0, 4294967296i64
                mov r.64.z, 0x100000000
                cmp r.64.x, r.64.z
                setl r.8._
                ; %4: i1 = slt %0, 18446744073709551615i64
                cmp r.64.x, 0xffffffffffffffff
                setl r.8._
            ",
            false,
        );
    }

    #[test]
    fn cg_icmp_guard() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i1 = eq %0, 3i8
                guard true, %1, []
            ",
            "
                ...
                ; %1: i1 = eq %0, 3i8
                and r.32.x, 0xff
                cmp r.32.x, 0x03
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
                %0: i8 = param reg
                %1: i1 = eq %0, 3i8
                guard true, %1, []
                %3: i8 = sext %1
                black_box %3
            ",
            "
                ...
                ; %1: i1 = eq %0, 3i8
                and r.32.x, 0xff
                cmp r.32.x, 0x03
                setz r.8._
                ; guard true, %1, [] ; ...
                bt r.32.x, 0x00
                jnb 0x...
                ; %3: i8 = sext %1
                and r.64.x, 0x01
                neg r.64.x
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
                ...
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
                %0: i8 = param reg
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
                ...
                ; %2: i8 = add %0, %0
                {{addr}} {{_}}: ...
                ...
                ; header_end [%0]
                ...
                {{_}} {{_}}: jmp 0x{{addr}}
            ",
            true,
        );
    }

    #[test]
    fn cg_srem_i8() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
                %1: i8 = param reg
                %2: i8 = srem %0, %1
                black_box %2
            ",
            "
                ...
                ; %2: i8 = srem %0, %1
                movsx rax, al
                movsx rcx, cl
                cqo
                idiv rcx
            ",
            false,
        );
    }

    #[test]
    fn cg_trunc() {
        codegen_and_test(
            "
              entry:
                %0: i32 = param reg
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
                ...
                and r.32._, 0xFF
                ; %2: i8 = trunc %0
                ...
                and r.32._, 0xFF
                ; %3: i8 = add %2, %1
                add r.32._, r.32._
            ",
            false,
        );
    }

    #[test]
    fn cg_select_i1_consts() {
        codegen_and_test(
            "
              entry:
                %0: i1 = param reg
                %1: i1 = param reg
                %2: i1 = %0 ? %1 : 0i1
                black_box %2
            ",
            "
                ...
                ; %2: i1 = %0 ? %1 : 0i1
                and eax, ecx
            ",
            false,
        );
        codegen_and_test(
            "
              entry:
                %0: i1 = param reg
                %1: i1 = param reg
                %2: i1 = %0 ? %1 : 1i1
                black_box %2
            ",
            "
                ...
                ; %2: i1 = %0 ? %1 : 1i1
                not eax
                or eax, ecx
            ",
            false,
        );
        codegen_and_test(
            "
              entry:
                %0: i1 = param reg
                %1: i1 = param reg
                %2: i1 = %0 ? 0i1 : %1
                black_box %2
            ",
            "
                ...
                ; %2: i1 = %0 ? 0i1 : %1
                not eax
                and eax, ecx
            ",
            false,
        );
        codegen_and_test(
            "
              entry:
                %0: i1 = param reg
                %1: i1 = param reg
                %2: i1 = %0 ? 1i1 : %1
                black_box %2
            ",
            "
                ...
                ; %2: i1 = %0 ? 1i1 : %1
                or eax, ecx
            ",
            false,
        );
    }

    #[test]
    fn cg_select_float() {
        codegen_and_test(
            "
              entry:
                %0: float = param reg
                %1: float = param reg
                %2: i1 = param reg
                %3: float = %2 ? %0 : %1
                %4: float = fadd %0, %1
                black_box %3
                black_box %4
            ",
            "
                ...
                ; %3: float = %2 ? %0 : %1
                {{_}} {{_}}: bt r.32._, 0x00
                {{_}} {{_}}: jb 0x{{true_label}}
                {{_}} {{_}}: movss fp.128.x, fp.128.y
                {{_}} {{_}}: jmp 0x{{done_label}}
                {{true_label}} {{_}}: movss fp.128.x, fp.128.z
                ; %4: float = fadd %0, %1
                {{done_label}} {{_}}: ...
            ",
            true,
        );

        codegen_and_test(
            "
              entry:
                %0: double = param reg
                %1: double = param reg
                %2: i1 = param reg
                %3: double = %2 ? %0 : %1
                %4: double = fadd %0, %1
                black_box %3
                black_box %4
            ",
            "
                ...
                ; %3: double = %2 ? %0 : %1
                {{_}} {{_}}: bt r.32._, 0x00
                {{_}} {{_}}: jb 0x{{true_label}}
                {{_}} {{_}}: movsd fp.128.x, fp.128.y
                {{_}} {{_}}: jmp 0x{{done_label}}
                {{true_label}} {{_}}: movsd fp.128.x, fp.128.z
                ; %4: double = fadd %0, %1
                {{done_label}} {{_}}: ...
            ",
            true,
        );
    }

    #[test]
    fn cg_const() {
        codegen_and_test(
            "
              entry:
                %0: i8 = param reg
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
                %0: i32 = param reg
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
                %0: i32 = param reg
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
    fn cg_uitofp_double() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param reg
                %1: double = ui_to_fp %0
                black_box %1
            ",
            "
                ...
                ; %1: double = ui_to_fp %0
                ...
                movq fp.128.x, r.64.x
                punpckldq fp.128.x, [0x{{addr1}}]
                subpd fp.128.x, [0x{{addr2}}]
                movapd fp.128.y, fp.128.x
                unpckhpd fp.128.y, fp.128.x
                addsd fp.128.y, fp.128.x
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
                %0: float = param reg
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
                %0: float = param reg
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
                %0: double = param reg
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
                %0: float = param reg
                %1: float = param reg
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
                %0: double = param reg
                %1: double = param reg
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
                %0: float = param reg
                %1: float = param reg
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
                %0: double = param reg
                %1: double = param reg
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
                %0: float = param reg
                %1: float = param reg
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
                %0: double = param reg
                %1: double = param reg
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
                %0: float = param reg
                %1: float = param reg
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
                %0: double = param reg
                %1: double = param reg
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
    fn cg_fcmp_float_ordered() {
        codegen_and_test(
            "
              entry:
                %0: float = param reg
                %1: float = param reg
                %2: i1 = f_oeq %0, %1
                %3: i1 = f_ogt %0, %1
                %4: i1 = f_oge %0, %1
                %5: i1 = f_olt %0, %1
                %6: i1 = f_ole %0, %1
                %7: i1 = f_one %0, %1
                black_box %2
                black_box %3
                black_box %4
                black_box %5
                black_box %6
                black_box %7
            ",
            "
            ...
            ; %2: i1 = f_oeq %0, %1
            ucomiss fp.128.x, fp.128.y
            setz r.8.i
            setnp r.8.j
            and r.8.j, r.8.i
            ; %3: i1 = f_ogt %0, %1
            ucomiss fp.128.x, fp.128.y
            setnbe r.8._
            ; %4: i1 = f_oge %0, %1
            ucomiss fp.128.x, fp.128.y
            setnb r.8._
            ; %5: i1 = f_olt %0, %1
            ucomiss fp.128.y, fp.128.x
            setnbe r.8._
            ; %6: i1 = f_ole %0, %1
            ucomiss fp.128.y, fp.128.x
            setnb r.8._
            ; %7: i1 = f_one %0, %1
            ucomiss fp.128.x, fp.128.y
            setnz r.8._
            ",
            false,
        );
    }

    #[test]
    fn cg_fcmp_float_unordered() {
        codegen_and_test(
            "
              entry:
                %0: float = param reg
                %1: float = param reg
                %2: i1 = f_ueq %0, %1
                %3: i1 = f_ugt %0, %1
                %4: i1 = f_uge %0, %1
                %5: i1 = f_ult %0, %1
                %6: i1 = f_ule %0, %1
                %7: i1 = f_une %0, %1
                black_box %2
                black_box %3
                black_box %4
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %2: i1 = f_ueq %0, %1
                ucomiss fp.128.x, fp.128.y
                setz r.8._
                ; %3: i1 = f_ugt %0, %1
                ucomiss fp.128.y, fp.128.x
                setb r.8._
                ; %4: i1 = f_uge %0, %1
                ucomiss fp.128.y, fp.128.x
                setbe r.8._
                ; %5: i1 = f_ult %0, %1
                ucomiss fp.128.x, fp.128.y
                setb r.8._
                ; %6: i1 = f_ule %0, %1
                ucomiss fp.128.x, fp.128.y
                setbe r.8._
                ; %7: i1 = f_une %0, %1
                ucomiss fp.128.x, fp.128.y
                setnz r.8.i
                setp r.8.j
                or r.8.j, r.8.i
                ",
            false,
        );
    }

    #[test]
    fn cg_fcmp_double_ordered() {
        codegen_and_test(
            "
              entry:
                %0: double = param reg
                %1: double = param reg
                %2: i1 = f_oeq %0, %1
                %3: i1 = f_ogt %0, %1
                %4: i1 = f_oge %0, %1
                %5: i1 = f_olt %0, %1
                %6: i1 = f_ole %0, %1
                %7: i1 = f_one %0, %1
                black_box %2
                black_box %3
                black_box %4
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %2: i1 = f_oeq %0, %1
                ucomisd fp.128.x, fp.128.y
                setz r.8.i
                setnp r.8.j
                and r.8.j, r.8.i
                ; %3: i1 = f_ogt %0, %1
                ucomisd fp.128.x, fp.128.y
                setnbe r.8._
                ; %4: i1 = f_oge %0, %1
                ucomisd fp.128.x, fp.128.y
                setnb r.8._
                ; %5: i1 = f_olt %0, %1
                ucomisd fp.128.y, fp.128.x
                setnbe r.8._
                ; %6: i1 = f_ole %0, %1
                ucomisd fp.128.y, fp.128.x
                setnb r.8._
                ; %7: i1 = f_one %0, %1
                ucomisd fp.128.x, fp.128.y
                setnz r.8._
                ",
            false,
        );
    }

    #[test]
    fn cg_fcmp_double_unordered() {
        codegen_and_test(
            "
              entry:
                %0: double = param reg
                %1: double = param reg
                %2: i1 = f_ueq %0, %1
                %3: i1 = f_ugt %0, %1
                %4: i1 = f_uge %0, %1
                %5: i1 = f_ult %0, %1
                %6: i1 = f_ule %0, %1
                %7: i1 = f_une %0, %1
                black_box %2
                black_box %3
                black_box %4
                black_box %5
                black_box %6
                black_box %7
            ",
            "
                ...
                ; %2: i1 = f_ueq %0, %1
                ucomisd fp.128.x, fp.128.y
                setz r.8.x
                ; %3: i1 = f_ugt %0, %1
                ucomisd fp.128.y, fp.128.x
                setb r.8._
                ; %4: i1 = f_uge %0, %1
                ucomisd fp.128.y, fp.128.x
                setbe r.8._
                ; %5: i1 = f_ult %0, %1
                ucomisd fp.128.x, fp.128.y
                setb r.8._
                ; %6: i1 = f_ule %0, %1
                ucomisd fp.128.x, fp.128.y
                setbe r.8._
                ; %7: i1 = f_une %0, %1
                ucomisd fp.128.x, fp.128.y
                setnz r.8.i
                setp r.8.j
                or r.8.j, r.8.i
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
                %0: i8 = param reg
                header_start [%0]
                %2: i8 = 42i8
                header_end [%2]
            ",
            "
                ...
                ; %0: i8 = param ...
                ...
                ; header_start [%0]
                ...
                ; header_end [42i8]
                mov eax, 0x2a
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
                %0: float = param reg
                %1: double = param reg
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
    fn cg_ptrtoint() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i64 = ptr_to_int %0
                %2: i32 = ptr_to_int %0
                %3: i11 = ptr_to_int %0
                %4: i1 = ptr_to_int %0
                black_box %1
                black_box %2
                black_box %3
                black_box %4
            ",
            "
                ...
                ; %0: ptr = param ...
                ; %1: i64 = ptr_to_int %0
                mov r.64.x, r.64.w
                ; %2: i32 = ptr_to_int %0
                mov r.64.y, r.64.x
                ; %3: i11 = ptr_to_int %0
                mov r.64.z, r.64.y
            ",
            false,
        );
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn cg_ptrtoint_to_larger() {
        codegen_and_test(
            "
              entry:
                %0: ptr = param reg
                %1: i128 = ptr_to_int %0
                black_box %1
            ",
            "todo",
            false,
        );
    }

    #[test]
    fn cg_inttoptr() {
        codegen_and_test(
            "
              entry:
                %0: i64 = param reg
                %1: i32 = param reg
                %2: i17 = param reg
                %3: i1 = param reg
                %4: ptr = int_to_ptr %0
                %5: ptr = int_to_ptr %1
                %6: ptr = int_to_ptr %2
                %7: ptr = int_to_ptr %3
                black_box %4
                black_box %5
                black_box %6
                black_box %7
            ",
            "
            ...
            ; %4: ptr = int_to_ptr %0
            ; %5: ptr = int_to_ptr %1
            mov r.32.x, r.32.x
            ; %6: ptr = int_to_ptr %2
            and r.32.y, 0x1ffff
            ; %7: ptr = int_to_ptr %3
            and r.32.z, 0x01
            ",
            false,
        );
    }

    #[test]
    #[should_panic]
    fn cg_inttoptr_from_larger() {
        codegen_and_test(
            "
              entry:
                %0: i128 = param reg
                %1: ptr = int_to_ptr %0
                black_box %1
            ",
            "todo",
            false,
        );
    }

    #[test]
    fn cg_aliasing_params() {
        let mut m = jit_ir::Module::new(TraceKind::HeaderOnly, TraceId::testing(), 0).unwrap();

        // Create two trace paramaters whose locations alias.
        let loc = yksmp::Location::Register(13, 1, smallvec![]);
        m.push_param(loc.clone());
        let pinst1: Inst =
            jit_ir::ParamInst::new(ParamIdx::try_from(0).unwrap(), m.int8_tyidx()).into();
        m.push(pinst1).unwrap();
        m.push_param(loc);
        let pinst2: Inst =
            jit_ir::ParamInst::new(ParamIdx::try_from(1).unwrap(), m.int8_tyidx()).into();
        m.push(pinst2).unwrap();
        let op1 = m.push_and_make_operand(pinst1).unwrap();
        let op2 = m.push_and_make_operand(pinst2).unwrap();

        let add_inst = jit_ir::BinOpInst::new(op1, jit_ir::BinOp::Add, op2);
        m.push(add_inst.into()).unwrap();

        let mt = MT::new().unwrap();
        let hl = HotLocation {
            kind: HotLocationKind::Tracing(mt.next_trace_id()),
            tracecompilation_errors: 0,
            #[cfg(feature = "ykd")]
            debug_str: None,
        };

        Assemble::new(&m)
            .unwrap()
            .codegen(mt, Arc::new(Mutex::new(hl)))
            .unwrap()
            .as_any()
            .downcast::<X64CompiledTrace>()
            .unwrap();
    }
}
