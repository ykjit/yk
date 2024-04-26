//! The X86_64 JIT Code Generator.
//!
//! FIXME: the code generator clobbers registers willy-nilly because at the time of writing we have
//! a register allocator that doesn't actually use any registers. Later we will have to audit the
//! backend and insert register save/restore for clobbered registers.

use super::{
    super::{
        jit_ir::{self, FuncDeclIdx, InstIdx, Operand, Ty},
        CompilationError,
    },
    abs_stack::AbstractStack,
    reg_alloc::{LocalAlloc, RegisterAllocator, StackDirection},
    CodeGen,
};
use crate::compile::CompiledTrace;
use byteorder::{NativeEndian, ReadBytesExt};
use dynasmrt::{
    components::StaticLabel, dynasm, x64::Rq, AssemblyOffset, DynasmApi, DynasmError,
    DynasmLabelApi, ExecutableBuffer, Register,
};
use std::sync::Arc;
#[cfg(any(debug_assertions, test))]
use std::{cell::Cell, collections::HashMap, error::Error, slice};
use ykaddr::addr::symbol_to_ptr;

mod deopt;

use deopt::__yk_deopt;

/// Argument registers as defined by the X86_64 SysV ABI.
static ARG_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// The argument index of the live variables struct argument in the JITted code function.
static JITFUNC_LIVEVARS_ARGIDX: usize = 0;

/// The size of a 64-bit register in bytes.
static REG64_SIZE: usize = 8;
static RBP_DWARF_NUM: u16 = 6;

/// Work registers, i.e. the registers we use temporarily (where possible) for operands to, and
/// results of, intermediate computations.
///
/// We choose callee-save registers so that we don't have to worry about storing/restoring them
/// when we do a function call to external code.
static WR0: Rq = Rq::R12;
static WR1: Rq = Rq::R13;
static WR2: Rq = Rq::R14;

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
    /// Comments used by the trace printer for debugging and testing only.
    ///
    /// Each assembly offset can have zero or more comment lines.
    #[cfg(any(debug_assertions, test))]
    comments: Cell<HashMap<usize, Vec<String>>>,
}

impl<'a> CodeGen<'a> for X64CodeGen<'a> {
    fn new(
        m: &'a jit_ir::Module,
        ra: Box<dyn RegisterAllocator>,
    ) -> Result<Box<X64CodeGen<'a>>, CompilationError> {
        let asm = dynasmrt::x64::Assembler::new()
            .map_err(|e| CompilationError::ResourceExhausted(Box::new(e)))?;
        Ok(Box::new(Self {
            m,
            asm,
            stack: Default::default(),
            ra,
            deoptinfo: Vec::new(),
            #[cfg(any(debug_assertions, test))]
            comments: Cell::new(HashMap::new()),
        }))
    }

    fn codegen(mut self: Box<Self>) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let alloc_off = self.emit_prologue();

        // FIXME: we'd like to be able to assemble code backwards as this would simplify register
        // allocation and side-step the need to patch up the prolog after the fact. dynasmrs
        // doesn't support this, but it's on their roadmap:
        // https://github.com/CensoredUsername/dynasm-rs/issues/48
        for (idx, inst) in self.m.insts().iter().enumerate() {
            self.codegen_inst(jit_ir::InstIdx::new(idx)?, inst)?;
        }

        // Loop the JITted code if the backedge target label is present.
        let label = StaticLabel::global("trace_loop_start");
        match self.asm.labels().resolve_static(&label) {
            Ok(_) => {
                // Found the label, emit a jump to it.
                #[cfg(any(debug_assertions, test))]
                self.comment(self.asm.offset(), "Trace loop backedge".to_owned());
                dynasm!(self.asm; jmp ->trace_loop_start);
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

        #[cfg(not(any(debug_assertions, test)))]
        return Ok(Arc::new(X64CompiledTrace {
            buf,
            deoptinfo: self.deoptinfo,
        }));
        #[cfg(any(debug_assertions, test))]
        {
            let comments = self.comments.take();
            Ok(Arc::new(X64CompiledTrace {
                buf,
                deoptinfo: self.deoptinfo,
                comments,
            }))
        }
    }
}

impl<'a> X64CodeGen<'a> {
    /// Codegen an instruction.
    fn codegen_inst(
        &mut self,
        inst_idx: jit_ir::InstIdx,
        inst: &jit_ir::Inst,
    ) -> Result<(), CompilationError> {
        #[cfg(any(debug_assertions, test))]
        self.comment(
            self.asm.offset(),
            inst.display(inst_idx, self.m).to_string(),
        );

        match inst {
            jit_ir::Inst::LoadTraceInput(i) => self.codegen_loadtraceinput_inst(inst_idx, i),
            jit_ir::Inst::Load(i) => self.codegen_load_inst(inst_idx, i),
            jit_ir::Inst::PtrAdd(i) => self.codegen_ptradd_inst(inst_idx, i),
            jit_ir::Inst::Store(i) => self.codegen_store_inst(i),
            jit_ir::Inst::LookupGlobal(i) => self.codegen_lookupglobal_inst(inst_idx, i),
            jit_ir::Inst::Call(i) => self.codegen_call_inst(inst_idx, i)?,
            jit_ir::Inst::VACall(i) => self.codegen_vacall_inst(inst_idx, i)?,
            jit_ir::Inst::Icmp(i) => self.codegen_icmp_inst(inst_idx, i),
            jit_ir::Inst::Guard(i) => self.codegen_guard_inst(i),
            jit_ir::Inst::Arg(i) => self.codegen_arg(inst_idx, *i),
            jit_ir::Inst::TraceLoopStart => self.codegen_traceloopstart_inst(),
            jit_ir::Inst::SignExtend(i) => self.codegen_signextend_inst(inst_idx, i),
            // Binary operations
            jit_ir::Inst::Add(i) => self.codegen_add_inst(inst_idx, i),
            jit_ir::Inst::Or(i) => self.codegen_or_inst(inst_idx, i),
            x => todo!("{x:?}"),
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
    #[allow(clippy::fn_to_numeric_cast)]
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
            dynasm!(self.asm
                ; mov r11, QWORD __yk_break as i64
                ; call r11
            );
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

    /// Load a local variable out of its stack slot into the specified register.
    fn load_local(&mut self, reg: Rq, local: InstIdx) {
        match self.ra.allocation(local) {
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
        }
    }

    /// Load a constant into the specified register.
    fn load_const(&mut self, reg: Rq, cidx: jit_ir::ConstIdx) {
        let cst = self.m.const_(cidx);
        let mut bytes = cst.bytes().as_slice();
        let size = self.m.type_(cst.ty_idx()).byte_size().unwrap();
        debug_assert_eq!(bytes.len(), size);
        match size {
            8 => {
                let val = bytes.read_i64::<NativeEndian>().unwrap();
                dynasm!(self.asm; mov Rq(reg.code()), QWORD val);
            }
            4 => {
                let val = bytes.read_i32::<NativeEndian>().unwrap();
                dynasm!(self.asm; mov Rq(reg.code()), DWORD val);
            }
            1 => {
                let val = bytes.read_i8().unwrap();
                dynasm!(self.asm; mov Rq(reg.code()), val as i32);
            }
            _ => todo!("{}", size),
        };
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
        }
    }

    /// Store a value held in a register into a new local variable.
    fn reg_into_new_local(&mut self, local: InstIdx, reg: Rq) {
        let size = self.m.inst(local).def_byte_size(self.m);
        let l = self.ra.allocate(local, size, &mut self.stack);
        self.store_local(&l, reg, size);
    }

    fn codegen_add_inst(&mut self, inst_idx: jit_ir::InstIdx, inst: &jit_ir::AddInst) {
        let lhs = inst.lhs();
        let rhs = inst.rhs();

        // Operand types must be the same.
        debug_assert_eq!(lhs.type_(self.m), rhs.type_(self.m));

        self.operand_into_reg(WR0, &lhs); // FIXME: assumes value will fit in a reg.
        self.operand_into_reg(WR1, &rhs); // ^^^ same

        match lhs.byte_size(self.m) {
            8 => dynasm!(self.asm; add Rq(WR0.code()), Rq(WR1.code())),
            4 => dynasm!(self.asm; add Rd(WR0.code()), Rd(WR1.code())),
            2 => dynasm!(self.asm; add Rw(WR0.code()), Rw(WR1.code())),
            1 => dynasm!(self.asm; add Rb(WR0.code()), Rb(WR1.code())),
            _ => todo!(),
        }

        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_or_inst(&mut self, inst_idx: jit_ir::InstIdx, inst: &jit_ir::OrInst) {
        let lhs = inst.lhs();
        let rhs = inst.rhs();

        // Operand types must be the same.
        debug_assert_eq!(lhs.type_(self.m), rhs.type_(self.m));

        self.operand_into_reg(WR0, &lhs); // FIXME: assumes value will fit in a reg.
        self.operand_into_reg(WR1, &rhs); // ^^^ same

        match lhs.byte_size(self.m) {
            8 => dynasm!(self.asm; and Rq(WR0.code()), Rq(WR1.code())),
            4 => dynasm!(self.asm; and Rd(WR0.code()), Rd(WR1.code())),
            2 => dynasm!(self.asm; and Rw(WR0.code()), Rw(WR1.code())),
            1 => dynasm!(self.asm; and Rb(WR0.code()), Rb(WR1.code())),
            _ => todo!(),
        }

        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_loadtraceinput_inst(
        &mut self,
        inst_idx: jit_ir::InstIdx,
        inst: &jit_ir::LoadTraceInputInst,
    ) {
        // Find the argument register containing the pointer to the live variables struct.
        let base_reg = ARG_REGS[JITFUNC_LIVEVARS_ARGIDX].code();

        // Now load the value into a new local variable from [base_reg+off].
        match i32::try_from(inst.off()) {
            Ok(off) => {
                let size = self.m.inst(inst_idx).def_byte_size(self.m);
                debug_assert!(size <= REG64_SIZE);
                match size {
                    8 => dynasm!(self.asm ; mov Rq(WR0.code()), [Rq(base_reg) + off]),
                    4 => dynasm!(self.asm ; mov Rd(WR0.code()), [Rq(base_reg) + off]),
                    2 => dynasm!(self.asm ; movzx Rd(WR0.code()), WORD [Rq(base_reg) + off]),
                    1 => dynasm!(self.asm ; movzx Rq(WR0.code()), BYTE [Rq(base_reg) + off]),
                    _ => todo!("{}", size),
                };
                self.reg_into_new_local(inst_idx, WR0);
            }
            _ => todo!(),
        }
    }

    fn codegen_load_inst(&mut self, inst_idx: jit_ir::InstIdx, inst: &jit_ir::LoadInst) {
        self.operand_into_reg(WR0, &inst.operand()); // FIXME: assumes value will fit in a reg.
        let size = self.m.inst(inst_idx).def_byte_size(self.m);
        debug_assert!(size <= REG64_SIZE);
        match size {
            8 => dynasm!(self.asm ; mov Rq(WR0.code()), [Rq(WR0.code())]),
            4 => dynasm!(self.asm ; mov Rd(WR0.code()), [Rq(WR0.code())]),
            2 => dynasm!(self.asm ; movzx Rd(WR0.code()), WORD [Rq(WR0.code())]),
            1 => dynasm!(self.asm ; movzx Rq(WR0.code()), BYTE [Rq(WR0.code())]),
            _ => todo!("{}", size),
        };
        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_ptradd_inst(&mut self, inst_idx: jit_ir::InstIdx, inst: &jit_ir::PtrAddInst) {
        self.operand_into_reg(WR0, &inst.ptr());
        let off = inst.offset();
        // unwrap cannot fail
        if off <= u32::try_from(i32::MAX).unwrap() {
            // `as` safe due to above guard.
            dynasm!(self.asm ; add Rq(WR0.code()), off as i32);
        } else {
            todo!();
        }
        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_store_inst(&mut self, inst: &jit_ir::StoreInst) {
        self.operand_into_reg(WR0, &inst.ptr());
        let val = inst.val();
        self.operand_into_reg(WR1, &val); // FIXME: assumes the value fits in a reg
        match val.byte_size(self.m) {
            8 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rq(WR1.code())),
            4 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rd(WR1.code())),
            2 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rw(WR1.code())),
            1 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rb(WR1.code())),
            _ => todo!(),
        }
    }

    #[cfg(not(test))]
    fn codegen_lookupglobal_inst(
        &mut self,
        inst_idx: jit_ir::InstIdx,
        inst: &jit_ir::LookupGlobalInst,
    ) {
        let decl = inst.decl(self.m);
        if decl.is_threadlocal() {
            todo!();
        }
        let sym_addr = self.m.globalvar_ptr(inst.global_decl_idx()).addr();
        dynasm!(self.asm ; mov Rq(WR0.code()), QWORD i64::try_from(sym_addr).unwrap());
        self.reg_into_new_local(inst_idx, WR0);
    }

    #[cfg(test)]
    fn codegen_lookupglobal_inst(
        &mut self,
        _inst_idx: jit_ir::InstIdx,
        _inst: &jit_ir::LookupGlobalInst,
    ) {
        panic!("Cannot lookup globals in cfg(test) as ykllvm will not have compiled this binary");
    }

    pub(super) fn emit_call(
        &mut self,
        inst_idx: InstIdx,
        func_decl_idx: FuncDeclIdx,
        args: &[Operand],
    ) -> Result<(), CompilationError> {
        // FIXME: floating point args
        // FIXME: non-SysV ABIs
        let fty = self.m.func_type(func_decl_idx);
        let num_args = args.len();

        if num_args > ARG_REGS.len() {
            todo!(); // needs spill
        }

        if fty.is_vararg() {
            // SysV X86_64 ABI says "rax is used to indicate the number of vector arguments passed
            // to a function requiring a variable number of arguments".
            //
            // We don't yet support vectors, so for now rax=0.
            dynasm!(self.asm; mov rax, 0);
        }

        for (i, reg) in ARG_REGS.into_iter().take(num_args).enumerate() {
            let op = &args[i];
            // We can type check the static args (but not varargs).
            debug_assert!(
                i >= fty.num_args() || op.type_(self.m) == fty.arg_type(self.m, i),
                "argument type mismatch in call"
            );
            self.operand_into_reg(reg, op);
        }

        // unwrap safe on account of linker symbol names not containing internal NULL bytes.
        let va = symbol_to_ptr(self.m.func_decl(func_decl_idx).name())
            .map_err(|e| CompilationError::General(e.to_string()))?;

        // The SysV x86_64 ABI requires the stack to be 16-byte aligned prior to a call.
        self.stack.align(SYSV_CALL_STACK_ALIGN);

        // Actually perform the call.
        dynasm!(self.asm
            ; mov Rq(WR0.code()), QWORD va as i64
            ; call Rq(WR0.code())
        );

        // If the function we called has a return value, then store it into a local variable.
        if fty.ret_type(self.m) != &Ty::Void {
            self.reg_into_new_local(inst_idx, Rq::RAX);
        }

        Ok(())
    }

    /// Codegen a (non-varargs) call.
    pub(super) fn codegen_call_inst(
        &mut self,
        inst_idx: InstIdx,
        inst: &jit_ir::CallInst,
    ) -> Result<(), CompilationError> {
        let func_decl_idx = inst.target();
        let func_type = self.m.func_type(func_decl_idx);
        let args = (0..(func_type.num_args()))
            .map(|i| inst.operand(self.m, i))
            .collect::<Vec<_>>();
        self.emit_call(inst_idx, func_decl_idx, &args)
    }

    /// Codegen a varargs call.
    pub(super) fn codegen_vacall_inst(
        &mut self,
        inst_idx: InstIdx,
        inst: &jit_ir::VACallInst,
    ) -> Result<(), CompilationError> {
        let func_decl_idx = inst.target();
        let args = (0..(inst.num_args()))
            .map(|i| inst.operand(self.m, i))
            .collect::<Vec<_>>();
        self.emit_call(inst_idx, func_decl_idx, &args)
    }

    pub(super) fn codegen_icmp_inst(&mut self, inst_idx: InstIdx, inst: &jit_ir::IcmpInst) {
        let (left, pred, right) = (inst.left(), inst.predicate(), inst.right());

        // FIXME: We should be checking type equality here, but since constants currently don't
        // have a type, checking their size is close enough. This won't be correct for struct
        // types, but this function can't deal with those anyway at the moment.
        debug_assert_eq!(
            left.byte_size(self.m),
            right.byte_size(self.m),
            "icmp of differing types"
        );
        debug_assert!(
            matches!(left.type_(self.m), jit_ir::Ty::Integer(_)),
            "icmp of non-integer types"
        );

        // FIXME: assumes values fit in a registers
        self.operand_into_reg(WR0, &left);
        self.operand_into_reg(WR1, &right);

        // Perform the comparison.
        match left.byte_size(self.m) {
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
            // Note: when float predicates added: `_ => panic!()`
        }
        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_arg(&mut self, inst_idx: InstIdx, idx: u16) {
        // For arguments passed into the trace function we simply inform the register allocator
        // where they are stored and let the allocator take things from there.
        self.reg_into_new_local(inst_idx, ARG_REGS[usize::from(idx)]);
    }

    fn codegen_traceloopstart_inst(&mut self) {
        // FIXME: peel the initial iteration of the loop to allow us to hoist loop invariants.
        dynasm!(self.asm; ->trace_loop_start:);
    }

    fn codegen_signextend_inst(&mut self, inst_idx: InstIdx, i: &jit_ir::SignExtendInst) {
        let from_val = i.val();
        let from_type = from_val.type_(self.m);
        let from_size = from_type.byte_size().unwrap();

        let to_type = self.m.type_(i.dest_ty_idx());
        let to_size = to_type.byte_size().unwrap();

        // You can only sign-extend a smaller integer to a larger integer.
        debug_assert!(matches!(to_type, jit_ir::Ty::Integer(_)));
        debug_assert!(matches!(from_type, jit_ir::Ty::Integer(_)));
        debug_assert!(from_size < to_size);

        // FIXME: assumes the input and output fit in a register.
        self.operand_into_reg(WR0, &from_val);

        match (from_size, to_size) {
            (1, 8) => dynasm!(self.asm; movsx Rq(WR0.code()), Rb(WR0.code())),
            (4, 8) => dynasm!(self.asm; movsx Rq(WR0.code()), Rd(WR0.code())),
            _ => todo!(),
        }

        self.reg_into_new_local(inst_idx, WR0);
    }

    #[allow(clippy::fn_to_numeric_cast)]
    fn codegen_guard_inst(&mut self, inst: &jit_ir::GuardInst) {
        let cond = inst.cond();

        // ICmp instructions evaluate to a one-byte zero/one value.
        debug_assert_eq!(cond.byte_size(self.m), 1);

        // Convert the guard info into deopt info and store it on the heap.
        let mut locs: Vec<LocalAlloc> = Vec::new();
        let gi = inst.guard_info(self.m);
        for lidx in gi.lives() {
            locs.push(*self.ra.allocation(*lidx));
        }

        // FIXME: Move `frames` instead of copying them (requires JIT module to be consumable).
        let deoptinfo = DeoptInfo {
            frames: gi.frames().clone(),
            lives: locs,
        };
        // Unwrap is safe since in this architecture usize and i64 have the same size.
        let deoptid = self.deoptinfo.len().try_into().unwrap();
        self.deoptinfo.push(deoptinfo);

        // The simplest thing we can do to crash the program when the guard fails.
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

    fn const_u64_into_reg(&mut self, reg: Rq, cv: u64) {
        dynasm!(self.asm
            ; mov Rq(reg.code()), QWORD cv as i64 // `as` intentional.
        )
    }

    /// Load an operand into a register.
    fn operand_into_reg(&mut self, reg: Rq, op: &Operand) {
        match op {
            Operand::Local(li) => self.load_local(reg, *li),
            Operand::Const(c) => self.load_const(reg, *c),
        }
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
    comments: HashMap<usize, Vec<String>>,
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
    comments: &'a HashMap<usize, Vec<String>>,
}

#[cfg(any(debug_assertions, test))]
impl<'a> AsmPrinter<'a> {
    fn new(buf: &'a ExecutableBuffer, comments: &'a HashMap<usize, Vec<String>>) -> Self {
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
            jit_ir::{self, IntegerTy, Ty},
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
    pub(crate) fn match_asm(cgo: Arc<X64CompiledTrace>, pattern: &str) {
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
            Err(e) => panic!(
                "\n!!! Emitted code didn't match !!!\n\n{}\nFull asm:\n{}\n",
                e, dis
            ),
        }
    }

    mod with_spillalloc {
        use self::jit_ir::FuncTy;

        use super::*;
        use crate::compile::jitc_yk::codegen::reg_alloc::SpillAllocator;

        fn test_with_spillalloc(m: &jit_ir::Module, patt_lines: &[&str]) {
            match_asm(
                X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                    .unwrap()
                    .codegen()
                    .unwrap()
                    .as_any()
                    .downcast::<X64CompiledTrace>()
                    .unwrap(),
                &patt_lines.join("\n"),
            );
        }

        #[test]
        fn codegen_load_ptr() {
            let mut m = test_module();
            let ptr_ty_idx = m.ptr_ty_idx();
            let load_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, ptr_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadInst::new(load_op, ptr_ty_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; %1: ptr = Load %0",
                "... mov r12, [rbp-0x08]",
                "... mov r12, [r12]",
                "... mov [rbp-0x10], r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_load_i8() {
            let mut m = test_module();
            let i8_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(8))).unwrap();
            let load_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadInst::new(load_op, i8_ty_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; %1: i8 = Load %0",
                "... movzx r12, byte ptr [rbp-0x01]",
                "... movzx r12, byte ptr [r12]",
                "... mov [rbp-0x02], r12b",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_load_i32() {
            let mut m = test_module();
            let i32_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(32))).unwrap();
            let ti_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i32_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadInst::new(ti_op, i32_ty_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; %1: i32 = Load %0",
                "... mov r12d, [rbp-0x04]",
                "... mov r12d, [r12]",
                "... mov [rbp-0x08], r12d",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_ptradd() {
            let mut m = test_module();
            let ptr_ty_idx = m.ptr_ty_idx();
            let ti_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, ptr_ty_idx).into())
                .unwrap();
            m.push(jit_ir::PtrAddInst::new(ti_op, 64).into()).unwrap();
            let patt_lines = [
                "...",
                "; %1: ptr = PtrAdd %0, 64",
                "... mov r12, [rbp-0x08]",
                "... add r12, 0x40",
                "... mov [rbp-0x10], r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_store_ptr() {
            let mut m = test_module();
            let ptr_ty_idx = m.ptr_ty_idx();
            let ti1_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, ptr_ty_idx).into())
                .unwrap();
            let ti2_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(8, ptr_ty_idx).into())
                .unwrap();
            m.push(jit_ir::StoreInst::new(ti1_op, ti2_op).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; Store %0, %1",
                "... mov r12, [rbp-0x10]",
                "... mov r13, [rbp-0x08]",
                "... mov [r12], r13",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_loadtraceinput_i8() {
            let mut m = test_module();
            let u8_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(8))).unwrap();
            m.push(jit_ir::LoadTraceInputInst::new(0, u8_ty_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                &format!("; %0: i8 = LoadTraceInput 0, i8"),
                "... movzx r12, byte ptr [rdi]",
                "... mov [rbp-0x01], r12b",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_loadtraceinput_i16_with_offset() {
            let mut m = test_module();
            let u16_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(16))).unwrap();
            m.push(jit_ir::LoadTraceInputInst::new(32, u16_ty_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                &format!("; %0: i16 = LoadTraceInput 32, i16"),
                "... movzx r12d, word ptr [rdi+0x20]",
                "... mov [rbp-0x02], r12w",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_loadtraceinput_many_offset() {
            let mut m = test_module();
            let i8_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(8))).unwrap();
            let ptr_ty_idx = m.ptr_ty_idx();
            m.push(jit_ir::LoadTraceInputInst::new(0, i8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadTraceInputInst::new(1, i8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadTraceInputInst::new(2, i8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadTraceInputInst::new(3, i8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::LoadTraceInputInst::new(8, ptr_ty_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                &format!("; %0: i8 = LoadTraceInput 0, i8"),
                "... movzx r12, byte ptr [rdi]",
                "... mov [rbp-0x01], r12b",
                &format!("; %1: i8 = LoadTraceInput 1, i8"),
                "... movzx r12, byte ptr [rdi+0x01]",
                "... mov [rbp-0x02], r12b",
                &format!("; %2: i8 = LoadTraceInput 2, i8"),
                "... movzx r12, byte ptr [rdi+0x02]",
                "... mov [rbp-0x03], r12b",
                &format!("; %3: i8 = LoadTraceInput 3, i8"),
                "... movzx r12, byte ptr [rdi+0x03]",
                "... mov [rbp-0x04], r12b",
                &format!("; %4: ptr = LoadTraceInput 8, ptr"),
                "... mov r12, [rdi+0x08]",
                "... mov [rbp-0x10], r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_add_i16() {
            let mut m = test_module();
            let i16_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(16))).unwrap();
            let op1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i16_ty_idx).into())
                .unwrap();
            let op2 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(16, i16_ty_idx).into())
                .unwrap();
            m.push(jit_ir::AddInst::new(op1, op2).into()).unwrap();
            let patt_lines = [
                "...",
                "; %2: i16 = Add %0, %1",
                "... movzx r12, word ptr [rbp-0x02]",
                "... movzx r13, word ptr [rbp-0x04]",
                "... add r12w, r13w",
                "... mov [rbp-0x06], r12w",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_add_i64() {
            let mut m = test_module();
            let i64_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(64))).unwrap();
            let op1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i64_ty_idx).into())
                .unwrap();
            let op2 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(64, i64_ty_idx).into())
                .unwrap();
            m.push(jit_ir::AddInst::new(op1, op2).into()).unwrap();
            let patt_lines = [
                "...",
                "; %2: i64 = Add %0, %1",
                "... mov r12, [rbp-0x08]",
                "... mov r13, [rbp-0x10]",
                "... add r12, r13",
                "... mov [rbp-0x18], r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[cfg(debug_assertions)]
        #[should_panic]
        #[test]
        fn codegen_add_wrong_types() {
            let mut m = test_module();
            let i64_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(64))).unwrap();
            let i32_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(32))).unwrap();
            let op1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i64_ty_idx).into())
                .unwrap();
            let op2 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(64, i32_ty_idx).into())
                .unwrap();
            m.push(jit_ir::AddInst::new(op1, op2).into()).unwrap();

            X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                .unwrap()
                .codegen()
                .unwrap();
        }

        /// A function whose symbol is present in the current address space.
        ///
        /// Used for testing code generation for calls.
        ///
        /// This function is never called, we just need something that dlsym(3) will find an
        /// address for. As such, you can generate calls to this with any old signature for the
        /// purpose of testing codegen.
        #[cfg(unix)]
        const CALL_TESTS_CALLEE: &str = "puts";

        #[test]
        fn codegen_call_simple() {
            let mut m = test_module();
            let void_ty_idx = m.void_ty_idx();
            let func_ty_idx = m
                .ty_idx(&jit_ir::Ty::Func(FuncTy::new(vec![], void_ty_idx, false)))
                .unwrap();

            let func_decl_idx = m
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();
            let call_inst = jit_ir::CallInst::new(&mut m, func_decl_idx, &[]).unwrap();
            m.push(call_inst.into()).unwrap();

            let sym_addr = symbol_to_ptr(CALL_TESTS_CALLEE).unwrap().addr();
            let patt_lines = [
                "...",
                &format!("... mov r12, 0x{:X}", sym_addr),
                "... call r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_call_with_args() {
            let mut m = test_module();
            let void_ty_idx = m.void_ty_idx();
            let i32_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(32))).unwrap();
            let func_ty_idx = m
                .ty_idx(&jit_ir::Ty::Func(FuncTy::new(
                    vec![i32_ty_idx; 3],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = m
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            let arg1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i32_ty_idx).into())
                .unwrap();
            let arg2 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(4, i32_ty_idx).into())
                .unwrap();
            let arg3 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(8, i32_ty_idx).into())
                .unwrap();

            let call_inst =
                jit_ir::CallInst::new(&mut m, func_decl_idx, &[arg1, arg2, arg3]).unwrap();
            m.push(call_inst.into()).unwrap();

            let sym_addr = symbol_to_ptr(CALL_TESTS_CALLEE).unwrap().addr();
            let patt_lines = [
                "...",
                "; Call @puts(%0, %1, %2)",
                "... mov edi, [rbp-0x04]",
                "... mov esi, [rbp-0x08]",
                "... mov edx, [rbp-0x0C]",
                &format!("... mov r12, 0x{:X}", sym_addr),
                "... call r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_call_with_different_args() {
            let mut m = test_module();
            let void_ty_idx = m.void_ty_idx();
            let i8_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(8))).unwrap();
            let i16_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(16))).unwrap();
            let i32_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(32))).unwrap();
            let i64_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(64))).unwrap();
            let ptr_ty_idx = m.ptr_ty_idx();
            let func_ty_idx = m
                .ty_idx(&jit_ir::Ty::Func(FuncTy::new(
                    vec![
                        i8_ty_idx, i16_ty_idx, i32_ty_idx, i64_ty_idx, ptr_ty_idx, i8_ty_idx,
                    ],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = m
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            let arg1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i8_ty_idx).into())
                .unwrap();
            let arg2 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(8, i16_ty_idx).into())
                .unwrap();
            let arg3 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(16, i32_ty_idx).into())
                .unwrap();
            let arg4 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(24, i64_ty_idx).into())
                .unwrap();
            let arg5 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(32, ptr_ty_idx).into())
                .unwrap();
            let arg6 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(40, i8_ty_idx).into())
                .unwrap();

            let call_inst =
                jit_ir::CallInst::new(&mut m, func_decl_idx, &[arg1, arg2, arg3, arg4, arg5, arg6])
                    .unwrap();
            m.push(call_inst.into()).unwrap();

            let sym_addr = symbol_to_ptr(CALL_TESTS_CALLEE).unwrap().addr();
            let patt_lines = [
                "...",
                "; Call @puts(%0, %1, %2, %3, %4, %5)",
                "... movzx rdi, byte ptr [rbp-0x01]",
                "... movzx rsi, word ptr [rbp-0x04]",
                "... mov edx, [rbp-0x08]",
                "... mov rcx, [rbp-0x10]",
                "... mov r8, [rbp-0x18]",
                "... movzx r9, byte ptr [rbp-0x19]",
                &format!("... mov r12, 0x{:X}", sym_addr),
                "... call r12",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[should_panic] // until we implement spill args
        #[test]
        fn codegen_call_spill_args() {
            let mut m = test_module();
            let void_ty_idx = m.void_ty_idx();
            let i32_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(32))).unwrap();
            let func_ty_idx = m
                .ty_idx(&jit_ir::Ty::Func(FuncTy::new(
                    vec![i32_ty_idx; 7],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = m
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            let arg1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i32_ty_idx).into())
                .unwrap();

            let args = (0..7).map(|_| arg1.clone()).collect::<Vec<_>>();
            let call_inst = jit_ir::CallInst::new(&mut m, func_decl_idx, &args).unwrap();
            m.push(call_inst.into()).unwrap();

            X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[test]
        fn codegen_call_ret() {
            let mut m = test_module();
            let i32_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(32))).unwrap();
            let func_ty_idx = m
                .ty_idx(&jit_ir::Ty::Func(FuncTy::new(vec![], i32_ty_idx, false)))
                .unwrap();

            let func_decl_idx = m
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();
            let call_inst = jit_ir::CallInst::new(&mut m, func_decl_idx, &[]).unwrap();
            m.push(call_inst.into()).unwrap();

            let sym_addr = symbol_to_ptr(CALL_TESTS_CALLEE).unwrap().addr();
            let patt_lines = [
                "...",
                &format!("... mov r12, 0x{:X}", sym_addr),
                "... call r12",
                "... mov [rbp-0x04], eax",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[cfg(debug_assertions)]
        #[should_panic(expected = "argument type mismatch in call")]
        #[test]
        fn codegen_call_bad_arg_type() {
            let mut m = test_module();
            let void_ty_idx = m.void_ty_idx();
            let i32_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(32))).unwrap();
            let func_ty_idx = m
                .ty_idx(&jit_ir::Ty::Func(FuncTy::new(
                    vec![i32_ty_idx],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = m
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            // Make a call that passes a i8 argument, instead of an i32 as in the func sig.
            let i8_ty_idx = m.ty_idx(&Ty::Integer(IntegerTy::new(8))).unwrap();
            let arg1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i8_ty_idx).into())
                .unwrap();
            let call_inst = jit_ir::CallInst::new(&mut m, func_decl_idx, &[arg1]).unwrap();
            m.push(call_inst.into()).unwrap();

            X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[test]
        fn codegen_icmp_i64() {
            let mut m = test_module();
            let i64_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(64))).unwrap();
            let op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i64_ty_idx).into())
                .unwrap();
            m.push(jit_ir::IcmpInst::new(op.clone(), jit_ir::Predicate::Equal, op).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; %1: i8 = Icmp %0, Equal, %0",
                "... mov r12, [rbp-0x08]",
                "... mov r13, [rbp-0x08]",
                "... cmp r12, r13",
                "... setz r12b",
                "... mov [rbp-0x09], r12b",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_icmp_i8() {
            let mut m = test_module();
            let i8_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(8))).unwrap();
            let op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::IcmpInst::new(op.clone(), jit_ir::Predicate::Equal, op).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; %1: i8 = Icmp %0, Equal, %0",
                "... movzx r12, byte ptr [rbp-0x01]",
                "... movzx r13, byte ptr [rbp-0x01]",
                "... cmp r12b, r13b",
                "... setz r12b",
                "... mov [rbp-0x02], r12b",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[cfg(debug_assertions)]
        #[test]
        #[should_panic(expected = "icmp of non-integer types")]
        fn codegen_icmp_non_ints() {
            let mut m = test_module();
            let ptr_ty_idx = m.ptr_ty_idx();
            let op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, ptr_ty_idx).into())
                .unwrap();
            m.push(jit_ir::IcmpInst::new(op.clone(), jit_ir::Predicate::Equal, op).into())
                .unwrap();
            X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[cfg(debug_assertions)]
        #[test]
        #[should_panic(expected = "icmp of differing types")]
        fn codegen_icmp_diff_types() {
            let mut m = test_module();
            let i8_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(8))).unwrap();
            let i64_ty_idx = m.ty_idx(&jit_ir::Ty::Integer(IntegerTy::new(64))).unwrap();
            let op1 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, i8_ty_idx).into())
                .unwrap();
            let op2 = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(8, i64_ty_idx).into())
                .unwrap();
            m.push(jit_ir::IcmpInst::new(op1, jit_ir::Predicate::Equal, op2).into())
                .unwrap();
            X64CodeGen::new(&m, Box::new(SpillAllocator::new(STACK_DIRECTION)))
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[test]
        fn codegen_guard_true() {
            let mut m = test_module();
            let gi = jit_ir::GuardInfo::new(vec![0], Vec::new());
            let gi_idx = m.push_guardinfo(gi).unwrap();
            let cond_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, m.int8_ty_idx()).into())
                .unwrap();
            m.push(jit_ir::GuardInst::new(cond_op, true, gi_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; Guard %0, true",
                "{{vaddr1}} {{off1}}: jmp 0x00000000{{cmpoff}}",
                "{{vaddr2}} {{failoff}}: mov rdi, [rbp]",
                "... mov rsi, 0x00",
                "... mov rdx, rbp",
                "... mov rax, ...",
                "... call rax",
                "{{vaddr3}} {{cmpoff}}: cmp r12b, 0x01",
                "{{vaddr4}} {{off4}}: jnz 0x00000000{{failoff}}",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn codegen_guard_false() {
            let mut m = test_module();
            let gi = jit_ir::GuardInfo::new(vec![0], Vec::new());
            let gi_idx = m.push_guardinfo(gi).unwrap();
            let cond_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, m.int8_ty_idx()).into())
                .unwrap();
            m.push(jit_ir::GuardInst::new(cond_op, false, gi_idx).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; Guard %0, false",
                "{{vaddr1}} {{off1}}: jmp 0x00000000{{cmpoff}}",
                "{{vaddr2}} {{failoff}}: mov rdi, [rbp]",
                "... mov rsi, 0x00",
                "... mov rdx, rbp",
                "... mov rax, ...",
                "... call rax",
                "{{vaddr3}} {{cmpoff}}: cmp r12b, 0x00",
                "{{vaddr4}} {{off4}}: jnz 0x00000000{{failoff}}",
                "...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn unterminated_trace() {
            let m = test_module();
            let patt_lines = ["...", "; Unterminated trace", "{{vaddr}} {{off}}: ud2"];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn looped_trace_smallest() {
            let mut m = test_module();
            m.push(jit_ir::Inst::TraceLoopStart).unwrap();
            let patt_lines = [
                "...",
                "; TraceLoopStart",
                "; Trace loop backedge",
                // FIXME: make the offset and disassembler format hex the same so we can match
                // easier (capitalisation of hex differs).
                "{{vaddr}} {{off}}: jmp {{target}}",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }

        #[test]
        fn looped_trace_bigger() {
            let mut m = test_module();
            let int8_ty_idx = m.int8_ty_idx();
            let ti_op = m
                .push_and_make_operand(jit_ir::LoadTraceInputInst::new(0, int8_ty_idx).into())
                .unwrap();
            m.push(jit_ir::Inst::TraceLoopStart).unwrap();
            m.push(jit_ir::AddInst::new(ti_op.clone(), ti_op).into())
                .unwrap();
            let patt_lines = [
                "...",
                "; %0: i8 = LoadTraceInput 0, i8",
                "...",
                "; TraceLoopStart",
                "; %2: i8 = add %0, %0",
                "...",
                "; Trace loop backedge",
                "...: jmp ...",
            ];
            test_with_spillalloc(&m, &patt_lines);
        }
    }
}
