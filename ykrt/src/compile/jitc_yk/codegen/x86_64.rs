//! The X86_64 JIT Code Generator.
//!
//! FIXME: the code generator clobbers registers willy-nilly because at the time of writing we have
//! a register allocator that doesn't actually use any registers. Later we will have to audit the
//! backend and insert register save/restore for clobbered registers.

use super::{
    super::{
        jit_ir::{self, InstrIdx, JitIRDisplay, Operand, Type},
        CompilationError,
    },
    abs_stack::AbstractStack,
    reg_alloc::{LocalAlloc, RegisterAllocator, StackDirection},
    CodeGen,
};
use crate::compile::CompiledTrace;
use dynasmrt::{
    dynasm, x64::Rq, AssemblyOffset, DynasmApi, DynasmLabelApi, ExecutableBuffer, Register,
};
#[cfg(any(debug_assertions, test))]
use std::{cell::Cell, collections::HashMap, slice};
use std::{error::Error, ffi::CString, sync::Arc};
use ykaddr::addr::symbol_vaddr;

/// Argument registers as defined by the X86_64 SysV ABI.
static ARG_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// The argument index of the live variables struct argument in the JITted code function.
static JITFUNC_LIVEVARS_ARGIDX: usize = 0;

/// The size of a 64-bit register in bytes.
static REG64_SIZE: usize = 8;

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

/// The X86_64 code generator.
pub(crate) struct X64CodeGen<'a> {
    jit_mod: &'a jit_ir::Module,
    asm: dynasmrt::x64::Assembler,
    /// Abstract stack pointer, as a relative offset from `RBP`. The higher this number, the larger
    /// the JITted code's stack. That means that even on a host where the stack grows down, this
    /// value grows up.
    stack: AbstractStack,
    /// Register allocator.
    ra: &'a mut dyn RegisterAllocator,
    /// Comments used by the trace printer for debugging and testing only.
    ///
    /// Each assembly offset can have zero or more comment lines.
    #[cfg(any(debug_assertions, test))]
    comments: Cell<HashMap<usize, Vec<String>>>,
}

impl<'a> CodeGen<'a> for X64CodeGen<'a> {
    fn new(
        jit_mod: &'a jit_ir::Module,
        ra: &'a mut dyn RegisterAllocator,
    ) -> Result<X64CodeGen<'a>, CompilationError> {
        let asm = dynasmrt::x64::Assembler::new().map_err(|e| CompilationError(e.to_string()))?;
        Ok(Self {
            jit_mod,
            asm,
            stack: Default::default(),
            ra,
            #[cfg(any(debug_assertions, test))]
            comments: Cell::new(HashMap::new()),
        })
    }

    fn codegen(mut self) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let alloc_off = self.emit_prologue();

        // FIXME: we'd like to be able to assemble code backwards as this would simplify register
        // allocation and side-step the need to patch up the prolog after the fact. dynasmrs
        // doesn't support this, but it's on their roadmap:
        // https://github.com/CensoredUsername/dynasm-rs/issues/48
        for (idx, inst) in self.jit_mod.instrs().iter().enumerate() {
            self.codegen_inst(jit_ir::InstrIdx::new(idx)?, inst)?;
        }

        // Now we know the size of the stack frame (i.e. self.asp), patch the allocation with the
        // correct amount.
        self.patch_frame_allocation(alloc_off);

        self.asm
            .commit()
            .map_err(|e| CompilationError(e.to_string()))?;

        let buf = self
            .asm
            .finalize()
            .map_err(|e| CompilationError(format!("failed to finalize assembler: {e:?}").into()))?;

        #[cfg(not(any(debug_assertions, test)))]
        return Ok(Arc::new(X64CompiledTrace { buf }));
        #[cfg(any(debug_assertions, test))]
        {
            let comments = self.comments.take();
            return Ok(Arc::new(X64CompiledTrace { buf, comments }));
        }
    }
}

impl<'a> X64CodeGen<'a> {
    /// Codegen an instruction.
    fn codegen_inst(
        &mut self,
        instr_idx: jit_ir::InstrIdx,
        inst: &jit_ir::Instruction,
    ) -> Result<(), CompilationError> {
        #[cfg(any(debug_assertions, test))]
        self.comment(self.asm.offset(), inst.to_string(self.jit_mod).unwrap());

        match inst {
            jit_ir::Instruction::Add(i) => self.codegen_add_instr(instr_idx, &i),
            jit_ir::Instruction::LoadTraceInput(i) => {
                self.codegen_loadtraceinput_instr(instr_idx, &i)
            }
            jit_ir::Instruction::Load(i) => self.codegen_load_instr(instr_idx, &i),
            jit_ir::Instruction::PtrAdd(i) => self.codegen_ptradd_instr(instr_idx, &i),
            jit_ir::Instruction::Store(i) => self.codegen_store_instr(&i),
            jit_ir::Instruction::LookupGlobal(i) => self.codegen_lookupglobal_instr(instr_idx, &i),
            jit_ir::Instruction::Call(i) => self.codegen_call_instr(instr_idx, &i)?,
            jit_ir::Instruction::Icmp(i) => self.codegen_icmp_instr(instr_idx, &i),
            jit_ir::Instruction::Guard(i) => self.codegen_guard_instr(&i),
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
        );

        // Emit a dummy frame allocation instruction that initially allocates 0 bytes, but will be
        // patched later when we know how big the frame needs to be.
        let alloc_off = self.asm.offset();
        dynasm!(self.asm
            ; sub rsp, DWORD 0
        );

        // FIXME: load/allocate trace inputs here.

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
    fn load_local(&mut self, reg: Rq, local: InstrIdx) {
        match self.ra.allocation(local) {
            LocalAlloc::Stack { frame_off } => {
                match i32::try_from(*frame_off) {
                    Ok(foff) => {
                        let size = local.instr(self.jit_mod).def_byte_size(self.jit_mod);
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
        match self.jit_mod.const_(cidx) {
            jit_ir::Constant::U32(v) => {
                dynasm!(self.asm; mov Rq(reg.code()), DWORD *v as i32)
            }
            jit_ir::Constant::Usize(_) => {
                todo!()
            }
        };
    }

    fn store_local(&mut self, l: &LocalAlloc, reg: Rq, size: usize) {
        match l {
            LocalAlloc::Stack { frame_off } => match i32::try_from(*frame_off) {
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
    fn reg_into_new_local(&mut self, local: InstrIdx, reg: Rq) {
        let size = local.instr(self.jit_mod).def_byte_size(self.jit_mod);
        let l = self.ra.allocate(local, size, &mut self.stack);
        self.store_local(&l, reg, size);
    }

    fn codegen_add_instr(&mut self, inst_idx: jit_ir::InstrIdx, inst: &jit_ir::AddInstruction) {
        let op1 = inst.op1();
        let op2 = inst.op2();

        // FIXME: We should be checking type equality here, but since constants currently don't
        // have a type, checking their size is close enough. This won't be correct for struct
        // types, but this function can't deal with those anyway at the moment.
        debug_assert_eq!(
            op1.byte_size(self.jit_mod),
            op2.byte_size(self.jit_mod),
            "attempt to add different byte-sized types"
        );

        self.operand_into_reg(WR0, &inst.op1()); // FIXME: assumes value will fit in a reg.
        self.operand_into_reg(WR1, &inst.op2()); // ^^^ same

        match op1.byte_size(self.jit_mod) {
            8 => dynasm!(self.asm; add Rq(WR0.code()), Rq(WR1.code())),
            4 => dynasm!(self.asm; add Rd(WR0.code()), Rd(WR1.code())),
            2 => dynasm!(self.asm; add Rw(WR0.code()), Rw(WR1.code())),
            1 => dynasm!(self.asm; add Rb(WR0.code()), Rb(WR1.code())),
            _ => todo!(),
        }

        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_loadtraceinput_instr(
        &mut self,
        inst_idx: jit_ir::InstrIdx,
        inst: &jit_ir::LoadTraceInputInstruction,
    ) {
        // Find the argument register containing the pointer to the live variables struct.
        let base_reg = ARG_REGS[JITFUNC_LIVEVARS_ARGIDX].code();

        // Now load the value into a new local variable from [base_reg+off].
        match i32::try_from(inst.off()) {
            Ok(off) => {
                let size = inst_idx.instr(self.jit_mod).def_byte_size(self.jit_mod);
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

    fn codegen_load_instr(&mut self, inst_idx: jit_ir::InstrIdx, inst: &jit_ir::LoadInstruction) {
        self.operand_into_reg(WR0, &inst.operand()); // FIXME: assumes value will fit in a reg.
        let size = inst_idx.instr(self.jit_mod).def_byte_size(self.jit_mod);
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

    fn codegen_ptradd_instr(
        &mut self,
        inst_idx: jit_ir::InstrIdx,
        inst: &jit_ir::PtrAddInstruction,
    ) {
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

    fn codegen_store_instr(&mut self, inst: &jit_ir::StoreInstruction) {
        self.operand_into_reg(WR0, &inst.ptr());
        let val = inst.val();
        self.operand_into_reg(WR1, &val); // FIXME: assumes the value fits in a reg
        match val.byte_size(self.jit_mod) {
            8 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rq(WR1.code())),
            4 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rd(WR1.code())),
            2 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rw(WR1.code())),
            1 => dynasm!(self.asm ; mov [Rq(WR0.code())], Rb(WR1.code())),
            _ => todo!(),
        }
    }

    fn codegen_lookupglobal_instr(
        &mut self,
        inst_idx: jit_ir::InstrIdx,
        inst: &jit_ir::LookupGlobalInstruction,
    ) {
        let decl = inst.decl(self.jit_mod);
        if decl.is_threadlocal() {
            todo!();
        }
        // Unwrap is safe as the JIT can't contain globals that don't exist in AOT.
        let sym_addr = symbol_vaddr(&CString::new(decl.name()).unwrap()).unwrap();
        dynasm!(self.asm ; mov Rq(WR0.code()), QWORD i64::try_from(sym_addr).unwrap());
        self.reg_into_new_local(inst_idx, WR0);
    }

    pub(super) fn codegen_call_instr(
        &mut self,
        inst_idx: InstrIdx,
        inst: &jit_ir::CallInstruction,
    ) -> Result<(), CompilationError> {
        // FIXME: floating point args
        // FIXME: non-SysV ABIs
        let fdecl = self.jit_mod.func_decl(inst.target());
        let fty = fdecl.func_type(self.jit_mod);
        let num_args = fty.num_args();

        if num_args > ARG_REGS.len() {
            todo!(); // needs spill
        }

        if fty.is_vararg() {
            // When implementing, note the SysV X86_64 ABI says "rax is used to indicate the number
            // of vector arguments passed to a function requiring a variable number of arguments".
            todo!();
        }

        for i in 0..num_args {
            let reg = ARG_REGS[i];
            let op = inst.operand(self.jit_mod, i);
            debug_assert!(
                op.type_(self.jit_mod) == fty.arg_type(self.jit_mod, i),
                "argument type mismatch in call"
            );
            self.operand_into_reg(reg, &op);
        }

        // unwrap safe on account of linker symbol names not containing internal NULL bytes.
        let va = symbol_vaddr(&CString::new(fdecl.name()).unwrap()).ok_or_else(|| {
            CompilationError(format!("couldn't find AOT symbol: {}", fdecl.name()))
        })?;

        // The SysV x86_64 ABI requires the stack to be 16-byte aligned prior to a call.
        self.stack.align(SYSV_CALL_STACK_ALIGN);

        // Actually perform the call.
        dynasm!(self.asm
            ; mov Rq(WR0.code()), QWORD va as i64
            ; call Rq(WR0.code())
        );

        // If the function we called has a return value, then store it into a local variable.
        if fty.ret_type(self.jit_mod) != &Type::Void {
            self.reg_into_new_local(inst_idx, Rq::RAX);
        }

        Ok(())
    }

    pub(super) fn codegen_icmp_instr(
        &mut self,
        inst_idx: InstrIdx,
        inst: &jit_ir::IcmpInstruction,
    ) {
        let (left, pred, right) = (inst.left(), inst.predicate(), inst.right());

        // FIXME: We should be checking type equality here, but since constants currently don't
        // have a type, checking their size is close enough. This won't be correct for struct
        // types, but this function can't deal with those anyway at the moment.
        debug_assert_eq!(
            left.byte_size(self.jit_mod),
            right.byte_size(self.jit_mod),
            "icmp of differing types"
        );
        debug_assert!(
            matches!(left.type_(self.jit_mod), jit_ir::Type::Integer(_)),
            "icmp of non-integer types"
        );

        // FIXME: assumes values fit in a registers
        self.operand_into_reg(WR0, &left);
        self.operand_into_reg(WR1, &right);

        // Perform the comparison.
        match left.byte_size(self.jit_mod) {
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

    fn codegen_guard_instr(&mut self, inst: &jit_ir::GuardInstruction) {
        let cond = inst.cond();

        // ICmp instructions evaluate to a one-byte zero/one value.
        debug_assert_eq!(cond.byte_size(self.jit_mod), 1);

        // The simplest thing we can do to crash the program when the guard fails.
        // FIXME: deoptimise!
        dynasm!(self.asm
            ; jmp >check_cond
            ; guard_fail:
            ; ud2 // undefined instruction, crashes the program.
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

#[derive(Debug)]
pub(super) struct X64CompiledTrace {
    /// The executable code itself.
    buf: ExecutableBuffer,
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
            if let Some(lines) = self.comments.get(
                // FIXME: This could fail if we test on an arch where usize is less than 64-bit.
                &usize::try_from(off).unwrap(),
            ) {
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
            jit_ir::{self, IntegerType, Type},
        },
        CompiledTrace,
    };
    use fm::FMatcher;
    use std::ffi::CString;
    use std::sync::Arc;
    use ykaddr::addr::symbol_vaddr;

    fn test_module() -> jit_ir::Module {
        jit_ir::Module::new("test".into())
    }

    /// Test helper to use `fm` to match a disassembled trace.
    pub(crate) fn match_asm(cgo: Arc<X64CompiledTrace>, pattern: &str) {
        let dis = cgo.disassemble().unwrap();
        match FMatcher::new(pattern).unwrap().matches(&dis) {
            Ok(()) => (),
            Err(e) => panic!(
                "\n!!! Emitted code didn't match !!!\n\n{}\nFull asm:\n{}\n",
                e, dis
            ),
        }
    }

    mod with_spillalloc {
        use self::jit_ir::FuncType;

        use super::*;
        use crate::compile::jitc_yk::codegen::reg_alloc::SpillAllocator;

        fn test_with_spillalloc(jit_mod: &jit_ir::Module, patt_lines: &[&str]) {
            let mut ra = SpillAllocator::new(STACK_DIRECTION);
            match_asm(
                X64CodeGen::new(&jit_mod, &mut ra)
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
            let mut jit_mod = test_module();
            let ptr_ty_idx = jit_mod.ptr_type_idx();
            let load_op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, ptr_ty_idx).into())
                .unwrap();
            jit_mod.push(jit_ir::LoadInstruction::new(load_op, ptr_ty_idx).into());
            let patt_lines = [
                "...",
                "; %1: ptr = Load %0",
                "... mov r12, [rbp-0x08]",
                "... mov r12, [r12]",
                "... mov [rbp-0x10], r12",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_load_i8() {
            let mut jit_mod = test_module();
            let i8_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(8)))
                .unwrap();
            let load_op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i8_ty_idx).into())
                .unwrap();
            jit_mod.push(jit_ir::LoadInstruction::new(load_op, i8_ty_idx).into());
            let patt_lines = [
                "...",
                "; %1: i8 = Load %0",
                "... movzx r12, byte ptr [rbp-0x01]",
                "... movzx r12, byte ptr [r12]",
                "... mov [rbp-0x02], r12b",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_load_i32() {
            let mut jit_mod = test_module();
            let i32_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(32)))
                .unwrap();
            let ti_op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i32_ty_idx).into())
                .unwrap();
            jit_mod.push(jit_ir::LoadInstruction::new(ti_op, i32_ty_idx).into());
            let patt_lines = [
                "...",
                "; %1: i32 = Load %0",
                "... mov r12d, [rbp-0x04]",
                "... mov r12d, [r12]",
                "... mov [rbp-0x08], r12d",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_ptradd() {
            let mut jit_mod = test_module();
            let ptr_ty_idx = jit_mod.ptr_type_idx();
            let ti_op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, ptr_ty_idx).into())
                .unwrap();
            jit_mod.push(jit_ir::PtrAddInstruction::new(ti_op, 64).into());
            let patt_lines = [
                "...",
                "; %1: ptr = PtrAdd %0, 64",
                "... mov r12, [rbp-0x08]",
                "... add r12, 0x40",
                "... mov [rbp-0x10], r12",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_store_ptr() {
            let mut jit_mod = test_module();
            let ptr_ty_idx = jit_mod.ptr_type_idx();
            let ti1_op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, ptr_ty_idx).into())
                .unwrap();
            let ti2_op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(8, ptr_ty_idx).into())
                .unwrap();
            jit_mod.push(jit_ir::StoreInstruction::new(ti1_op, ti2_op).into());
            let patt_lines = [
                "...",
                "; Store %0, %1",
                "... mov r12, [rbp-0x10]",
                "... mov r13, [rbp-0x08]",
                "... mov [r12], r13",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_loadtraceinput_i8() {
            let mut jit_mod = test_module();
            let u8_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(8)))
                .unwrap();
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(0, u8_ty_idx).into());
            let patt_lines = [
                "...",
                &format!("; %0: i8 = LoadTraceInput 0, i8"),
                "... movzx r12, byte ptr [rdi]",
                "... mov [rbp-0x01], r12b",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_loadtraceinput_i16_with_offset() {
            let mut jit_mod = test_module();
            let u16_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(16)))
                .unwrap();
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(32, u16_ty_idx).into());
            let patt_lines = [
                "...",
                &format!("; %0: i16 = LoadTraceInput 32, i16"),
                "... movzx r12d, word ptr [rdi+0x20]",
                "... mov [rbp-0x02], r12w",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_loadtraceinput_many_offset() {
            let mut jit_mod = test_module();
            let i8_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(8)))
                .unwrap();
            let ptr_ty_idx = jit_mod.ptr_type_idx();
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(0, i8_ty_idx).into());
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(1, i8_ty_idx).into());
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(2, i8_ty_idx).into());
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(3, i8_ty_idx).into());
            jit_mod.push(jit_ir::LoadTraceInputInstruction::new(8, ptr_ty_idx).into());
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
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_add_i16() {
            let mut jit_mod = test_module();
            let i16_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(16)))
                .unwrap();
            let op1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i16_ty_idx).into())
                .unwrap();
            let op2 = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(16, i16_ty_idx).into(),
                )
                .unwrap();
            jit_mod.push(jit_ir::AddInstruction::new(op1, op2).into());
            let patt_lines = [
                "...",
                "; %2: i16 = Add %0, %1",
                "... movzx r12, word ptr [rbp-0x02]",
                "... movzx r13, word ptr [rbp-0x04]",
                "... add r12w, r13w",
                "... mov [rbp-0x06], r12w",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_add_i64() {
            let mut jit_mod = test_module();
            let i64_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(64)))
                .unwrap();
            let op1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i64_ty_idx).into())
                .unwrap();
            let op2 = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(64, i64_ty_idx).into(),
                )
                .unwrap();
            jit_mod.push(jit_ir::AddInstruction::new(op1, op2).into());
            let patt_lines = [
                "...",
                "; %2: i64 = Add %0, %1",
                "... mov r12, [rbp-0x08]",
                "... mov r13, [rbp-0x10]",
                "... add r12, r13",
                "... mov [rbp-0x18], r12",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[cfg(debug_assertions)]
        #[should_panic]
        #[test]
        fn codegen_add_wrong_types() {
            let mut jit_mod = test_module();
            let i64_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(64)))
                .unwrap();
            let i32_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(32)))
                .unwrap();
            let op1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i64_ty_idx).into())
                .unwrap();
            let op2 = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(64, i32_ty_idx).into(),
                )
                .unwrap();
            jit_mod.push(jit_ir::AddInstruction::new(op1, op2).into());

            let mut ra = SpillAllocator::new(STACK_DIRECTION);
            X64CodeGen::new(&jit_mod, &mut ra)
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
            let mut jit_mod = test_module();
            let void_ty_idx = jit_mod.void_type_idx();
            let func_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Func(FuncType::new(
                    vec![],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = jit_mod
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();
            let call_inst = jit_ir::CallInstruction::new(&mut jit_mod, func_decl_idx, &[]).unwrap();
            jit_mod.push(call_inst.into());

            let sym_addr = symbol_vaddr(&CString::new(CALL_TESTS_CALLEE).unwrap()).unwrap();
            let patt_lines = [
                "...",
                &format!("... mov r12, 0x{:X}", sym_addr),
                "... call r12",
                "...",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_call_with_args() {
            let mut jit_mod = test_module();
            let void_ty_idx = jit_mod.void_type_idx();
            let i32_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(32)))
                .unwrap();
            let func_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Func(FuncType::new(
                    vec![i32_ty_idx; 3],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = jit_mod
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            let arg1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i32_ty_idx).into())
                .unwrap();
            let arg2 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(4, i32_ty_idx).into())
                .unwrap();
            let arg3 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(8, i32_ty_idx).into())
                .unwrap();

            let call_inst =
                jit_ir::CallInstruction::new(&mut jit_mod, func_decl_idx, &[arg1, arg2, arg3])
                    .unwrap();
            jit_mod.push(call_inst.into());

            let sym_addr = symbol_vaddr(&CString::new(CALL_TESTS_CALLEE).unwrap()).unwrap();
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
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_call_with_different_args() {
            let mut jit_mod = test_module();
            let void_ty_idx = jit_mod.void_type_idx();
            let i8_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(8)))
                .unwrap();
            let i16_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(16)))
                .unwrap();
            let i32_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(32)))
                .unwrap();
            let i64_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(64)))
                .unwrap();
            let ptr_ty_idx = jit_mod.ptr_type_idx();
            let func_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Func(FuncType::new(
                    vec![
                        i8_ty_idx, i16_ty_idx, i32_ty_idx, i64_ty_idx, ptr_ty_idx, i8_ty_idx,
                    ],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = jit_mod
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            let arg1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i8_ty_idx).into())
                .unwrap();
            let arg2 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(8, i16_ty_idx).into())
                .unwrap();
            let arg3 = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(16, i32_ty_idx).into(),
                )
                .unwrap();
            let arg4 = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(24, i64_ty_idx).into(),
                )
                .unwrap();
            let arg5 = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(32, ptr_ty_idx).into(),
                )
                .unwrap();
            let arg6 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(40, i8_ty_idx).into())
                .unwrap();

            let call_inst = jit_ir::CallInstruction::new(
                &mut jit_mod,
                func_decl_idx,
                &[arg1, arg2, arg3, arg4, arg5, arg6],
            )
            .unwrap();
            jit_mod.push(call_inst.into());

            let sym_addr = symbol_vaddr(&CString::new(CALL_TESTS_CALLEE).unwrap()).unwrap();
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
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[should_panic] // until we implement spill args
        #[test]
        fn codegen_call_spill_args() {
            let mut jit_mod = test_module();
            let void_ty_idx = jit_mod.void_type_idx();
            let i32_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(32)))
                .unwrap();
            let func_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Func(FuncType::new(
                    vec![i32_ty_idx; 7],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = jit_mod
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            let arg1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i32_ty_idx).into())
                .unwrap();

            let args = (0..7).map(|_| arg1.clone()).collect::<Vec<_>>();
            let call_inst =
                jit_ir::CallInstruction::new(&mut jit_mod, func_decl_idx, &args).unwrap();
            jit_mod.push(call_inst.into());

            let mut ra = SpillAllocator::new(STACK_DIRECTION);
            X64CodeGen::new(&jit_mod, &mut ra)
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[test]
        fn codegen_call_ret() {
            let mut jit_mod = test_module();
            let i32_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(32)))
                .unwrap();
            let func_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Func(FuncType::new(
                    vec![],
                    i32_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = jit_mod
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();
            let call_inst = jit_ir::CallInstruction::new(&mut jit_mod, func_decl_idx, &[]).unwrap();
            jit_mod.push(call_inst.into());

            let sym_addr = symbol_vaddr(&CString::new(CALL_TESTS_CALLEE).unwrap()).unwrap();
            let patt_lines = [
                "...",
                &format!("... mov r12, 0x{:X}", sym_addr),
                "... call r12",
                "... mov [rbp-0x04], eax",
                "...",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[cfg(debug_assertions)]
        #[should_panic(expected = "argument type mismatch in call")]
        #[test]
        fn codegen_call_bad_arg_type() {
            let mut jit_mod = test_module();
            let void_ty_idx = jit_mod.void_type_idx();
            let i32_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(32)))
                .unwrap();
            let func_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Func(FuncType::new(
                    vec![i32_ty_idx],
                    void_ty_idx,
                    false,
                )))
                .unwrap();

            let func_decl_idx = jit_mod
                .func_decl_idx(&jit_ir::FuncDecl::new(
                    CALL_TESTS_CALLEE.into(),
                    func_ty_idx,
                ))
                .unwrap();

            // Make a call that passes a i8 argument, instead of an i32 as in the func sig.
            let i8_ty_idx = jit_mod
                .type_idx(&Type::Integer(IntegerType::new(8)))
                .unwrap();
            let arg1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i8_ty_idx).into())
                .unwrap();
            let call_inst =
                jit_ir::CallInstruction::new(&mut jit_mod, func_decl_idx, &[arg1]).unwrap();
            jit_mod.push(call_inst.into());

            let mut ra = SpillAllocator::new(STACK_DIRECTION);
            X64CodeGen::new(&jit_mod, &mut ra)
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[test]
        fn codegen_icmp_i64() {
            let mut jit_mod = test_module();
            let i64_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(64)))
                .unwrap();
            let op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i64_ty_idx).into())
                .unwrap();
            jit_mod.push(
                jit_ir::IcmpInstruction::new(op.clone(), jit_ir::Predicate::Equal, op).into(),
            );
            let patt_lines = [
                "...",
                "; %1: i8 = Icmp %0, %0",
                "... mov r12, [rbp-0x08]",
                "... mov r13, [rbp-0x08]",
                "... cmp r12, r13",
                "... setz r12b",
                "... mov [rbp-0x09], r12b",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_icmp_i8() {
            let mut jit_mod = test_module();
            let i8_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(8)))
                .unwrap();
            let op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i8_ty_idx).into())
                .unwrap();
            jit_mod.push(
                jit_ir::IcmpInstruction::new(op.clone(), jit_ir::Predicate::Equal, op).into(),
            );
            let patt_lines = [
                "...",
                "; %1: i8 = Icmp %0, %0",
                "... movzx r12, byte ptr [rbp-0x01]",
                "... movzx r13, byte ptr [rbp-0x01]",
                "... cmp r12b, r13b",
                "... setz r12b",
                "... mov [rbp-0x02], r12b",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[cfg(debug_assertions)]
        #[test]
        #[should_panic(expected = "icmp of non-integer types")]
        fn codegen_icmp_non_ints() {
            let mut jit_mod = test_module();
            let ptr_ty_idx = jit_mod.ptr_type_idx();
            let op = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, ptr_ty_idx).into())
                .unwrap();
            jit_mod.push(
                jit_ir::IcmpInstruction::new(op.clone(), jit_ir::Predicate::Equal, op).into(),
            );
            let mut ra = SpillAllocator::new(STACK_DIRECTION);
            X64CodeGen::new(&jit_mod, &mut ra)
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[cfg(debug_assertions)]
        #[test]
        #[should_panic(expected = "icmp of differing types")]
        fn codegen_icmp_diff_types() {
            let mut jit_mod = test_module();
            let i8_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(8)))
                .unwrap();
            let i64_ty_idx = jit_mod
                .type_idx(&jit_ir::Type::Integer(IntegerType::new(64)))
                .unwrap();
            let op1 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(0, i8_ty_idx).into())
                .unwrap();
            let op2 = jit_mod
                .push_and_make_operand(jit_ir::LoadTraceInputInstruction::new(8, i64_ty_idx).into())
                .unwrap();
            jit_mod.push(jit_ir::IcmpInstruction::new(op1, jit_ir::Predicate::Equal, op2).into());
            let mut ra = SpillAllocator::new(STACK_DIRECTION);
            X64CodeGen::new(&jit_mod, &mut ra)
                .unwrap()
                .codegen()
                .unwrap();
        }

        #[test]
        fn codegen_guard_true() {
            let mut jit_mod = test_module();
            let cond_op = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(0, jit_mod.int8_type_idx()).into(),
                )
                .unwrap();
            jit_mod.push(jit_ir::GuardInstruction::new(cond_op, true).into());
            let patt_lines = [
                "...",
                "; Guard %0, true",
                "... 0000001b: jmp 0x0000000000000022",
                "... 00000020: ud2",
                "... 00000022: cmp r12b, 0x01",
                "... 00000026: jnz 0x0000000000000020",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }

        #[test]
        fn codegen_guard_false() {
            let mut jit_mod = test_module();
            let cond_op = jit_mod
                .push_and_make_operand(
                    jit_ir::LoadTraceInputInstruction::new(0, jit_mod.int8_type_idx()).into(),
                )
                .unwrap();
            jit_mod.push(jit_ir::GuardInstruction::new(cond_op, false).into());
            let patt_lines = [
                "...",
                "; Guard %0, false",
                "... 0000001b: jmp 0x0000000000000022",
                "... 00000020: ud2",
                "... 00000022: cmp r12b, 0x00",
                "... 00000026: jnz 0x0000000000000020",
            ];
            test_with_spillalloc(&jit_mod, &patt_lines);
        }
    }
}
