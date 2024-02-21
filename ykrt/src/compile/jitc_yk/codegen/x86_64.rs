//! The X86_64 JIT Code Generator.

use super::{
    super::{
        jit_ir::{self, InstrIdx, Operand},
        CompilationError,
    },
    abs_stack::AbstractStack,
    reg_alloc::{LocalAlloc, RegisterAllocator, StackDirection},
    CodeGen, CodeGenOutput,
};
use dynasmrt::{dynasm, x64::Rq, AssemblyOffset, DynasmApi, ExecutableBuffer, Register};
#[cfg(any(debug_assertions, test))]
use std::{cell::Cell, collections::HashMap, slice};

/// Argument registers as defined by the X86_64 SysV ABI.
static ARG_REGS: [Rq; 6] = [Rq::RDI, Rq::RSI, Rq::RDX, Rq::RCX, Rq::R8, Rq::R9];

/// The size of a 64-bit register in bytes.
static REG64_SIZE: usize = 8;

/// Work registers, i.e. the registers we use temproarily (where possible) for operands to, and
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
pub(super) struct X64CodeGen<'a> {
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
        let asm = dynasmrt::x64::Assembler::new()
            .map_err(|e| CompilationError::Unrecoverable(e.to_string()))?;
        Ok(Self {
            jit_mod,
            asm,
            stack: Default::default(),
            ra,
            #[cfg(any(debug_assertions, test))]
            comments: Cell::new(HashMap::new()),
        })
    }

    fn codegen(mut self) -> Result<Box<dyn CodeGenOutput>, CompilationError> {
        let alloc_off = self.emit_prologue();

        // FIXME: we'd like to be able to assemble code backwards as this would simplify register
        // allocation and side-step the need to patch up the prolog after the fact. dynasmrs
        // doesn't support this, but it's on their roadmap:
        // https://github.com/CensoredUsername/dynasm-rs/issues/48
        for (idx, inst) in self.jit_mod.instrs().iter().enumerate() {
            self.codegen_inst(jit_ir::InstrIdx::new(idx)?, inst);
        }

        // Now we know the size of the stack frame (i.e. self.asp), patch the allocation with the
        // correct amount.
        self.patch_frame_allocation(alloc_off);

        self.asm
            .commit()
            .map_err(|e| CompilationError::Unrecoverable(e.to_string()))?;

        let buf = self
            .asm
            .finalize()
            .map_err(|_| CompilationError::Unrecoverable("failed to finalize assembler".into()))?;

        #[cfg(not(any(debug_assertions, test)))]
        return Ok(Box::new(X64CodeGenOutput { buf }));
        #[cfg(any(debug_assertions, test))]
        {
            let comments = self.comments.take();
            return Ok(Box::new(X64CodeGenOutput { buf, comments }));
        }
    }
}

impl<'a> X64CodeGen<'a> {
    /// Codegen an instruction.
    fn codegen_inst(&mut self, instr_idx: jit_ir::InstrIdx, inst: &jit_ir::Instruction) {
        #[cfg(any(debug_assertions, test))]
        self.comment(self.asm.offset(), inst.to_string());
        match inst {
            jit_ir::Instruction::LoadArg(i) => self.codegen_loadarg_instr(instr_idx, &i),
            jit_ir::Instruction::Load(i) => self.codegen_load_instr(instr_idx, &i),
            _ => todo!(),
        }
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
                        // We use `movzx` where possible to avoid partial register stalls.
                        match local.instr(self.jit_mod).def_abi_size() {
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

    fn reg_into_new_local(&mut self, local: InstrIdx, reg: Rq) {
        let l = self.ra.allocate(
            local,
            local.instr(self.jit_mod).def_abi_size(),
            &mut self.stack,
        );
        self.store_local(&l, reg, local.instr(self.jit_mod).def_abi_size())
    }

    fn codegen_loadarg_instr(
        &mut self,
        inst_idx: jit_ir::InstrIdx,
        _inst: &jit_ir::LoadArgInstruction,
    ) {
        // FIXME: LoadArg instructions are not yet specified. This hack is just to satisfy data
        // flow dependencies during code generation.
        dynasm!(self.asm ; mov Rq(WR0.code()), 0);
        self.reg_into_new_local(inst_idx, WR0);
    }

    fn codegen_load_instr(&mut self, inst_idx: jit_ir::InstrIdx, inst: &jit_ir::LoadInstruction) {
        self.operand_into_reg(WR0, &inst.operand());
        let size = inst_idx.instr(self.jit_mod).def_abi_size();
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

    fn const_u64_into_reg(&mut self, reg: Rq, cv: u64) {
        dynasm!(self.asm
            ; mov Rq(reg.code()), QWORD cv as i64 // `as` intentional.
        )
    }

    /// Load an operand into a register.
    fn operand_into_reg(&mut self, reg: Rq, op: &Operand) {
        match op {
            Operand::Local(li) => self.load_local(reg, *li),
            _ => todo!("{}", op),
        }
    }
}

pub(super) struct X64CodeGenOutput {
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

impl CodeGenOutput for X64CodeGenOutput {
    #[cfg(any(debug_assertions, test))]
    fn disassemble(&self) -> String {
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
    fn to_string(&self) -> String {
        let mut out = Vec::new();
        out.push("--- Begin jit-asm ---".to_string());
        let len = self.buf.len();
        let bptr = self.buf.ptr(AssemblyOffset(0));
        let code = unsafe { slice::from_raw_parts(bptr, len) };
        // `as` is safe as it casts from a raw pointer to a pointer-sized integer.
        let mut dis =
            iced_x86::Decoder::with_ip(64, code, u64::try_from(bptr as usize).unwrap(), 0);
        let mut remain = len;
        while remain != 0 {
            let off = len - remain;
            if let Some(lines) = self.comments.get(&off) {
                for line in lines {
                    out.push(format!("; {line}"));
                }
            }
            let inst = dis.decode();
            out.push(format!("{:08x} {:08x}: {}", inst.ip(), off, inst));
            remain -= inst.len();
        }
        out.push("--- End jit-asm ---".into());
        out.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::{CodeGen, X64CodeGen, STACK_DIRECTION};
    use crate::compile::jitc_yk::{
        aot_ir,
        codegen::{
            reg_alloc::{RegisterAllocator, SpillAllocator},
            tests::match_asm,
        },
        jit_ir,
    };

    #[test]
    fn simple_codegen() {
        let mut aot_mod = aot_ir::Module::default();
        aot_mod.push_type(aot_ir::Type::Ptr);

        let mut jit_mod = jit_ir::Module::new("test".into());
        jit_mod.push(jit_ir::LoadArgInstruction::new().into());
        jit_mod.push(
            jit_ir::LoadInstruction::new(
                jit_ir::Operand::Local(jit_ir::InstrIdx::new(0).unwrap()),
                jit_ir::TypeIdx::new(0).unwrap(),
            )
            .into(),
        );
        let patt_lines = [
            "...",
            "; Load %0",
            "... 00000019: mov r12,[rbp-8]",
            "... 00000020: mov r12,[r12]",
            "... 00000025: mov [rbp-10h],r12",
            "--- End jit-asm ---",
        ];
        let mut ra = SpillAllocator::new(STACK_DIRECTION);
        match_asm(
            X64CodeGen::new(&jit_mod, &mut ra)
                .unwrap()
                .codegen()
                .unwrap(),
            &patt_lines.join("\n"),
        );
    }
}
