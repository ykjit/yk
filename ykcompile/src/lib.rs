#![feature(proc_macro_hygiene)]
#![feature(test)]

#[macro_use]
extern crate dynasm;
extern crate dynasmrt;
extern crate test;

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::process::Command;

use yktrace::tir::{
    Constant, ConstantInt, Guard, Operand, Rvalue, Statement, TirOp, TirTrace, UnsignedInt,
};

use dynasmrt::DynasmApi;

#[derive(Debug)]
pub enum CompileError {
    /// We ran out of registers.
    /// In the long-run, when we have a proper register allocator, this won't be needed.
    OutOfRegisters,
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::OutOfRegisters => "Ran out of registers",
        };
        write!(f, "{}", msg)
    }
}

/// Converts a register number into it's string name.
fn reg_num_to_name(r: u8) -> &'static str {
    match r {
        0 => "rax",
        1 => "rcx",
        2 => "rdx",
        3 => "rbx",
        4 => "rsp",
        5 => "rbp",
        6 => "rsi",
        7 => "rdi",
        8 => "r8",
        9 => "r9",
        10 => "r10",
        11 => "r11",
        12 => "r12",
        13 => "r13",
        14 => "r14",
        15 => "r15",
        _ => unimplemented!(),
    }
}

/// A compiled SIRTrace.
pub struct CompiledTrace {
    /// A compiled trace.
    mc: dynasmrt::ExecutableBuffer,
}

impl CompiledTrace {
    pub fn execute(&self) -> u64 {
        // For now a compiled trace always returns whatever has been left in register RAX. We also
        // assume for now that this will be a `u64`.
        let func: fn() -> u64 = unsafe { mem::transmute(self.mc.ptr(dynasmrt::AssemblyOffset(0))) };
        func()
    }
}

/// The `TraceCompiler` takes a `SIRTrace` and compiles it to machine code. Returns a `CompiledTrace`.
pub struct TraceCompiler {
    asm: dynasmrt::x64::Assembler,
    /// Contains the list of currently available registers.
    available_regs: Vec<u8>,
    /// Maps locals to their assigned registers.
    assigned_regs: HashMap<u32, u8>,
}

impl TraceCompiler {
    fn local_to_reg(&mut self, l: u32) -> Result<u8, CompileError> {
        // This is a really dumb register allocator, which runs out of available registers after 7
        // locals. We can do better than this by using StorageLive/StorageDead from the MIR to free
        // up registers again, and allocate additional locals on the stack. Though, ultimately we
        // probably want to implement a proper register allocator, e.g. linear scan.

        if l == 0 {
            // In SIR, `Local` zero is the (implicit) return value, so it makes sense to allocate
            // it to the return register of the underlying X86_64 calling convention.
            Ok(0)
        } else {
            if self.assigned_regs.contains_key(&l) {
                Ok(self.assigned_regs[&l])
            } else {
                if let Some(reg) = self.available_regs.pop() {
                    self.assigned_regs.insert(l, reg);
                    Ok(reg)
                } else {
                    Err(CompileError::OutOfRegisters)
                }
            }
        }
    }

    /// Move constant `c` of type `usize` into local `a`.
    pub fn mov_local_usize(&mut self, local: u32, cnst: usize) -> Result<(), CompileError> {
        let reg = self.local_to_reg(local)?;
        dynasm!(self.asm
            ; mov Rq(reg), cnst as i32
        );
        Ok(())
    }

    /// Move constant `c` of type `u8` into local `a`.
    pub fn mov_local_u8(&mut self, local: u32, cnst: u8) -> Result<(), CompileError> {
        let reg = self.local_to_reg(local)?;
        dynasm!(self.asm
            ; mov Rq(reg), cnst as i32
        );
        Ok(())
    }

    /// Move local `var2` into local `var1`.
    fn mov_local_local(&mut self, l1: u32, l2: u32) -> Result<(), CompileError> {
        let lreg = self.local_to_reg(l1)?;
        let rreg = self.local_to_reg(l2)?;
        dynasm!(self.asm
            ; mov Rq(lreg), Rq(rreg)
        );
        Ok(())
    }

    fn nop(&mut self) {
        dynasm!(self.asm
            ; nop
        );
    }

    fn c_mov_int(&mut self, local: u32, constant: &ConstantInt) -> Result<(), CompileError> {
        let reg = self.local_to_reg(local)?;
        let val = match constant {
            ConstantInt::UnsignedInt(UnsignedInt::U8(i)) => *i as i64,
            ConstantInt::UnsignedInt(UnsignedInt::Usize(i)) => *i as i64,
            e => todo!("SignedInt, etc: {}", e),
        };
        dynasm!(self.asm
            ; mov Rq(reg), QWORD val
        );
        Ok(())
    }

    fn c_mov_bool(&mut self, local: u32, b: bool) -> Result<(), CompileError> {
        let reg = self.local_to_reg(local)?;
        dynasm!(self.asm
            ; mov Rq(reg), QWORD b as i64
        );
        Ok(())
    }

    fn statement(&mut self, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::Assign(l, r) => {
                let local = l.local.0;
                match r {
                    Rvalue::Use(Operand::Place(p)) => self.mov_local_local(local, p.local.0)?,
                    Rvalue::Use(Operand::Constant(c)) => match c {
                        Constant::Int(ci) => self.c_mov_int(local, ci)?,
                        Constant::Bool(b) => self.c_mov_bool(local, *b)?,
                        c => todo!("Not implemented: {}", c),
                    },
                    unimpl => todo!("Not implemented: {:?}", unimpl),
                };
            }
            Statement::Return => {}
            Statement::Nop => {}
            Statement::Unimplemented(mir_stmt) => todo!("Can't compile: {}", mir_stmt),
        }

        Ok(())
    }

    fn guard(&mut self, _grd: &Guard) -> Result<(), CompileError> {
        self.nop(); // FIXME compile guards
        Ok(())
    }

    fn crash_dump(self, e: CompileError) -> ! {
        eprintln!("\nThe trace compiler crashed!\n");
        eprintln!("Reason: {}.\n", e);

        // To help us figure out what has gone wrong, we can print the disassembled instruction
        // stream with the help of `rasm2`.
        eprintln!("Executable code buffer:");
        let code = &*self.asm.finalize().unwrap();
        if code.is_empty() {
            eprintln!("  <empty buffer>");
        } else {
            let hex_code = hex::encode(code);
            let res = Command::new("rasm2")
                .arg("-d")
                .arg("-b 64") // x86_64.
                .arg(hex_code.clone())
                .output()
                .unwrap();
            if !res.status.success() {
                eprintln!("  Failed to invoke rasm2. Raw bytes follow...");
                eprintln!("  {}", hex_code);
            } else {
                let asm = String::from_utf8(res.stdout).unwrap();
                for line in asm.lines() {
                    eprintln!("  {}", line);
                }
            }
        }

        // Print the register allocation.
        eprintln!("\nRegister allocation (local -> reg):");
        for (local, reg) in &self.assigned_regs {
            eprintln!("  {:2} -> {:3} ({})", local, reg, reg_num_to_name(*reg));
        }
        eprintln!();

        panic!("stopped due to trace compilation error");
    }

    fn finish(mut self) -> dynasmrt::ExecutableBuffer {
        dynasm!(self.asm
            ; ret
        );
        self.asm.finalize().unwrap()
    }

    pub fn compile(tt: TirTrace) -> CompiledTrace {
        let assembler = dynasmrt::x64::Assembler::new().unwrap();

        let mut tc = TraceCompiler {
            asm: assembler,
            // Use all the 64-bit registers we can (R15-R8, RDX, RCX).
            available_regs: vec![15, 14, 13, 12, 11, 10, 9, 8, 2, 1],
            assigned_regs: HashMap::new(),
        };

        for i in 0..tt.len() {
            let res = match tt.op(i) {
                TirOp::Statement(st) => tc.statement(st),
                TirOp::Guard(g) => tc.guard(g),
            };

            if let Err(e) = res {
                tc.crash_dump(e);
            }
        }

        CompiledTrace { mc: tc.finish() }
    }
}

#[cfg(test)]
mod tests {
    use super::{HashMap, TraceCompiler};
    use yktrace::tir::TirTrace;
    use yktrace::{start_tracing, TracingKind};

    #[inline(never)]
    fn simple() -> u8 {
        let x = 13;
        x
    }

    #[test]
    pub(crate) fn test_simple() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        simple();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::compile(tir_trace);
        assert_eq!(ct.execute(), 13);
    }

    // Repeatedly fetching the register for the same local should yield the same register and
    // should not exhaust the allocator.
    #[test]
    pub fn reg_alloc_same_local() {
        let mut tc = TraceCompiler {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            available_regs: vec![15, 14, 13, 12, 11, 10, 9, 8, 2, 1],
            assigned_regs: HashMap::new(),
        };

        for _ in 0..32 {
            assert_eq!(tc.local_to_reg(1).unwrap(), tc.local_to_reg(1).unwrap());
        }
    }
}
