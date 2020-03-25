#![feature(proc_macro_hygiene)]
#![feature(test)]

#[macro_use]
extern crate dynasm;
extern crate dynasmrt;
extern crate test;

use std::collections::HashMap;
use std::mem;

use yktrace::tir::{
    Constant, ConstantInt, Operand, Rvalue, Statement, TirOp, TirTrace, UnsignedInt,
};

use dynasmrt::DynasmApi;

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
    fn local_to_reg(&mut self, l: u32) -> u8 {
        // This is a really dumb register allocator, which runs out of available registers after 7
        // locals. We can do better than this by using StorageLive/StorageDead from the MIR to free
        // up registers again, and allocate additional locals on the stack. Though, ultimately we
        // probably want to implement a proper register allocator, e.g. linear scan.
        if l == 0 {
            0
        } else {
            let reg = self
                .available_regs
                .pop()
                .expect("Can't allocate more than 7 locals yet!");
            *self.assigned_regs.entry(l).or_insert(reg)
        }
    }

    /// Move constant `c` of type `usize` into local `a`.
    pub fn mov_local_usize(&mut self, local: u32, cnst: usize) {
        let reg = self.local_to_reg(local);
        dynasm!(self.asm
            ; mov Rq(reg), cnst as i32
        );
    }

    /// Move constant `c` of type `u8` into local `a`.
    pub fn mov_local_u8(&mut self, local: u32, cnst: u8) {
        let reg = self.local_to_reg(local);
        dynasm!(self.asm
            ; mov Rq(reg), cnst as i32
        );
    }

    /// Move local `var2` into local `var1`.
    fn mov_local_local(&mut self, l1: u32, l2: u32) {
        let lreg = self.local_to_reg(l1);
        let rreg = self.local_to_reg(l2);
        dynasm!(self.asm
            ; mov Rq(lreg), Rq(rreg)
        );
    }

    fn nop(&mut self) {
        dynasm!(self.asm
            ; nop
        );
    }

    fn c_mov_int(&mut self, local: u32, constant: &ConstantInt) {
        let reg = self.local_to_reg(local);
        let val = match constant {
            ConstantInt::UnsignedInt(UnsignedInt::U8(i)) => *i as i64,
            ConstantInt::UnsignedInt(UnsignedInt::Usize(i)) => *i as i64,
            e => todo!("SignedInt, etc: {}", e),
        };
        dynasm!(self.asm
            ; mov Rq(reg), QWORD val
        );
    }

    fn c_mov_bool(&mut self, local: u32, b: bool) {
        let reg = self.local_to_reg(local);
        dynasm!(self.asm
            ; mov Rq(reg), QWORD b as i64
        );
    }

    fn statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Assign(l, r) => {
                let local = l.local.0;
                match r {
                    Rvalue::Use(Operand::Place(p)) => self.mov_local_local(local, p.local.0),
                    Rvalue::Use(Operand::Constant(c)) => match c {
                        Constant::Int(ci) => self.c_mov_int(local, ci),
                        Constant::Bool(b) => self.c_mov_bool(local, *b),
                        c => todo!("Not implemented: {}", c),
                    },
                    unimpl => todo!("Not implemented: {:?}", unimpl),
                };
            }
            Statement::Return => {}
            Statement::Nop => {}
            Statement::Unimplemented(mir_stmt) => todo!("Can't compile: {}", mir_stmt),
        }
    }

    fn finish(mut self) -> dynasmrt::ExecutableBuffer {
        dynasm!(self.asm
            ; ret
        );
        self.asm.finalize().unwrap()
    }

    pub fn compile(tt: TirTrace) -> CompiledTrace {
        // Set available registers to R11-R8, RDX, RCX
        let regs = vec![11, 10, 9, 8, 2, 1];
        let assembler = dynasmrt::x64::Assembler::new().unwrap();
        let mut tc = TraceCompiler {
            asm: assembler,
            available_regs: regs,
            assigned_regs: HashMap::new(),
        };
        for i in 0..tt.len() {
            let t = tt.op(i);
            match t {
                TirOp::Statement(st) => tc.statement(st),
                TirOp::Guard(_) => tc.nop(), // FIXME Implement guards.
            }
        }
        CompiledTrace { mc: tc.finish() }
    }
}

#[cfg(test)]
mod tests {

    use super::TraceCompiler;
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
}
