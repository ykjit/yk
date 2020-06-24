//! The Yorick TIR trace compiler.

#![feature(proc_macro_hygiene)]
#![feature(test)]
#![feature(core_intrinsics)]

#[macro_use]
extern crate dynasm;
extern crate dynasmrt;
extern crate test;

mod stack_builder;

use libc::{c_void, dlsym, RTLD_DEFAULT};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::process::Command;

use yktrace::tir::{
    CallOperand, Constant, ConstantInt, Guard, Local, Operand, Place, Rvalue, Statement, TirOp,
    TirTrace,
};

use dynasmrt::{DynasmApi, DynasmLabelApi};

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum CompileError {
    /// No return local found.
    NoReturnLocal,
    /// We ran out of registers.
    /// In the long-run, when we have a proper register allocator, this won't be needed.
    OutOfRegisters,
    /// Compiling this statement is not yet implemented.
    /// The string inside is a hint as to what kind of statement needs to be implemented.
    Unimplemented(String),
    /// The binary symbol could not be found.
    UnknownSymbol(String),
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoReturnLocal => write!(f, "No return local found"),
            Self::OutOfRegisters => write!(f, "Ran out of registers"),
            Self::Unimplemented(s) => write!(f, "Unimplemented compilation: {}", s),
            Self::UnknownSymbol(s) => write!(f, "Unknown symbol: {}", s),
        }
    }
}

/// Converts a register number into it's string name.
fn local_to_reg_name(loc: &Location) -> &'static str {
    match loc {
        Location::Register(r) => match r {
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
        },
        _ => "",
    }
}

/// A compiled `SIRTrace`.
pub struct CompiledTrace {
    /// A compiled trace.
    mc: dynasmrt::ExecutableBuffer,
}

impl CompiledTrace {
    /// Execute the trace by calling (not jumping to) the first instruction's address.
    pub fn execute(&self) -> u64 {
        // For now a compiled trace always returns whatever has been left in register RAX. We also
        // assume for now that this will be a `u64`.
        let func: fn() -> u64 = unsafe { mem::transmute(self.mc.ptr(dynasmrt::AssemblyOffset(0))) };
        self.exec_trace(func)
    }

    /// Actually call the code. This is a separate unmangled function to make it easy to set a
    /// debugger breakpoint right before entering the trace.
    #[no_mangle]
    fn exec_trace(&self, t_fn: fn() -> u64) -> u64 {
        t_fn()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Location {
    Register(u8),
    Stack(i32),
    NotLive,
}

/// The `TraceCompiler` takes a `SIRTrace` and compiles it to machine code. Returns a `CompiledTrace`.
pub struct TraceCompiler {
    /// The dynasm assembler which will do all of the heavy lifting of the assembly.
    asm: dynasmrt::x64::Assembler,
    /// Stores the content of each register.
    register_content_map: HashMap<u8, Option<Local>>,
    /// Maps trace locals to their location (register, stack).
    variable_location_map: HashMap<Local, Location>,
    /// Stores the destination local of the outermost function and moves its content into RAX at
    /// the end of the trace.
    rtn_var: Option<Place>,
    /// The amount of stack space in bytes used so far by spilled variables.
    cur_stack_offset: i32,
}

impl TraceCompiler {
    /// Given a local, returns the register allocation for it, or, if there is no allocation yet,
    /// performs one.
    fn local_to_location(&mut self, l: Local) -> Result<Location, CompileError> {
        if l == Local(0) {
            // In SIR, `Local` zero is the (implicit) return value, so it makes sense to allocate
            // it to the return register of the underlying X86_64 calling convention.
            Ok(Location::Register(0))
        } else {
            if let Some(location) = self.variable_location_map.get(&l) {
                // We already have a location for this local.
                Ok(location.clone())
            } else {
                // Find a free register to store this local.
                let loc = if let Some(reg) = self.register_content_map.iter().find_map(|(k, v)| {
                    if v == &None {
                        Some(*k)
                    } else {
                        None
                    }
                }) {
                    self.register_content_map.insert(reg, Some(l));
                    Location::Register(reg)
                } else {
                    // All registers are occupied, so we need to spill the local to the stack. For
                    // now we assume that all spilled locals are 8 bytes big.
                    self.cur_stack_offset += 8;
                    let loc = Location::Stack(self.cur_stack_offset);
                    loc
                };
                let ret = loc.clone();
                self.variable_location_map.insert(l, loc);
                Ok(ret)
            }
        }
    }

    /// Notifies the register allocator that the register allocated to `local` may now be re-used.
    fn free_register(&mut self, local: &Local) -> Result<(), CompileError> {
        let is_rtn_var = local
            == &self
                .rtn_var
                .as_ref()
                .ok_or_else(|| CompileError::NoReturnLocal)?
                .local;
        match self.variable_location_map.get(local) {
            Some(Location::Register(reg)) => {
                // If this local is currently stored in a register, free it.
                self.register_content_map.insert(*reg, None);
                if is_rtn_var {
                    // We currently assume that we only trace a single function which leaves its return
                    // value in RAX. Since we now inline a function's return variable this won't happen
                    // automatically anymore. To keep things working, we thus copy the return value of
                    // the outer-most function into RAX at the end of the trace.
                    dynasm!(self.asm
                        ; mov rax, Rq(reg)
                    );
                }
            }
            Some(Location::Stack(offset)) => {
                if is_rtn_var {
                    dynasm!(self.asm
                        ; mov rax, [rbp - *offset]
                    );
                }
            }
            Some(Location::NotLive) => {}
            None => {}
        }
        self.variable_location_map.insert(*local, Location::NotLive);
        Ok(())
    }

    /// Copy the contents of the local `l2` into  `l1`.
    fn mov_local_local(&mut self, l1: Local, l2: Local) -> Result<(), CompileError> {
        let lloc = self.local_to_location(l1)?;
        let rloc = self.local_to_location(l2)?;
        match (lloc, rloc) {
            (Location::Register(lreg), Location::Register(rreg)) => {
                dynasm!(self.asm
                    ; mov Rq(lreg), Rq(rreg)
                );
            }
            (Location::Register(reg), Location::Stack(off)) => {
                dynasm!(self.asm
                    ; mov Rq(reg), [rbp - off]
                );
            }
            (Location::Stack(off), Location::Register(reg)) => {
                dynasm!(self.asm
                    ; mov [rbp - off], Rq(reg)
                );
            }
            (Location::Stack(off1), Location::Stack(off2)) => {
                // Since RAX is currently not available to the register allocator, we can use it
                // here to simplify moving values from the stack back onto the stack (which x86
                // does not support). Otherwise, we would have to free up a register via spilling,
                // making this operation more complicated and costly.
                dynasm!(self.asm
                    ; mov rax, [rbp - off2]
                    ; mov [rbp - off1], rax
                );
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Emit a NOP operation.
    fn nop(&mut self) {
        dynasm!(self.asm
            ; nop
        );
    }

    /// Move a constant integer into a `Local`.
    fn mov_local_constint(
        &mut self,
        local: Local,
        constant: &ConstantInt,
    ) -> Result<(), CompileError> {
        let loc = self.local_to_location(local)?;
        let c_val = constant.i64_cast();
        match loc {
            Location::Register(reg) => {
                dynasm!(self.asm
                    ; mov Rq(reg), QWORD c_val
                );
            }
            Location::Stack(offset) => {
                if c_val <= u32::MAX.into() {
                    let val = c_val as u32 as i32;
                    dynasm!(self.asm
                        ; mov QWORD [rbp-offset], val
                    );
                } else {
                    // x86 doesn't allow writing 64bit immediates directly to the stack. We thus
                    // have to split up the immediate into two 32bit values and write them one at a
                    // time.
                    let v1 = c_val as u32 as i32;
                    let v2 = (c_val >> 32) as u32 as i32;
                    dynasm!(self.asm
                        ; mov DWORD [rbp-offset], v1
                        ; mov DWORD [rbp-offset+4], v2
                    );
                }
            }
            Location::NotLive => unreachable!(),
        }
        Ok(())
    }

    /// Move a Boolean into a `Local`.
    fn mov_local_bool(&mut self, local: Local, b: bool) -> Result<(), CompileError> {
        match self.local_to_location(local)? {
            Location::Register(reg) => {
                dynasm!(self.asm
                    ; mov Rq(reg), QWORD b as i64
                );
            }
            Location::Stack(offset) => {
                let val = b as i32;
                dynasm!(self.asm
                    ; mov QWORD [rbp-offset], val
                );
            }
            Location::NotLive => unreachable!(),
        }
        Ok(())
    }

    /// Compile the entry into an inlined function call.
    fn c_enter(
        &mut self,
        op: &CallOperand,
        args: &Vec<Operand>,
        dest: &Option<Place>,
        off: u32,
    ) -> Result<(), CompileError> {
        // FIXME Currently, we still get a call to `stop_tracing` here, since the call is part of
        // the last block in the trace. We may be able to always skip the last n instructions of the
        // trace, but this requires some looking into to make sure we don't accidentally skip other
        // things. So for now, let's just skip the call here to get the tests working.
        match op {
            ykpack::CallOperand::Fn(s) => {
                if s.contains("stop_tracing") {
                    return Ok(());
                }
            }
            ykpack::CallOperand::Unknown => {}
        };
        // Move call arguments into registers.
        for (op, i) in args.iter().zip(1..) {
            let arg_idx = Local(i + off);
            match op {
                Operand::Place(p) => self.mov_local_local(arg_idx, p.local)?,
                Operand::Constant(c) => match c {
                    Constant::Int(ci) => self.mov_local_constint(arg_idx, ci)?,
                    Constant::Bool(b) => self.mov_local_bool(arg_idx, *b)?,
                    c => return Err(CompileError::Unimplemented(format!("{}", c))),
                },
            }
        }
        if self.rtn_var.is_none() {
            // Remember the return variable of the most outer function.
            self.rtn_var = dest.as_ref().cloned();
        }
        Ok(())
    }

    /// Compile a call to a native symbol using the Sys-V ABI. This is used for occasions where you
    /// don't want to, or cannot, inline the callee (e.g. it's a foreign function).
    ///
    /// For now we do something very simple. There are limitations (FIXME):
    ///
    ///  - We assume there are no more than 6 arguments (spilling is not yet implemented).
    ///
    ///  - We push all of the callee save registers on the stack, and local variable arguments are
    ///    then loaded back from the stack into the correct ABI-specified registers. We can
    ///    optimise this later by only loading an argument from the stack if it cannot be loaded
    ///    from its original register location (because another argument overwrote it already).
    ///
    ///  - We assume the return value fits in rax. 128-bit return values are not yet supported.
    ///
    ///  - We don't support varags calls.
    fn c_call(
        &mut self,
        opnd: &CallOperand,
        args: &Vec<Operand>,
        dest: &Option<Place>,
    ) -> Result<(), CompileError> {
        let sym = if let CallOperand::Fn(sym) = opnd {
            sym
        } else {
            return Err(CompileError::Unimplemented(
                "unknown call target".to_owned(),
            ));
        };

        if args.len() > 6 {
            return Err(CompileError::Unimplemented(
                "call with spilled args".to_owned(),
            ));
        }

        // Figure out where the return value (if there is one) is going.
        let dest_location: Option<Location> = if let Some(d) = dest {
            Some(self.local_to_location(d.local)?)
        } else {
            None
        };

        let dest_reg: Option<u8> = match dest_location {
            Some(Location::Register(reg)) => Some(reg),
            _ => None,
        };

        // Save Sys-V caller save registers to the stack, but skip the one (if there is one) that
        // will store the return value. It's safe to assume the caller expects this to be
        // clobbered.
        //
        // FIXME: Note that we don't save rax. Although this is a caller save register, the way the
        // tests currently work is they check the last value returned at the end of the trace. This
        // value is assumed to remain in rax. If we were to restore rax, we'd break that. Note that
        // the register allocator never gives out rax for this precise reason.
        //
        // rdi, rsi, rdx, rcx, r8, r9, r10, r11.
        let save_regs = [7, 6, 2, 1, 8, 9, 10, 11]
            .iter()
            .filter(|r| Some(**r) != dest_reg)
            .map(|r| *r)
            .collect::<Vec<u8>>();
        for reg in &save_regs {
            dynasm!(self.asm
                ; push Rq(reg)
            );
        }

        // Helper function to find the index of a caller-save register previously pushed to the stack.
        // The first register pushed is at the highest stack offset (from the stack pointer), hence
        // reversing the order of `save_regs`.
        let stack_index = |reg: u8| -> i32 {
            i32::try_from(save_regs.iter().rev().position(|&r| r == reg).unwrap()).unwrap()
        };

        // Sys-V ABI dictates the first 6 arguments are passed in: rdi, rsi, rdx, rcx, r8, r9.
        let mut arg_regs = vec![9, 8, 1, 2, 6, 7]; // reversed so they pop() in the right order.
        for arg in args {
            // `unwrap()` must succeed, as we checked there are no more than 6 args above.
            let arg_reg = arg_regs.pop().unwrap();

            match arg {
                Operand::Place(Place { local, projection }) => {
                    if !projection.is_empty() {
                        return Err(CompileError::Unimplemented("projected argument".to_owned()));
                    }
                    // Load argument back from the stack.
                    match self.local_to_location(*local)? {
                        Location::Register(reg) => {
                            let off = stack_index(reg) * 8;
                            dynasm!(self.asm
                                ; mov Rq(arg_reg), [rsp + off]
                            );
                        }
                        Location::Stack(off) => {
                            dynasm!(self.asm
                                ; mov Rq(arg_reg), [rbp - off]
                            );
                        }
                        Location::NotLive => unreachable!(),
                    };
                }
                Operand::Constant(c) => {
                    dynasm!(self.asm
                        ; mov Rq(arg_reg), QWORD c.i64_cast()
                    );
                }
            };
        }

        let sym_addr = TraceCompiler::find_symbol(sym)? as i64;
        dynasm!(self.asm
            // In Sys-V ABI, `al` is a hidden argument used to specify the number of vector args
            // for a vararg call. We don't support this right now, so set it to zero.
            ; xor rax, rax
            ; mov r11, QWORD sym_addr
            ; call r11
        );

        // Put return value in place.
        match dest_location {
            Some(Location::Register(reg)) => {
                dynasm!(self.asm
                    ; mov Rq(reg), rax
                );
            }
            Some(Location::Stack(off)) => {
                dynasm!(self.asm
                    ; mov QWORD [rbp-off], rax
                );
            }
            _ => unreachable!(),
        }

        // To avoid breaking tests we need the same hack as `c_enter()` uses for now.
        if self.rtn_var.is_none() {
            self.rtn_var = dest.as_ref().cloned();
        }

        // Restore caller-save registers.
        for reg in save_regs.iter().rev() {
            dynasm!(self.asm
                ; pop Rq(reg)
            );
        }

        Ok(())
    }

    /// Compile a TIR statement.
    fn statement(&mut self, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::Assign(l, r) => {
                if !l.projection.is_empty() {
                    return Err(CompileError::Unimplemented(format!("{}", l)));
                }
                match r {
                    Rvalue::Use(Operand::Place(p)) => {
                        if !p.projection.is_empty() {
                            return Err(CompileError::Unimplemented(format!("{}", r)));
                        }
                        self.mov_local_local(l.local, p.local)?;
                    }
                    Rvalue::Use(Operand::Constant(c)) => match c {
                        Constant::Int(ci) => self.mov_local_constint(l.local, ci)?,
                        Constant::Bool(b) => self.mov_local_bool(l.local, *b)?,
                        c => return Err(CompileError::Unimplemented(format!("{}", c))),
                    },
                    unimpl => return Err(CompileError::Unimplemented(format!("{}", unimpl))),
                };
            }
            Statement::Enter(op, args, dest, off) => self.c_enter(op, args, dest, *off)?,
            Statement::Leave => {}
            Statement::StorageLive(_) => {}
            Statement::StorageDead(l) => self.free_register(l)?,
            Statement::Call(target, args, dest) => self.c_call(target, args, dest)?,
            Statement::Nop => {}
            Statement::Unimplemented(s) => {
                return Err(CompileError::Unimplemented(format!("{:?}", s)))
            }
        }

        Ok(())
    }

    /// Compile a guard in the trace, emitting code to abort execution in case the guard fails.
    fn guard(&mut self, _grd: &Guard) -> Result<(), CompileError> {
        self.nop(); // FIXME compile guards
        Ok(())
    }

    /// Print information about the state of the compiler in the hope that it can help with
    /// debugging efforts.
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
        for (local, location) in &self.variable_location_map {
            eprintln!(
                "  {:2} -> {:?} ({})",
                local,
                location,
                local_to_reg_name(location)
            );
        }
        eprintln!();

        panic!("stopped due to trace compilation error");
    }

    /// Emit a return instruction.
    fn ret(&mut self) {
        // Reset the stack/base pointers and return from the trace. We also need to generate the
        // code that reserves stack space for spilled locals here, since we don't know at the
        // beginning of the trace how many locals are going to be spilled.
        dynasm!(self.asm
            ; add rsp, self.cur_stack_offset
            ; pop rbp
            ; ret
            ; ->reserve:
            ; push rbp
            ; mov rbp, rsp
            ; sub rsp, self.cur_stack_offset
            ; jmp ->main
        );
    }

    fn init(&mut self) {
        // Jump to the label that reserves stack space for spilled locals.
        dynasm!(self.asm
            ; jmp ->reserve
            ; ->main:
        );
    }

    /// Finish compilation and return the executable code that was assembled.
    fn finish(self) -> dynasmrt::ExecutableBuffer {
        self.asm.finalize().unwrap()
    }

    #[cfg(test)]
    fn test_compile(tt: TirTrace) -> (CompiledTrace, i32) {
        // Changing the registers available to the register allocator affects the number of spills,
        // and thus also some tests. To make sure we notice when this happens we also check the
        // number of spills in those tests. We thus need a slightly different version of the
        // `compile` function that provides this information to the test.
        let tc = TraceCompiler::_compile(tt);
        let spills = tc.cur_stack_offset;
        let ct = CompiledTrace { mc: tc.finish() };
        (ct, spills)
    }

    /// Compile a TIR trace, returning executable code.
    pub fn compile(tt: TirTrace) -> CompiledTrace {
        let tc = TraceCompiler::_compile(tt);
        CompiledTrace { mc: tc.finish() }
    }

    fn _compile(tt: TirTrace) -> TraceCompiler {
        let assembler = dynasmrt::x64::Assembler::new().unwrap();

        let mut tc = TraceCompiler {
            asm: assembler,
            // Use all the 64-bit registers we can (R11-R8, RDX, RCX). We probably also want to use the
            // callee-saved registers R15-R12 here in the future.
            register_content_map: [11, 10, 9, 8, 2, 1]
                .iter()
                .cloned()
                .map(|r| (r, None))
                .collect(),
            variable_location_map: HashMap::new(),
            rtn_var: None,
            cur_stack_offset: 0,
        };

        tc.init();

        for i in 0..tt.len() {
            let res = match tt.op(i) {
                TirOp::Statement(st) => tc.statement(st),
                TirOp::Guard(g) => tc.guard(g),
            };

            if let Err(e) = res {
                tc.crash_dump(e);
            }
        }

        tc.ret();
        tc
    }

    /// Returns a pointer to the static symbol `sym`, or an error if it cannot be found.
    fn find_symbol(sym: &str) -> Result<*mut c_void, CompileError> {
        use std::ffi::CString;

        let sym_arg = CString::new(sym).unwrap();
        let addr = unsafe { dlsym(RTLD_DEFAULT, sym_arg.into_raw()) };

        if addr == 0 as *mut c_void {
            Err(CompileError::UnknownSymbol(sym.to_owned()))
        } else {
            Ok(addr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompileError, HashMap, Local, Location, TraceCompiler};
    use libc::{abs, c_void, getuid};
    use yktrace::tir::{CallOperand, Statement, TirOp, TirTrace};
    use yktrace::{start_tracing, TracingKind};

    #[inline(never)]
    fn simple() -> u8 {
        let x = 13;
        x
    }

    #[test]
    fn test_simple() {
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
    fn reg_alloc_same_local() {
        let mut tc = TraceCompiler {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: [15, 14, 13, 12, 11, 10, 9, 8, 2, 1]
                .iter()
                .cloned()
                .map(|r| (r, None))
                .collect(),
            variable_location_map: HashMap::new(),
            rtn_var: None,
            cur_stack_offset: 8,
        };

        for _ in 0..32 {
            assert_eq!(
                tc.local_to_location(Local(1)).unwrap(),
                tc.local_to_location(Local(1)).unwrap()
            );
        }
    }

    // Locals should be allocated to different registers.
    #[test]
    fn reg_alloc() {
        let mut tc = TraceCompiler {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: [15, 14, 13, 12, 11, 10, 9, 8, 2, 1]
                .iter()
                .cloned()
                .map(|r| (r, None))
                .collect(),
            variable_location_map: HashMap::new(),
            rtn_var: None,
            cur_stack_offset: 0,
        };

        let mut seen: Vec<Result<Location, CompileError>> = Vec::new();
        for l in 0..7 {
            let reg = tc.local_to_location(Local(l));
            assert!(!seen.contains(&reg));
            seen.push(reg);
        }
    }

    #[inline(never)]
    fn farg(i: u8) -> u8 {
        i
    }

    #[inline(never)]
    fn fcall() -> u8 {
        let y = farg(13); // assigns 13 to $1
        let _z = farg(14); // overwrites $1 within the call
        y // returns $1
    }

    #[test]
    fn test_function_call_simple() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        fcall();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::compile(tir_trace);
        assert_eq!(ct.execute(), 13);
    }

    fn fnested3(i: u8, _j: u8) -> u8 {
        let c = i;
        c
    }

    fn fnested2(i: u8) -> u8 {
        fnested3(i, 10)
    }

    fn fnested() -> u8 {
        let a = fnested2(20);
        a
    }

    #[test]
    fn test_function_call_nested() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        fnested();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::compile(tir_trace);
        assert_eq!(ct.execute(), 20);
    }

    // Test finding a symbol in a shared object.
    #[test]
    fn find_symbol_shared() {
        assert!(TraceCompiler::find_symbol("printf") == Ok(libc::printf as *mut c_void));
    }

    // Test finding a symbol in the main binary.
    // For this to work the binary must have been linked with `--export-dynamic`, which ykrustc
    // appends to the linker command line.
    #[test]
    #[no_mangle]
    fn find_symbol_main() {
        assert!(
            TraceCompiler::find_symbol("find_symbol_main") == Ok(find_symbol_main as *mut c_void)
        );
    }

    // Check that a non-existent symbol cannot be found.
    #[test]
    fn find_nonexistent_symbol() {
        assert_eq!(
            TraceCompiler::find_symbol("__xxxyyyzzz__"),
            Err(CompileError::UnknownSymbol("__xxxyyyzzz__".to_owned()))
        );
    }

    // A trace which contains a call to something which we don't have SIR for should emit a TIR
    // call operation.
    #[test]
    fn call_symbol() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let _ = core::intrinsics::wrapping_add(10u64, 40u64);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();

        let mut found_call = false;
        for i in 0..tir_trace.len() {
            if let TirOp::Statement(Statement::Call(CallOperand::Fn(sym), ..)) = tir_trace.op(i) {
                if sym.contains("wrapping_add") {
                    found_call = true;
                }
                break;
            }
        }
        assert!(found_call);
    }

    /// Execute a trace which calls a symbol accepting no arguments, but which does return a value.
    #[test]
    fn exec_call_symbol_no_args() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let expect = unsafe { getuid() };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let got = TraceCompiler::compile(tir_trace).execute();
        assert_eq!(expect as u64, got);
    }

    /// Execute a trace which calls a symbol accepting arguments and returns a value.
    #[test]
    fn exec_call_symbol_with_arg() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let v = -56;
        let expect = unsafe { abs(v) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let got = TraceCompiler::compile(tir_trace).execute();
        assert_eq!(expect as u64, got);
    }

    /// The same as `exec_call_symbol_args_with_rv`, just using a constant argument.
    #[test]
    fn exec_call_symbol_with_const_arg() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let expect = unsafe { abs(-123) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let got = TraceCompiler::compile(tir_trace).execute();
        assert_eq!(expect as u64, got);
    }

    #[test]
    fn exec_call_symbol_with_many_args() {
        extern "C" {
            fn add6(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) -> u64;
        }

        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let expect = unsafe { add6(1, 2, 3, 4, 5, 6) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let got = TraceCompiler::compile(tir_trace).execute();
        assert_eq!(expect, 21);
        assert_eq!(expect, got);
    }

    #[test]
    fn exec_call_symbol_with_many_args_some_ignored() {
        extern "C" {
            fn add_some(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
        }

        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let expect = unsafe { add_some(1, 2, 3, 4, 5) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let got = TraceCompiler::compile(tir_trace).execute();
        assert_eq!(expect, 7);
        assert_eq!(expect, got);
    }

    fn many_locals() -> u8 {
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _f = 6;
        let h = 7;
        let _g = true;
        h
    }

    #[test]
    fn test_spilling_simple() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        many_locals();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::test_compile(tir_trace);
        assert_eq!(ct.execute(), 7);
        assert_eq!(spills, 3 * 8);
    }

    fn u64value() -> u64 {
        // We need an extra function here to avoid SIR optimising this by assigning assigning the
        // constant directly to the return value (which is a register).
        4294967296 + 8
    }

    #[inline(never)]
    fn spill_u64() -> u64 {
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _f = 6;
        let h: u64 = u64value();
        h
    }

    #[test]
    fn test_spilling_u64() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        spill_u64();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::test_compile(tir_trace);
        let got = ct.execute();
        assert_eq!(got, 4294967296 + 8);
        assert_eq!(spills, 2 * 8);
    }

    fn register_to_stack(arg: u8) -> u8 {
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _f = 6;
        let h = arg;
        h
    }

    #[test]
    fn test_mov_register_to_stack() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        register_to_stack(8);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::test_compile(tir_trace);
        assert_eq!(ct.execute(), 8);
        assert_eq!(spills, 3 * 8);
    }

    fn stack_to_register() -> u8 {
        let _a = 1;
        let _b = 2;
        let c = 3;
        let _d = 4;
        // When returning from `farg` all registers are full, so `e` needs to be allocated on the
        // stack. However, after we have returned, anything allocated during `farg` is freed. Thus
        // returning `e` will allocate a new local in a (newly freed) register, resulting in a `mov
        // reg, [rbp]` instruction.
        let e = farg(c);
        e
    }

    #[test]
    fn test_mov_stack_to_register() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        stack_to_register();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::test_compile(tir_trace);
        assert_eq!(ct.execute(), 3);
        assert_eq!(spills, 1 * 8);
    }

    fn ext_call() -> u64 {
        extern "C" {
            fn add_some(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
        }
        let a = 1;
        let b = 2;
        let c = 3;
        let d = 4;
        let e = 5;
        // When calling `add_some` argument `a` is loaded from a register, while the remaining
        // arguments are loaded from the stack.
        let expect = unsafe { add_some(a, b, c, d, e) };
        expect
    }

    #[test]
    fn ext_call_and_spilling() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let expect = ext_call();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let got = TraceCompiler::compile(tir_trace).execute();
        assert_eq!(expect, 7);
        assert_eq!(expect, got);
    }
}
