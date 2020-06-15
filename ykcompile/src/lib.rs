//! The Yorick TIR trace compiler.

#![feature(proc_macro_hygiene)]
#![feature(test)]
#![feature(core_intrinsics)]

#[macro_use]
extern crate dynasm;
extern crate dynasmrt;
extern crate test;

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

use dynasmrt::DynasmApi;

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

#[derive(Debug)]
enum Location {
    Register(u8),
    Stack(u8),
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
}

impl TraceCompiler {
    /// Given a local, returns the register allocation for it, or, if there is no allocation yet,
    /// performs one.
    fn local_to_reg(&mut self, l: Local) -> Result<u8, CompileError> {
        // This is a really dumb register allocator, which runs out of available registers after 7
        // locals. We can do better than this by using StorageLive/StorageDead from the MIR to free
        // up registers again, and allocate additional locals on the stack. Though, ultimately we
        // probably want to implement a proper register allocator, e.g. linear scan.

        if l == Local(0) {
            // In SIR, `Local` zero is the (implicit) return value, so it makes sense to allocate
            // it to the return register of the underlying X86_64 calling convention.
            Ok(0)
        } else {
            //if self.variable_location_map.contains_key(&l) {
            match self.variable_location_map.get(&l) {
                Some(Location::Register(reg)) => Ok(*reg),
                Some(Location::Stack(_offset)) => todo!(),
                Some(Location::NotLive) => unreachable!(),
                None => {
                    // Find a free register to store this Local
                    if let Some(reg) = self.register_content_map.iter().find_map(|(k, v)| {
                        if v == &None {
                            Some(*k)
                        } else {
                            None
                        }
                    }) {
                        self.register_content_map.insert(reg, Some(l));
                        self.variable_location_map
                            .insert(l, Location::Register(reg));
                        Ok(reg)
                    } else {
                        // All registers are occupied. In the future we need to spill here.
                        Err(CompileError::OutOfRegisters)
                    }
                }
            }
        }
    }

    /// Notifies the register allocator that the register allocated to `local` may now be re-used.
    fn free_register(&mut self, local: &Local) -> Result<(), CompileError> {
        match self.variable_location_map.get(local) {
            Some(Location::Register(reg)) => {
                // If this local is currently stored in a register, free it.
                self.register_content_map.insert(*reg, None);
                if local
                    == &self
                        .rtn_var
                        .as_ref()
                        .ok_or_else(|| CompileError::NoReturnLocal)?
                        .local
                {
                    // We currently assume that we only trace a single function which leaves its return
                    // value in RAX. Since we now inline a function's return variable this won't happen
                    // automatically anymore. To keep things working, we thus copy the return value of
                    // the outer-most function into RAX at the end of the trace.
                    dynasm!(self.asm
                        ; mov rax, Rq(reg)
                    );
                }
            }
            Some(Location::Stack(_offset)) => {}
            Some(Location::NotLive) => {}
            None => {}
        }
        Ok(())
    }

    /// Copy the contents of the local `l2` into  `l1`.
    fn mov_local_local(&mut self, l1: Local, l2: Local) -> Result<(), CompileError> {
        let lreg = self.local_to_reg(l1)?;
        let rreg = self.local_to_reg(l2)?;
        dynasm!(self.asm
            ; mov Rq(lreg), Rq(rreg)
        );
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
        let reg = self.local_to_reg(local)?;
        let c_val = constant.i64_cast();
        dynasm!(self.asm
            ; mov Rq(reg), QWORD c_val
        );
        Ok(())
    }

    /// Move a Boolean into a `Local`.
    fn mov_local_bool(&mut self, local: Local, b: bool) -> Result<(), CompileError> {
        let reg = self.local_to_reg(local)?;
        dynasm!(self.asm
            ; mov Rq(reg), QWORD b as i64
        );
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
        let dest_reg: Option<u8> = if let Some(d) = dest {
            Some(self.local_to_reg(d.local)?)
        } else {
            None
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
                    let idx = stack_index(self.local_to_reg(*local)?);
                    dynasm!(self.asm
                        ; mov Rq(arg_reg), [rsp + idx * 8]
                    );
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
        if let Some(dest_reg) = dest_reg {
            dynasm!(self.asm
                ; mov Rq(dest_reg), rax
            );
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
        dynasm!(self.asm
            ; ret
        );
    }

    /// Finish compilation and return the executable code that was assembled.
    fn finish(self) -> dynasmrt::ExecutableBuffer {
        self.asm.finalize().unwrap()
    }

    /// Compile a TIR trace, returning executable code.
    pub fn compile(tt: TirTrace) -> CompiledTrace {
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

        tc.ret();
        CompiledTrace { mc: tc.finish() }
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
    use super::{CompileError, HashMap, Local, TraceCompiler};
    use libc::{abs, c_void, getuid};
    use std::collections::HashSet;
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
        };

        for _ in 0..32 {
            assert_eq!(
                tc.local_to_reg(Local(1)).unwrap(),
                tc.local_to_reg(Local(1)).unwrap()
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
        };

        let mut seen = HashSet::new();
        for l in 0..7 {
            let reg = tc.local_to_reg(Local(l));
            assert!(!seen.contains(&reg));
            seen.insert(reg);
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
}
