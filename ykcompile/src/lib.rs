//! The Yorick TIR trace compiler.

#![feature(proc_macro_hygiene)]
#![feature(test)]
#![feature(core_intrinsics)]
#![feature(yk)]

#[macro_use]
extern crate dynasmrt;
extern crate test;

mod stack_builder;

use dynasmrt::{x64::Rq::*, Register};
use libc::{c_void, dlsym, RTLD_DEFAULT};
use stack_builder::StackBuilder;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::process::Command;
use yktrace::tir::{
    BinOp, CallOperand, Constant, ConstantInt, Guard, Local, Operand, Place, Projection, Rvalue,
    Statement, TirOp, TirTrace,
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
pub struct CompiledTrace<TT> {
    /// A compiled trace.
    mc: dynasmrt::ExecutableBuffer,
    _pd: PhantomData<TT>,
}

impl<TT> CompiledTrace<TT> {
    /// Execute the trace by calling (not jumping to) the first instruction's address.
    pub fn execute(&self, args: TT) -> TT {
        // For now a compiled trace always returns whatever has been left in register RAX. We also
        // assume for now that this will be a `u64`.
        let func: fn(TT) -> TT =
            unsafe { mem::transmute(self.mc.ptr(dynasmrt::AssemblyOffset(0))) };
        self.exec_trace(func, args)
    }

    /// Actually call the code. This is a separate function making it easier to set a debugger
    /// breakpoint right before entering the trace.
    fn exec_trace(&self, t_fn: fn(TT) -> TT, args: TT) -> TT {
        t_fn(args)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Location {
    Register(u8),
    Stack(i32),
    Arg(i32),
    NotLive,
}

use std::marker::PhantomData;

/// The `TraceCompiler` takes a `SIRTrace` and compiles it to machine code. Returns a `CompiledTrace`.
pub struct TraceCompiler<TT> {
    /// The dynasm assembler which will do all of the heavy lifting of the assembly.
    asm: dynasmrt::x64::Assembler,
    /// Stores the content of each register.
    register_content_map: HashMap<u8, Option<Local>>,
    /// Maps trace locals to their location (register, stack).
    variable_location_map: HashMap<Local, Location>,
    /// Local referencing the input arguments to the trace.
    trace_inputs_local: Option<Local>,
    /// Stack builder for allocating objects on the stack.
    stack_builder: StackBuilder,
    _pd: PhantomData<TT>,
}

impl<TT> TraceCompiler<TT> {
    fn place_to_location(&mut self, p: &Place) -> Result<Location, CompileError> {
        if !p.projection.is_empty() {
            if Some(p.local) == self.trace_inputs_local {
                match &p.projection[0] {
                    Projection::Field(idx) => Ok(Location::Arg((idx * 8) as i32)),
                    Projection::Unimplemented(s) => {
                        Err(CompileError::Unimplemented(format!("{}", s)))
                    }
                }
            } else {
                // TODO deal with remaining projections
                match &p.projection[0] {
                    Projection::Field(0) => self.local_to_location(p.local),
                    _ => Err(CompileError::Unimplemented(format!("{}", p))),
                }
            }
        } else {
            self.local_to_location(p.local)
        }
    }

    /// Given a local, returns the register allocation for it, or, if there is no allocation yet,
    /// performs one.
    fn local_to_location(&mut self, l: Local) -> Result<Location, CompileError> {
        if l == Local(0) {
            // In SIR, `Local` zero is the (implicit) return value, so it makes sense to allocate
            // it to the return register of the underlying X86_64 calling convention.
            Ok(Location::Register(RAX.code()))
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
                    let loc = Location::Stack(self.stack_builder.alloc(8, 8) as i32);
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
        match self.variable_location_map.get(local) {
            Some(Location::Register(reg)) => {
                // If this local is currently stored in a register, free it.
                self.register_content_map.insert(*reg, None);
            }
            Some(Location::Stack(_)) => {}
            Some(Location::Arg(_)) => unreachable!(),
            Some(Location::NotLive) => unreachable!(),
            None => unreachable!(),
        }
        self.variable_location_map.insert(*local, Location::NotLive);
        Ok(())
    }

    /// Copy the contents of the place `p2` into `p1`.
    fn mov_place_place(&mut self, p1: &Place, p2: &Place) -> Result<(), CompileError> {
        let lloc = self.place_to_location(p1)?;
        let rloc = self.place_to_location(p2)?;
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
            (Location::Register(reg), Location::Arg(off)) => {
                dynasm!(self.asm
                    ; mov Rq(reg), [rdi + off]
                );
            }
            (Location::Stack(soff), Location::Arg(aoff)) => {
                dynasm!(self.asm
                    ; mov rax, [rdi + aoff]
                    ; mov [rbp - soff], rax
                );
            }
            (Location::Arg(off), Location::Register(reg)) => {
                dynasm!(self.asm
                    ; mov [rdi + off], Rq(reg)
                );
            }
            (Location::Arg(aoff), Location::Stack(soff)) => {
                dynasm!(self.asm
                    ; mov rax, [rbp - soff]
                    ; mov [rdi + aoff], rax
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

    /// Move a constant integer into a `Place`.
    fn mov_place_constint(
        &mut self,
        place: &Place,
        constant: &ConstantInt,
    ) -> Result<(), CompileError> {
        let loc = self.place_to_location(place)?;
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
            Location::Arg(off) => {
                dynasm!(self.asm
                    ; mov rax, QWORD c_val
                    ; mov [rdi + off], rax
                );
            }
            Location::NotLive => unreachable!(),
        }
        Ok(())
    }

    /// Move a Boolean into a `Place`.
    fn mov_place_bool(&mut self, place: &Place, b: bool) -> Result<(), CompileError> {
        match self.place_to_location(place)? {
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
            Location::Arg(_) => todo!(),
            Location::NotLive => unreachable!(),
        }
        Ok(())
    }

    /// Compile the entry into an inlined function call.
    fn c_enter(
        &mut self,
        op: &CallOperand,
        args: &Vec<Operand>,
        _dest: &Option<Place>,
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
            let arg_idx = Place::from(Local(i + off));
            match op {
                Operand::Place(p) => self.mov_place_place(&arg_idx, p)?,
                Operand::Constant(c) => match c {
                    Constant::Int(ci) => self.mov_place_constint(&arg_idx, ci)?,
                    Constant::Bool(b) => self.mov_place_bool(&arg_idx, *b)?,
                    c => return Err(CompileError::Unimplemented(format!("{}", c))),
                },
            }
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
            Some(self.place_to_location(d)?)
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
        let save_regs = [RDI, RSI, RDX, RCX, R8, R9, R10, R11]
            .iter()
            .map(|r| r.code())
            .filter(|r| Some(*r) != dest_reg)
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

        // Sys-V ABI dictates the first 6 arguments are passed in these registers.
        // The order is reversed so they pop() in the right order.
        let mut arg_regs = vec![R9, R8, RCX, RDX, RSI, RDI]
            .iter()
            .map(|r| r.code())
            .collect::<Vec<u8>>();

        for arg in args {
            // `unwrap()` must succeed, as we checked there are no more than 6 args above.
            let arg_reg = arg_regs.pop().unwrap();

            match arg {
                Operand::Place(place) => {
                    // Load argument back from the stack.
                    match self.place_to_location(place)? {
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
                        Location::Arg(_) => todo!(),
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

        let sym_addr = TraceCompiler::<TT>::find_symbol(sym)? as i64;
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

        // Restore caller-save registers.
        for reg in save_regs.iter().rev() {
            dynasm!(self.asm
                ; pop Rq(reg)
            );
        }

        Ok(())
    }

    fn c_checked_binop(
        &mut self,
        dest: &Place,
        binop: &BinOp,
        op1: &Operand,
        op2: &Operand,
    ) -> Result<(), CompileError> {
        // Move `op1` into `dest`.
        match op1 {
            Operand::Place(p) => self.mov_place_place(dest, &p)?,
            Operand::Constant(Constant::Int(ci)) => self.mov_place_constint(dest, &ci)?,
            Operand::Constant(Constant::Bool(_b)) => unreachable!(),
            Operand::Constant(c) => return Err(CompileError::Unimplemented(format!("{}", c))),
        };
        // Add together `dest` and `op2`.
        let lloc = self.place_to_location(dest)?;
        match op2 {
            Operand::Place(p) => {
                let rloc = self.place_to_location(&p)?;
                match binop {
                    BinOp::Add => self.checked_add_place(lloc, rloc),
                    _ => todo!(),
                }
            }
            Operand::Constant(Constant::Int(ci)) => match binop {
                BinOp::Add => self.checked_add_const(lloc, ci),
                _ => todo!(),
            },
            Operand::Constant(Constant::Bool(_b)) => todo!(),
            Operand::Constant(c) => return Err(CompileError::Unimplemented(format!("{}", c))),
        };
        // In the future this will set the overflow flag of the tuple in `lloc`, which will be
        // checked by a guard, allowing us to return from the trace more gracefully.
        dynasm!(self.asm
            ; jc ->crash
        );
        Ok(())
    }

    fn checked_add_place(&mut self, l1: Location, l2: Location) {
        match (l1, l2) {
            (Location::Register(lreg), Location::Register(rreg)) => {
                dynasm!(self.asm
                    ; add Rq(lreg), Rq(rreg)
                );
            }
            (Location::Register(reg), Location::Stack(off)) => {
                dynasm!(self.asm
                    ; add Rq(reg), [rbp - off]
                );
            }
            (Location::Stack(off), Location::Register(reg)) => {
                dynasm!(self.asm
                    ; add [rbp - off], Rq(reg)
                );
            }
            (Location::Stack(off1), Location::Stack(off2)) => {
                dynasm!(self.asm
                    ; mov rax, [rbp - off2]
                    ; add [rbp - off1], rax
                );
            }
            (Location::Arg(_), _) => {
                // It seems SIR doesn't directly assign binary operations to projections and
                // instead computes the operation in a separate variable first. This means we
                // shouldn't be able to reach this point.
                unreachable!()
            }
            (_, _) => todo!(),
        };
    }

    fn checked_add_const(&mut self, l: Location, c: &ConstantInt) {
        let c_val = c.i64_cast();
        match l {
            Location::Register(reg) => {
                if c_val <= u32::MAX.into() {
                    dynasm!(self.asm
                        ; add Rq(reg), c_val as u32 as i32
                    );
                } else {
                    dynasm!(self.asm
                        ; mov rax, QWORD c_val
                        ; add Rq(reg), rax
                    );
                }
            }
            Location::Stack(off) => {
                if c_val <= u32::MAX.into() {
                    dynasm!(self.asm
                        ; add QWORD [rbp - off], c_val as u32 as i32
                    );
                } else {
                    dynasm!(self.asm
                        ; mov rax, QWORD c_val
                        ; add [rbp - off], rax
                    );
                }
            }
            Location::Arg(_) => {
                // Same explanation as in checked_add_place.
                unreachable!()
            }
            _ => todo!(),
        }
    }

    /// Compile a TIR statement.
    fn statement(&mut self, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::Assign(l, r) => {
                match r {
                    Rvalue::Use(Operand::Place(p)) => {
                        self.mov_place_place(l, p)?;
                    }
                    Rvalue::Use(Operand::Constant(c)) => match c {
                        Constant::Int(ci) => self.mov_place_constint(l, ci)?,
                        Constant::Bool(b) => self.mov_place_bool(l, *b)?,
                        c => return Err(CompileError::Unimplemented(format!("{}", c))),
                    },
                    Rvalue::CheckedBinaryOp(binop, op1, op2) => {
                        self.c_checked_binop(l, binop, op1, op2)?
                    }
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
        eprintln!("\nRegister allocation (place -> reg):");
        for (place, location) in &self.variable_location_map {
            eprintln!(
                "  {:2} -> {:?} ({})",
                place,
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
        let soff = self.stack_builder.size();
        dynasm!(self.asm
            ; add rsp, soff as i32
            ; pop rbp
            ; ret
            ; ->reserve:
            ; push rbp
            ; mov rbp, rsp
            ; sub rsp, soff as i32
            ; jmp ->main
        );
    }

    fn init(&mut self) {
        // Jump to the label that reserves stack space for spilled locals.
        dynasm!(self.asm
            ; jmp ->reserve
            ; ->crash:
            ; ud2
            ; ->main:
        );
    }

    /// Finish compilation and return the executable code that was assembled.
    fn finish(self) -> dynasmrt::ExecutableBuffer {
        self.asm.finalize().unwrap()
    }

    #[cfg(test)]
    fn test_compile(tt: TirTrace) -> (CompiledTrace<TT>, u32) {
        // Changing the registers available to the register allocator affects the number of spills,
        // and thus also some tests. To make sure we notice when this happens we also check the
        // number of spills in those tests. We thus need a slightly different version of the
        // `compile` function that provides this information to the test.
        let tc = TraceCompiler::<TT>::_compile(tt);
        let spills = tc.stack_builder.size();
        let ct = CompiledTrace::<TT> {
            mc: tc.finish(),
            _pd: PhantomData,
        };
        (ct, spills)
    }

    /// Compile a TIR trace, returning executable code.
    pub fn compile(tt: TirTrace) -> CompiledTrace<TT> {
        let tc = TraceCompiler::<TT>::_compile(tt);
        CompiledTrace::<TT> {
            mc: tc.finish(),
            _pd: PhantomData,
        }
    }

    fn _compile(tt: TirTrace) -> Self {
        let assembler = dynasmrt::x64::Assembler::new().unwrap();

        let mut tc = TraceCompiler::<TT> {
            asm: assembler,
            // Use all the 64-bit registers we can (R11-R8, RDX, RCX). We probably also want to use the
            // callee-saved registers R15-R12 here in the future.
            register_content_map: [R11, R10, R9, R8, RDX, RCX]
                .iter()
                .cloned()
                .map(|r| (r.code(), None))
                .collect(),
            variable_location_map: HashMap::new(),
            trace_inputs_local: tt.inputs().map(|t| t.clone()),
            stack_builder: StackBuilder::default(),
            _pd: PhantomData,
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
    use crate::stack_builder::StackBuilder;
    use core::yk::trace_inputs;
    use dynasmrt::{x64::Rq::*, Register};
    use fm::FMBuilder;
    use libc::{abs, c_void, getuid};
    use regex::Regex;
    use std::marker::PhantomData;
    use yktrace::tir::TirTrace;
    use yktrace::{start_tracing, TracingKind};

    extern "C" {
        fn add6(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) -> u64;
    }

    /// Fuzzy matches the textual TIR for the trace `tt` with the pattern `ptn`.
    fn assert_tir(ptn: &str, tt: &TirTrace) {
        let ptn_re = Regex::new(r"%.+?\b").unwrap(); // Names are words prefixed with `%`.
        let text_re = Regex::new(r"\$?.+?\b").unwrap(); // Any word optionally prefixed with `$`.
        let matcher = FMBuilder::new(ptn)
            .unwrap()
            .name_matcher(Some((ptn_re, text_re)))
            .distinct_name_matching(true)
            .build()
            .unwrap();

        let res = matcher.matches(&format!("{}", tt));
        if let Err(e) = res {
            eprintln!("{}", e); // Visible when tests run with --nocapture.
            panic!(e);
        }
    }

    #[inline(never)]
    fn simple() -> u8 {
        let x = 13;
        x
    }

    #[test]
    fn test_simple() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = simple();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::<&(u64,)>::compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 13);
    }

    // Repeatedly fetching the register for the same local should yield the same register and
    // should not exhaust the allocator.
    #[test]
    fn reg_alloc_same_local() {
        let mut tc = TraceCompiler::<u8> {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: [R15, R14, R13, R12, R11, R10, R9, R8, RDX, RCX]
                .iter()
                .cloned()
                .map(|r| (r.code(), None))
                .collect(),
            variable_location_map: HashMap::new(),
            trace_inputs_local: None,
            stack_builder: StackBuilder::default(),
            _pd: PhantomData,
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
        let mut tc = TraceCompiler::<u8> {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: [R15, R14, R13, R12, R11, R10, R9, R8, RDX, RCX]
                .iter()
                .cloned()
                .map(|r| (r.code(), None))
                .collect(),
            variable_location_map: HashMap::new(),
            trace_inputs_local: None,
            stack_builder: StackBuilder::default(),
            _pd: PhantomData,
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
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = fcall();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::<&(u64,)>::compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 13);
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
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = fnested();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::<&(u64,)>::compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 20);
    }

    // Test finding a symbol in a shared object.
    #[test]
    fn find_symbol_shared() {
        assert!(TraceCompiler::<u8>::find_symbol("printf") == Ok(libc::printf as *mut c_void));
    }

    // Test finding a symbol in the main binary.
    // For this to work the binary must have been linked with `--export-dynamic`, which ykrustc
    // appends to the linker command line.
    #[test]
    #[no_mangle]
    fn find_symbol_main() {
        assert!(
            TraceCompiler::<u8>::find_symbol("find_symbol_main")
                == Ok(find_symbol_main as *mut c_void)
        );
    }

    // Check that a non-existent symbol cannot be found.
    #[test]
    fn find_nonexistent_symbol() {
        assert_eq!(
            TraceCompiler::<u8>::find_symbol("__xxxyyyzzz__"),
            Err(CompileError::UnknownSymbol("__xxxyyyzzz__".to_owned()))
        );
    }

    // A trace which contains a call to something which we don't have SIR for should emit a TIR
    // call operation.
    #[test]
    fn call_symbol_tir() {
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let _ = unsafe { add6(1, 1, 1, 1, 1, 1) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        assert_tir(
            "...\n\
            ops:\n\
              live(%a)\n\
              %a = call(add6, [1u64, 1u64, 1u64, 1u64, 1u64, 1u64])\n\
              dead(%a)",
            &tir_trace,
        );
    }

    /// Execute a trace which calls a symbol accepting no arguments, but which does return a value.
    #[test]
    fn exec_call_symbol_no_args() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = unsafe { getuid() };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let mut args = (0,);
        TraceCompiler::<&(u64,)>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0 as u64, args.0);
    }

    /// Execute a trace which calls a symbol accepting arguments and returns a value.
    #[test]
    fn exec_call_symbol_with_arg() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let v = -56;
        inputs.0 = unsafe { abs(v) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let mut args = (0,);
        TraceCompiler::<&(u64,)>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0 as u64, args.0);
    }

    /// The same as `exec_call_symbol_args_with_rv`, just using a constant argument.
    #[test]
    fn exec_call_symbol_with_const_arg() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = unsafe { abs(-123) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let mut args = (0,);
        TraceCompiler::<&(u64,)>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0 as u64, args.0);
    }

    #[test]
    fn exec_call_symbol_with_many_args() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = unsafe { add6(1, 2, 3, 4, 5, 6) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let mut args = (0,);
        TraceCompiler::<&(u64,)>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, 21);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn exec_call_symbol_with_many_args_some_ignored() {
        extern "C" {
            fn add_some(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
        }

        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = unsafe { add_some(1, 2, 3, 4, 5) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let mut args = (0,);
        TraceCompiler::<&(u64,)>::compile(tir_trace).execute(&mut args);
        assert_eq!(args.0, 7);
        assert_eq!(args.0, inputs.0);
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
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = many_locals();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<&(u64,)>::test_compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 7);
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
        let _g = 7;
        let h: u64 = u64value();
        h
    }

    #[test]
    fn test_spilling_u64() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = spill_u64();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<&(u64,)>::test_compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 4294967296 + 8);
        assert_eq!(spills, 2 * 8);
    }

    fn register_to_stack(arg: u8) -> u8 {
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _f = 6;
        let _g = 7;
        let h = arg;
        h
    }

    #[test]
    fn test_mov_register_to_stack() {
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = register_to_stack(8);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<&(u64,)>::test_compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 8);
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
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = stack_to_register();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<&(u64,)>::test_compile(tir_trace);
        let mut args = (0,);
        ct.execute(&mut args);
        assert_eq!(args.0, 3);
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
        let mut inputs = trace_inputs((0,));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = ext_call();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let mut args = (0,);
        TraceCompiler::<&(u64,)>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, 7);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn test_trace_inputs() {
        let mut inputs = trace_inputs((1, 2, 3));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = unsafe { add6(inputs.0, inputs.1, inputs.2, 4, 5, 6) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::<&(u64, u64, u64)>::compile(tir_trace);
        let mut args = (1, 2, 3);
        ct.execute(&mut args);
        assert_eq!(inputs.0, 21);
        assert_eq!(inputs.0, args.0);
        // Execute once more with different arguments.
        let mut args2 = (7, 8, 9);
        ct.execute(&mut args2);
        assert_eq!(args2.0, 39);
    }

    #[inline(never)]
    fn add(a: u8) -> u8 {
        let x = a + 3;
        let y = a + x;
        y
    }

    fn add64(a: u64) -> u64 {
        let x = a + 8589934592;
        x
    }

    #[test]
    fn test_binop_add() {
        let mut inputs = trace_inputs((0, 0, 0, 0));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        inputs.0 = add(13);
        inputs.1 = add64(1);
        inputs.2 = inputs.0 + 2;
        inputs.3 = inputs.0 + inputs.0;
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::<&(u64, u64, u64, u64)>::compile(tir_trace);
        let mut args = (0, 0, 0, 0);
        ct.execute(&mut args);
        assert_eq!(args.0, 29);
        assert_eq!(args.1, 8589934593);
        assert_eq!(args.2, 31);
        assert_eq!(args.3, 58);
    }

    // Similar test to the above, but makes sure the operations will be executed on the stack by
    // filling up all registers first.
    #[test]
    fn test_binop_add_stack() {
        let mut inputs = trace_inputs((0, 0));
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _d = 6;
        inputs.0 = add(13);
        inputs.1 = add64(1);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();
        let ct = TraceCompiler::<&(u64, u64)>::compile(tir_trace);
        let mut args = (0, 0);
        ct.execute(&mut args);
        assert_eq!(args.0, 29);
        assert_eq!(args.1, 8589934593);
    }

    #[test]
    fn field_projection() {
        struct S {
            _x: u64,
            y: u64,
        }

        fn get_y(s: S) -> u64 {
            s.y
        }

        let _ = trace_inputs(());
        let th = start_tracing(Some(TracingKind::HardwareTracing));
        let s = S { _x: 100, y: 200 };
        let _expect = get_y(s);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*sir_trace).unwrap();

        // %s1: Initial s in the outer function
        // %s2: A copy of s. Uninteresting.
        // %s3: s inside the function.
        // %res: the result of the call.
        assert_tir("
            local_decls:
              ...
              %s1: (%crate, %tid1) => StructTy { offsets: [0, 8], tys: [(%crate, %tid2), (%crate, %tid2)], align: 8, size: 16 }
              ...
              %res: (%crate, %tid2) => u64
              ...
              %s2: (%crate, %tid1)...
              ...
              %s3: (%crate, %tid1)...
              ...
            ops:
              ...
              (%s1).0 = 100u64
              (%s1).1 = 200u64
              ...
              %s2 = %s1
              enter(...
              %res = (%s3).1
              leave
              ...", &tir_trace);
    }
}
