//! The Yorick TIR trace compiler.

#![feature(proc_macro_hygiene)]
#![feature(test)]
#![feature(core_intrinsics)]

#[macro_use]
extern crate dynasmrt;
#[macro_use]
extern crate lazy_static;
extern crate test;

mod stack_builder;

use dynasmrt::{x64::Rq::*, Register};
use libc::{c_void, dlsym, RTLD_DEFAULT};
use stack_builder::StackBuilder;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::ffi::CString;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::process::Command;
use ykpack::{IPlace, OffT, SignedIntTy, Ty, TypeId, UnsignedIntTy};
use yktrace::tir::{
    BinOp, CallOperand, Constant, Guard, GuardKind, Local, Statement, TirOp, TirTrace,
};
use yktrace::{sir::SIR, INTERP_STEP_ARG};

use dynasmrt::{DynasmApi, DynasmLabelApi};

lazy_static! {
    // Registers that are caller-save as per the Sys-V ABI.
    static ref CALLER_SAVE_REGS: [u8; 8] = [RDI.code(), RSI.code(), RDX.code(), RCX.code(),
                                            R8.code(), R9.code(), R10.code(), R11.code()];

    // The register partitioning. These arrays must not overlap.
    // FIXME add callee save registers to the pool. Trace code will need to save/restore them.
    static ref TEMP_REG: u8 = R11.code();
    static ref REG_POOL: [u8; 5] = [R10.code(), R9.code(), R8.code(), RDX.code(), RCX.code()];

    static ref TEMP_LOC: Location = Location::Register(*TEMP_REG);
    static ref PTR_SIZE: u64 = u64::try_from(mem::size_of::<usize>()).unwrap();
}

macro_rules! binop_add_sub {
    ($name: ident, $op:expr) => {
        fn $name(&mut self, dest: &IPlace, size: u64, opnd2: &IPlace, temp_reg: u8) {
            let src_loc = self.iplace_to_location(opnd2);
            match src_loc {
                Location::Register(r) => match size {
                    1 => {
                        dynasm!(self.asm
                            ; $op Rb(temp_reg), Rb(r)
                        );
                    }
                    2 => {
                        dynasm!(self.asm
                            ; $op Rw(temp_reg), Rw(r)
                        );
                    }
                    4 => {
                        dynasm!(self.asm
                            ; $op Rd(temp_reg), Rd(r)
                        );
                    }
                    8 => {
                        dynasm!(self.asm
                            ; $op Rq(temp_reg), Rq(r)
                        );
                    }
                    _ => unreachable!(format!("{}", SIR.ty(&dest.ty()))),
                },
                Location::Mem(..) => todo!(),
                Location::Const { val, .. } => {
                    let val = val.i64_cast();
                    match size {
                        1 => {
                            dynasm!(self.asm
                                ; $op Rb(temp_reg), val as i8
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; $op Rw(temp_reg), val as i16
                            );
                        }
                        4 => {
                            dynasm!(self.asm
                                ; $op Rd(temp_reg), val as i32
                            );
                        }
                        8 => {
                            if i32::try_from(val).is_err() {
                                // FIXME Work around x86_64 encoding limitations (no imm64 operands).
                                todo!();
                            } else {
                                dynasm!(self.asm
                                    ; $op Rq(temp_reg), val as i32
                                );
                            }
                        }
                        _ => unreachable!(format!("{}", SIR.ty(&dest.ty()))),
                    }
                }
                Location::Indirect { .. } => todo!(),
                Location::NotLive => todo!(),
            }
        }
    }
}

macro_rules! binop_mul_div {
    ($name: ident, $op:expr) => {
        fn $name(&mut self, dest: &IPlace, size: u64, opnd2: &IPlace, temp_reg: u8) {
            // mul and div overwrite RAX, RDX, so save them first.
            dynasm!(self.asm
                ; push rax
                ; push rdx
                ; xor rdx, rdx
            );
            dynasm!(self.asm
                ; mov rax, Rq(temp_reg)
            );
            let src_loc = self.iplace_to_location(opnd2);
            match src_loc {
                Location::Register(r) => match size {
                    1 => {
                        dynasm!(self.asm
                            ; $op Rb(r)
                        );
                    }
                    2 => {
                        dynasm!(self.asm
                            ; $op Rw(r)
                        );
                    }
                    4 => {
                        dynasm!(self.asm
                            ; $op Rd(r)
                        );
                    }
                    8 => {
                        dynasm!(self.asm
                            ; $op Rq(r)
                        );
                    }
                    _ => unreachable!(format!("{}", SIR.ty(&dest.ty()))),
                },
                Location::Mem(..) => todo!(),
                Location::Const { val, .. } => {
                    let val = val.i64_cast();
                    match size {
                        1 => {
                            dynasm!(self.asm
                                ; mov Rb(temp_reg), val as i8
                                ; $op Rb(temp_reg)
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; mov Rw(temp_reg), val as i16
                                ; $op Rw(temp_reg)
                            );
                        }
                        4 => {
                            dynasm!(self.asm
                                ; mov Rd(temp_reg), val as i32
                                ; $op Rd(temp_reg)
                            );
                        }
                        8 => {
                            if i32::try_from(val).is_err() {
                                // FIXME Work around x86_64 encoding limitations (no imm64 operands).
                                todo!();
                            } else {
                                dynasm!(self.asm
                                    ; mov Rq(temp_reg), val as i32
                                    ; $op Rq(temp_reg)
                                );
                            }
                        }
                        _ => unreachable!(format!("{}", SIR.ty(&dest.ty()))),
                    }
                }
                Location::Indirect { .. } => todo!(),
                Location::NotLive => todo!(),
            }

            // Restore RAX, RDX
            dynasm!(self.asm
                ; mov Rq(*TEMP_REG), rax
                ; pop rax
                ; pop rdx
            );
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum CompileError {
    /// The binary symbol could not be found.
    UnknownSymbol(String),
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
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
    pub fn execute(&self, args: &mut TT) -> bool {
        let func: fn(&mut TT) -> bool =
            unsafe { mem::transmute(self.mc.ptr(dynasmrt::AssemblyOffset(0))) };
        self.exec_trace(func, args)
    }

    /// Actually call the code. This is a separate function making it easier to set a debugger
    /// breakpoint right before entering the trace.
    fn exec_trace(&self, t_fn: fn(&mut TT) -> bool, args: &mut TT) -> bool {
        t_fn(args)
    }
}

/// Represents a memory location using a register and an offset.
#[derive(Debug, Clone, PartialEq)]
pub struct RegAndOffset {
    reg: u8,
    off: OffT,
}

/// Describes the location of the pointer in Location::Indirect.
#[derive(Debug, Clone, PartialEq)]
pub enum IndirectLoc {
    /// There's a pointer in this register.
    Register(u8),
    /// There's a pointer in memory somewhere.
    Mem(RegAndOffset),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    /// A value in a register.
    Register(u8),
    /// A statically known memory location relative to a register.
    Mem(RegAndOffset),
    /// A location that contains a pointer to some underlying storage.
    Indirect { ptr: IndirectLoc, off: OffT },
    /// A statically known constant.
    Const { val: Constant, ty: TypeId },
    /// A non-live location. Used by the register allocator.
    NotLive,
}

impl Location {
    /// Creates a new memory location from a register and an offset.
    fn new_mem(reg: u8, off: OffT) -> Self {
        Self::Mem(RegAndOffset { reg, off })
    }

    #[cfg(test)]
    /// If `self` is a `Mem` then unwrap it, otherwise panic.
    fn unwrap_mem(&self) -> &RegAndOffset {
        if let Location::Mem(ro) = self {
            ro
        } else {
            panic!("tried to unwrap a Mem location when it wasn't a Mem");
        }
    }

    /// If `self` is a `Mem` then return a mutable reference to its innards, otherwise panic.
    fn unwrap_mem_mut(&mut self) -> &mut RegAndOffset {
        if let Location::Mem(ro) = self {
            ro
        } else {
            panic!("tried to unwrap a Mem location when it wasn't a Mem");
        }
    }

    /// Returns which register (if any) is used in addressing this location.
    fn uses_reg(&self) -> Option<u8> {
        match self {
            Location::Register(reg) => Some(*reg),
            Location::Mem(RegAndOffset { reg, .. }) => Some(*reg),
            Location::Indirect {
                ptr: IndirectLoc::Register(reg),
                ..
            }
            | Location::Indirect {
                ptr: IndirectLoc::Mem(RegAndOffset { reg, .. }),
                ..
            } => Some(*reg),
            Location::Const { .. } => None,
            Location::NotLive => unreachable!(),
        }
    }

    /// Apply an offset to the location, returning a new one.
    fn offset(self, off: OffT) -> Self {
        if off == 0 {
            return self;
        }
        match self {
            Location::Mem(ro) => Location::Mem(RegAndOffset {
                reg: ro.reg,
                off: ro.off + off,
            }),
            Location::Indirect { ptr, off: ind_off } => Location::Indirect {
                ptr,
                off: ind_off + off,
            },
            Location::Register(..) | Location::Const { .. } => todo!("offsetting a constant"),
            Location::NotLive => unreachable!(),
        }
    }

    /// Converts a direct place to an indirect place for use as a pointer.
    fn to_indirect(&self) -> Self {
        let ptr = match self {
            Location::Register(r) => IndirectLoc::Register(*r),
            Location::Mem(ro) => IndirectLoc::Mem(ro.clone()),
            _ => unreachable!(),
        };
        Location::Indirect { ptr, off: 0 }
    }
}

/// Allocation of one of the REG_POOL. Temporary registers are tracked separately.
enum RegAlloc {
    Local(Local),
    Free,
}

use std::marker::PhantomData;
use ykpack::LocalDecl;

/// The `TraceCompiler` takes a `SIRTrace` and compiles it to machine code. Returns a `CompiledTrace`.
pub struct TraceCompiler<TT> {
    /// The dynasm assembler which will do all of the heavy lifting of the assembly.
    asm: dynasmrt::x64::Assembler,
    /// Stores the content of each register.
    register_content_map: HashMap<u8, RegAlloc>,
    /// Maps trace locals to their location (register, stack).
    variable_location_map: HashMap<Local, Location>,
    /// Local decls of the tir trace.
    local_decls: HashMap<Local, LocalDecl>,
    /// Stack builder for allocating objects on the stack.
    stack_builder: StackBuilder,
    /// Stores the memory addresses of local functions.
    addr_map: HashMap<String, u64>,
    _pd: PhantomData<TT>,
}

impl<TT> TraceCompiler<TT> {
    fn can_live_in_register(decl: &LocalDecl) -> bool {
        if decl.referenced {
            // We must allocate it on the stack so that we can reference it.
            return false;
        }

        // FIXME: optimisation: small structs and tuples etc. could actually live in a register.
        let ty = SIR.ty(&decl.ty);
        match ty {
            Ty::UnsignedInt(ui) => !matches!(ui, UnsignedIntTy::U128),
            Ty::SignedInt(si) => !matches!(si, SignedIntTy::I128),
            Ty::Array { .. } => false,
            Ty::Slice(_) => false,
            Ty::Ref(_) | Ty::Bool | Ty::Char => true,
            Ty::Struct(..) | Ty::Tuple(..) => false,
            Ty::Unimplemented(..) => todo!("{}", ty),
        }
    }

    fn iplace_to_location(&mut self, ip: &IPlace) -> Location {
        match ip {
            IPlace::Val { local, off, .. } => self.local_to_location(*local).offset(*off),
            IPlace::Indirect { ptr, off, .. } => self
                .local_to_location(ptr.local)
                .offset(ptr.off)
                .to_indirect()
                .offset(*off),
            IPlace::Const { val, ty } => Location::Const {
                val: val.clone(),
                ty: *ty,
            },
            _ => todo!(),
        }
    }

    /// Given a local, returns the register allocation for it, or, if there is no allocation yet,
    /// performs one.
    fn local_to_location(&mut self, l: Local) -> Location {
        if l == INTERP_STEP_ARG {
            // The argument is a mutable reference in RDI.
            Location::Register(RDI.code())
        } else if let Some(location) = self.variable_location_map.get(&l) {
            // We already have a location for this local.
            location.clone()
        } else {
            let decl = &self.local_decls[&l];
            if Self::can_live_in_register(&decl) {
                // Find a free register to store this local.
                let loc = if let Some(reg) = self.get_free_register() {
                    self.register_content_map.insert(reg, RegAlloc::Local(l));
                    Location::Register(reg)
                } else {
                    // All registers are occupied, so we need to spill the local to the stack.
                    self.spill_local_to_stack(&l)
                };
                let ret = loc.clone();
                self.variable_location_map.insert(l, loc);
                ret
            } else {
                let ty = SIR.ty(&decl.ty);
                let loc = self.stack_builder.alloc(ty.size(), ty.align());
                self.variable_location_map.insert(l, loc.clone());
                loc
            }
        }
    }

    /// Returns a free register or `None` if all registers are occupied.
    fn get_free_register(&self) -> Option<u8> {
        self.register_content_map.iter().find_map(|(k, v)| match v {
            RegAlloc::Free => Some(*k),
            _ => None,
        })
    }

    /// Spill a local to the stack and return its location. Note: This does not update the
    /// `variable_location_map`.
    fn spill_local_to_stack(&mut self, local: &Local) -> Location {
        let tyid = self.local_decls[&local].ty;
        let ty = SIR.ty(&tyid);
        self.stack_builder.alloc(ty.size(), ty.align())
    }

    /// Notifies the register allocator that the register allocated to `local` may now be re-used.
    fn free_register(&mut self, local: &Local) -> Result<(), CompileError> {
        match self.variable_location_map.get(local) {
            Some(Location::Register(reg)) => {
                // If this local is currently stored in a register, free it.
                self.register_content_map.insert(*reg, RegAlloc::Free);
            }
            Some(Location::Mem { .. }) | Some(Location::Indirect { .. }) => {}
            Some(Location::NotLive) => unreachable!(),
            Some(Location::Const { .. }) => unreachable!(),
            None => {
                unreachable!("freeing unallocated register");
            }
        }
        self.variable_location_map.insert(*local, Location::NotLive);
        Ok(())
    }

    /// Copy bytes from one memory location to another.
    fn copy_memory(&mut self, dest: &RegAndOffset, src: &RegAndOffset, size: u64) {
        // We use memmove(3), as it's not clear if MIR (and therefore SIR) could cause copies
        // involving overlapping buffers.
        let sym = Self::find_symbol("memmove").unwrap();
        self.caller_save();
        dynasm!(self.asm
            ; push rax
            ; xor rax, rax
            ; lea rdi, [Rq(dest.reg) + dest.off]
            ; lea rsi, [Rq(src.reg) + src.off]
            ; mov rdx, size as i32
            ; mov r11, QWORD sym as i64
            ; call r11
            ; pop rax
        );
        self.caller_save_restore();
    }

    /// Emit a NOP operation.
    fn _nop(&mut self) {
        dynasm!(self.asm
            ; nop
        );
    }

    /// Push all of the caller-save registers to the stack.
    fn caller_save(&mut self) {
        for reg in CALLER_SAVE_REGS.iter() {
            dynasm!(self.asm
                ; push Rq(reg)
            );
        }
    }

    /// Restore caller-save registers from the stack.
    fn caller_save_restore(&mut self) {
        for reg in CALLER_SAVE_REGS.iter().rev() {
            dynasm!(self.asm
                ; pop Rq(reg)
            );
        }
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
    ///
    ///  - RAX is clobbered.
    fn c_call(
        &mut self,
        opnd: &CallOperand,
        args: &[IPlace],
        dest: &Option<IPlace>,
    ) -> Result<(), CompileError> {
        let sym = if let CallOperand::Fn(sym) = opnd {
            sym
        } else {
            todo!("unknown call target");
        };

        if args.len() > 6 {
            todo!("call with spilled args");
        }

        // Save Sys-V caller save registers to the stack, but skip the one (if there is one) that
        // will store the return value. It's safe to assume the caller expects this to be
        // clobbered.
        //
        // FIXME: Note that we don't save RAX. Although this is a caller save register, we are
        // currently using RAX as a general purpose register in parts of the compiler (the register
        // allocator thus never gives out RAX). In this case we use it to store the result from the
        // call in its destination, so we must not override it when returning from the call.
        self.caller_save();

        // Helper function to find the index of a caller-save register previously pushed to the stack.
        // The first register pushed is at the highest stack offset (from the stack pointer), hence
        // reversing the order of `save_regs`.
        let stack_index = |reg: u8| -> i32 {
            i32::try_from(
                CALLER_SAVE_REGS
                    .iter()
                    .rev()
                    .position(|&r| r == reg)
                    .unwrap(),
            )
            .unwrap()
        };

        // Sys-V ABI dictates the first 6 arguments are passed in these registers.
        // The order is reversed so they pop() in the right order.
        let mut arg_regs = vec![R9, R8, RCX, RDX, RSI, RDI]
            .iter()
            .map(|r| r.code())
            .collect::<Vec<u8>>();

        for arg in args {
            // In which register will this argument be passed?
            // `unwrap()` must succeed, as we checked there are no more than 6 args above.
            let arg_reg = arg_regs.pop().unwrap();

            // Now load the argument into the correct argument register.
            match self.iplace_to_location(arg) {
                Location::Register(reg) => {
                    // The value *was* in a register before we pushed it with caller_save().
                    // We load it back from the stack now.
                    //
                    // FIXME The following code assumes that arguments will all have previously
                    // been in caller save registers and that we will need to load them back off
                    // the stack. In reality any given argument may not have been in a caller save
                    // register in the first place. stack_index() will panic if this is the case.
                    let off = stack_index(reg) * 8;
                    dynasm!(self.asm
                        ; mov Rq(arg_reg), [rsp + off]
                    );
                }
                Location::Mem(ro) => dynasm!(self.asm
                    ; mov Rq(arg_reg), [Rq(ro.reg) + ro.off]
                ),
                Location::Indirect { .. } => todo!(),
                Location::Const { val, .. } => {
                    // FIXME assumes constant fits in a register.
                    dynasm!(self.asm
                        ; mov Rq(arg_reg), QWORD val.i64_cast()
                    );
                }
                Location::NotLive => unreachable!(),
            }
        }

        let sym_addr = if let Some(addr) = self.addr_map.get(sym) {
            *addr as i64
        } else {
            TraceCompiler::<TT>::find_symbol(sym)? as i64
        };
        dynasm!(self.asm
            // In Sys-V ABI, `al` is a hidden argument used to specify the number of vector args
            // for a vararg call. We don't support this right now, so set it to zero.
            ; xor rax, rax
            ; mov r11, QWORD sym_addr
            ; call r11
        );

        // Restore caller-save registers.
        self.caller_save_restore();

        if let Some(d) = dest {
            let dest_loc = self.iplace_to_location(d);
            self.store_raw(
                &dest_loc,
                &Location::Register(RAX.code()),
                SIR.ty(&d.ty()).size(),
            );
        }

        Ok(())
    }

    /// Load an IPlace into the given register. Panic if it doesn't fit.
    fn load_reg_iplace(&mut self, reg: u8, src_ip: &IPlace) -> Location {
        let dest_loc = Location::Register(reg);
        let src_loc = self.iplace_to_location(src_ip);
        self.store_raw(&dest_loc, &src_loc, SIR.ty(&src_ip.ty()).size());
        dest_loc
    }

    fn c_binop(&mut self, dest: &IPlace, op: BinOp, opnd1: &IPlace, opnd2: &IPlace, checked: bool) {
        match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                return self.c_condition(dest, &op, opnd1, opnd2);
            }
            _ => {}
        }

        // We do this in three stages.
        // 1) Copy the first operand into the temp register.
        self.load_reg_iplace(*TEMP_REG, opnd1);

        // 2) Apply the second operand.
        let size = SIR.ty(&opnd1.ty()).size();
        match op {
            BinOp::Add => self.c_binop_add(dest, size, opnd2, *TEMP_REG),
            BinOp::Sub => self.c_binop_sub(dest, size, opnd2, *TEMP_REG),
            BinOp::Mul => self.c_binop_mul(dest, size, opnd2, *TEMP_REG),
            BinOp::Div => self.c_binop_div(dest, size, opnd2, *TEMP_REG),
            _ => todo!(),
        }

        // FIXME. Check for overflow here. Examine either CF or OF depending on signedness.

        // 3) Move the result to where it is supposed to live.
        // If it is a checked operation, then we have to build a (value, overflow-flag) tuple.
        let mut dest_loc = self.iplace_to_location(dest);
        self.store_raw(&dest_loc, &*TEMP_LOC, size);
        if checked {
            // Set overflow flag.
            dynasm!(self.asm
                ; mov Rq(*TEMP_REG), 0
            );
            let ro = dest_loc.unwrap_mem_mut();
            ro.off += i32::try_from(size).unwrap();
            self.store_raw(&dest_loc, &*TEMP_LOC, 1);
        }
    }

    binop_add_sub!(c_binop_add, add);
    binop_add_sub!(c_binop_sub, sub);
    binop_mul_div!(c_binop_mul, mul);
    binop_mul_div!(c_binop_div, div);

    fn c_condition(&mut self, dest: &IPlace, binop: &BinOp, op1: &IPlace, op2: &IPlace) {
        let src1 = self.iplace_to_location(op1);
        let ty = SIR.ty(&op1.ty());

        self.load_reg_iplace(*TEMP_REG, op2);

        match &src1 {
            Location::Register(reg) => match ty.size() {
                1 => {
                    dynasm!(self.asm
                        ; cmp Rb(reg), Rb(*TEMP_REG)
                    );
                }
                2 => {
                    dynasm!(self.asm
                        ; cmp Rw(reg), Rw(*TEMP_REG)
                    );
                }
                4 => {
                    dynasm!(self.asm
                        ; cmp Rd(reg), Rd(*TEMP_REG)
                    );
                }
                8 => {
                    dynasm!(self.asm
                        ; cmp Rq(reg), Rq(*TEMP_REG)
                    );
                }
                _ => unreachable!(),
            },
            Location::Mem(ro) => match ty.size() {
                1 => {
                    dynasm!(self.asm
                        ; cmp BYTE [Rq(ro.reg) + ro.off], Rb(*TEMP_REG)
                    );
                }
                2 => {
                    dynasm!(self.asm
                        ; cmp WORD [Rq(ro.reg) + ro.off], Rw(*TEMP_REG)
                    );
                }
                4 => {
                    dynasm!(self.asm
                        ; cmp DWORD [Rq(ro.reg) + ro.off], Rd(*TEMP_REG)
                    );
                }
                8 => {
                    dynasm!(self.asm
                        ; cmp QWORD [Rq(ro.reg) + ro.off], Rq(*TEMP_REG)
                    );
                }
                _ => unreachable!(),
            },
            _ => todo!(),
        }
        dynasm!(self.asm
         ; mov Rq(*TEMP_REG), 1
        );
        match binop {
            BinOp::Eq => {
                dynasm!(self.asm
                    ; je >skip
                );
            }
            BinOp::Ne => {
                dynasm!(self.asm
                    ; jne >skip
                );
            }
            BinOp::Lt => {
                dynasm!(self.asm
                    ; jl >skip
                );
            }
            BinOp::Le => {
                dynasm!(self.asm
                    ; jle >skip
                );
            }
            BinOp::Gt => {
                dynasm!(self.asm
                    ; jg >skip
                );
            }
            BinOp::Ge => {
                dynasm!(self.asm
                    ; jge >skip
                );
            }
            _ => unreachable!(),
        }
        dynasm!(self.asm
         ; mov Rq(*TEMP_REG), 0
         ; skip:
        );
        let dest_loc = self.iplace_to_location(dest);
        self.store_raw(&dest_loc, &*TEMP_LOC, SIR.ty(&dest.ty()).size());
    }

    fn c_dynoffs(&mut self, dest: &IPlace, base: &IPlace, idx: &IPlace, scale: u32) {
        // FIXME possible optimisation, use LEA if scale fits in a u8.

        // MUL clobbers RDX:RAX, so store/restore those.
        // FIXME only do this if these registers are allocated?
        dynasm!(self.asm
            ; push rax
            ; push rdx
        );

        // 1) Multiply scale by idx, store in RAX.
        self.load_reg_iplace(RAX.code(), idx);
        dynasm!(self.asm
            ; mov Rq(*TEMP_REG), i32::try_from(scale).unwrap()
            ; mul Rq(*TEMP_REG)
            ; jo ->crash
        );

        // 2) Get the address of the thing we want to offset into a register.
        let base_loc = self.iplace_to_location(base);
        match base_loc {
            Location::Register(..) => todo!(),
            Location::Mem(..) => todo!(),
            Location::Indirect { ptr, off } => match ptr {
                IndirectLoc::Register(..) => todo!(),
                IndirectLoc::Mem(ind_ro) => {
                    dynasm!(self.asm
                        ; mov Rq(*TEMP_REG), [Rq(ind_ro.reg) + ind_ro.off]
                        ; add Rq(*TEMP_REG), off
                        ; jo ->crash
                    );
                }
            },
            Location::Const { .. } => todo!(),
            Location::NotLive => unreachable!(),
        }

        // 3) Apply the offset.
        dynasm!(self.asm
            ; add Rq(*TEMP_REG), rax
            ; jo ->crash
        );

        // Restore what we saved earlier.
        dynasm!(self.asm
            ; pop rdx
            ; pop rax
        );

        // 4) Store the resulting pointer into the destination.
        // The IR is constructed such that `dest_loc` will be indirect to ensure that subsequent
        // operations on this locatiion dereference the pointer.
        let dest_loc = self.iplace_to_location(dest);
        self.store_raw(&dest_loc, &*TEMP_LOC, *PTR_SIZE);
    }

    /// Compile a TIR statement.
    fn c_statement(&mut self, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::Store(dest, src) => self.c_istore(dest, src),
            Statement::BinaryOp {
                dest,
                op,
                opnd1,
                opnd2,
                checked,
            } => self.c_binop(dest, *op, opnd1, opnd2, *checked),
            Statement::MkRef(dest, src) => self.c_mkref(dest, src),
            Statement::DynOffs {
                dest,
                base,
                idx,
                scale,
            } => self.c_dynoffs(dest, base, idx, *scale),
            Statement::StorageDead(l) => self.free_register(l)?,
            Statement::Call(target, args, dest) => self.c_call(target, args, dest)?,
            Statement::Cast(dest, src) => self.c_cast(dest, src),
            Statement::Nop => {}
            Statement::Unimplemented(s) => todo!("{:?}", s),
        }

        Ok(())
    }

    fn c_mkref(&mut self, dest: &IPlace, src: &IPlace) {
        let src_loc = self.iplace_to_location(src);
        match src_loc {
            Location::Register(..) => {
                // This isn't possible as the allocator explicitely puts things which are
                // referenced onto the stack and never in registers.
                unreachable!()
            }
            Location::Mem(ref ro) => {
                debug_assert!(src_loc.uses_reg() != Some(*TEMP_REG));
                dynasm!(self.asm
                    ; lea Rq(*TEMP_REG), [Rq(ro.reg) + ro.off]
                );
            }
            Location::Const { .. } => todo!(),
            Location::Indirect { ref ptr, off } => {
                debug_assert!(src_loc.uses_reg() != Some(*TEMP_REG));
                match ptr {
                    IndirectLoc::Register(reg) => {
                        dynasm!(self.asm
                            ; lea Rq(*TEMP_REG), [Rq(reg) + off]
                        );
                    }
                    IndirectLoc::Mem(ro) => {
                        dynasm!(self.asm
                            ; lea Rq(*TEMP_REG), [Rq(ro.reg) + ro.off]
                            ; mov Rq(*TEMP_REG), [Rq(*TEMP_REG)]
                            ; add Rq(*TEMP_REG), off
                        );
                    }
                }
            }
            Location::NotLive => unreachable!(),
        }
        let dest_loc = self.iplace_to_location(dest);
        self.store_raw(&dest_loc, &*TEMP_LOC, SIR.ty(&src.ty()).size());
    }

    fn c_cast(&mut self, dest: &IPlace, src: &IPlace) {
        let src_loc = self.iplace_to_location(src);
        let ty = SIR.ty(&src.ty()); // Type of the source.
        let cty = SIR.ty(&dest.ty()); // Type of the cast (same as dest type).
        match ty {
            Ty::UnsignedInt(_) => self.c_cast_uint(src_loc, &ty, cty),
            _ => todo!(),
        }
        let dest_loc = self.iplace_to_location(dest);
        self.store_raw(&dest_loc, &*TEMP_LOC, SIR.ty(&dest.ty()).size());
    }

    fn c_cast_uint(&mut self, src: Location, ty: &Ty, cty: &Ty) {
        match src {
            Location::Register(reg) => {
                match cty.size() {
                    1 => {
                        dynasm!(self.asm
                            ; mov Rb(*TEMP_REG), Rb(reg)
                        );
                    }
                    2 => match ty.size() {
                        1 => {
                            dynasm!(self.asm
                                ; movzx Rw(*TEMP_REG), Rb(reg)
                            );
                        }
                        2 | 4 | 8 => {
                            dynasm!(self.asm
                                ; mov Rw(*TEMP_REG), Rw(reg)
                            );
                        }
                        _ => todo!("{}", ty.size()),
                    },
                    4 => match ty.size() {
                        1 => {
                            dynasm!(self.asm
                                ; movzx Rd(*TEMP_REG), Rb(reg)
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; movzx Rd(*TEMP_REG), Rw(reg)
                            );
                        }
                        4 | 8 => {
                            dynasm!(self.asm
                                ; mov Rd(*TEMP_REG), Rd(reg)
                            );
                        }
                        _ => todo!("{}", ty.size()),
                    },
                    8 => {
                        match ty.size() {
                            1 => {
                                dynasm!(self.asm
                                    ; movzx Rq(*TEMP_REG), Rb(reg)
                                );
                            }
                            2 => {
                                dynasm!(self.asm
                                    ; movzx Rq(*TEMP_REG), Rw(reg)
                                );
                            }
                            4 => {
                                // mov reg32, reg32 in x64 automatically zero-extends to
                                // reg64.
                                dynasm!(self.asm
                                    ; mov Rd(*TEMP_REG), Rd(reg)
                                );
                            }
                            8 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), Rq(reg)
                                );
                            }
                            _ => todo!("{}", ty.size()),
                        }
                    }
                    _ => todo!("{}", ty.size()),
                }
            }
            Location::Mem(_ro) => todo!(),
            Location::Indirect { .. } => todo!(),
            Location::Const { .. } => todo!(),
            Location::NotLive => unreachable!(),
        }
    }

    fn c_istore(&mut self, dest: &IPlace, src: &IPlace) {
        self.store(dest, src);
    }

    /// Store the value in `src_loc` into `dest_loc`.
    fn store(&mut self, dest_ip: &IPlace, src_ip: &IPlace) {
        let dest_loc = self.iplace_to_location(dest_ip);
        let src_loc = self.iplace_to_location(src_ip);
        self.store_raw(&dest_loc, &src_loc, SIR.ty(&dest_ip.ty()).size());
    }

    /// Stores src_loc into dest_loc.
    fn store_raw(&mut self, dest_loc: &Location, src_loc: &Location, size: u64) {
        // This is the one place in the compiler where we allow an explosion of cases over the
        // variants of `Location`. If elsewhere you find yourself matching over a pair of locations
        // you should try and re-work you code so it calls this.
        //
        // FIXME avoid partial register stalls.
        // FIXME this is massive. Move this (and store() to a new file).
        // FIXME constants are assumed to fit in a 64-bit register.

        /// Break a 64-bit value down into two 32-bit values. Used in scenarios where the X86_64
        /// ISA doesn't allow 64-bit constant encodings.
        fn split_i64(v: i64) -> (i32, i32) {
            ((v >> 32) as i32, (v & 0xffffffff) as i32)
        }

        // This can happen due to ZSTs.
        if size == 0 {
            return;
        }

        match (&dest_loc, &src_loc) {
            (Location::Register(dest_reg), Location::Register(src_reg)) => {
                dynasm!(self.asm
                    ; mov Rq(dest_reg), Rq(src_reg)
                );
            }
            (Location::Mem(dest_ro), Location::Register(src_reg)) => match size {
                1 => dynasm!(self.asm
                    ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off], Rb(src_reg)
                ),
                2 => dynasm!(self.asm
                    ; mov WORD [Rq(dest_ro.reg) + dest_ro.off], Rw(src_reg)
                ),
                4 => dynasm!(self.asm
                    ; mov DWORD [Rq(dest_ro.reg) + dest_ro.off], Rd(src_reg)
                ),
                8 => dynasm!(self.asm
                    ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(src_reg)
                ),
                _ => unreachable!(),
            },
            (Location::Mem(dest_ro), Location::Mem(src_ro)) => {
                if size <= 8 {
                    debug_assert!(dest_ro.reg != *TEMP_REG);
                    debug_assert!(src_ro.reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rb(*TEMP_REG), BYTE [Rq(src_ro.reg) + src_ro.off]
                            ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rw(*TEMP_REG), WORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov WORD [Rq(dest_ro.reg) + dest_ro.off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rd(*TEMP_REG), DWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov DWORD [Rq(dest_ro.reg) + dest_ro.off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    self.copy_memory(dest_ro, src_ro, size);
                }
            }
            (Location::Register(dest_reg), Location::Mem(src_ro)) => match size {
                1 => dynasm!(self.asm
                    ; mov Rb(dest_reg), BYTE [Rq(src_ro.reg) + src_ro.off]
                ),
                2 => dynasm!(self.asm
                    ; mov Rw(dest_reg), WORD [Rq(src_ro.reg) + src_ro.off]
                ),
                4 => dynasm!(self.asm
                    ; mov Rd(dest_reg), DWORD [Rq(src_ro.reg) + src_ro.off]
                ),
                8 => dynasm!(self.asm
                    ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                ),
                _ => unreachable!(),
            },
            (Location::Register(dest_reg), Location::Const { val: c_val, .. }) => {
                let i64_c = c_val.i64_cast();
                if i64_c <= i64::from(u32::MAX) {
                    dynasm!(self.asm
                        ; mov Rq(dest_reg), i64_c as i32
                    );
                } else {
                    // Can't move 64-bit constants in x86_64.
                    let i64_c = c_val.i64_cast();
                    let hi_word = (i64_c >> 32) as i32;
                    let lo_word = (i64_c & 0xffffffff) as i32;
                    dynasm!(self.asm
                        ; mov Rq(dest_reg), hi_word
                        ; shl Rq(dest_reg), 32
                        ; or Rq(dest_reg), lo_word
                    );
                }
            }
            (Location::Mem(ro), Location::Const { val: c_val, ty }) => {
                let c_i64 = c_val.i64_cast();
                match SIR.ty(&ty).size() {
                    1 => dynasm!(self.asm
                        ; mov BYTE [Rq(ro.reg) + ro.off], c_i64 as i8
                    ),
                    2 => dynasm!(self.asm
                        ; mov WORD [Rq(ro.reg) + ro.off], c_i64 as i16
                    ),
                    4 => dynasm!(self.asm
                        ; mov DWORD [Rq(ro.reg) + ro.off], c_i64 as i32
                    ),
                    8 => {
                        let (hi, lo) = split_i64(c_i64);
                        dynasm!(self.asm
                            ; mov DWORD [Rq(ro.reg) + ro.off], lo as i32
                            ; mov DWORD [Rq(ro.reg) + ro.off + 4], hi as i32
                        );
                    }
                    _ => todo!(),
                }
            }
            (
                Location::Register(dest_reg),
                Location::Indirect {
                    ptr: src_indloc,
                    off: src_off,
                },
            ) => match src_indloc {
                IndirectLoc::Register(src_reg) => match size {
                    1 => dynasm!(self.asm
                            ; mov Rb(dest_reg), BYTE [Rq(src_reg) + *src_off]
                    ),
                    2 => dynasm!(self.asm
                            ; mov Rw(dest_reg), WORD [Rq(src_reg) + *src_off]
                    ),
                    4 => dynasm!(self.asm
                            ; mov Rd(dest_reg), DWORD [Rq(src_reg) + *src_off]
                    ),
                    8 => dynasm!(self.asm
                            ; mov Rq(dest_reg), QWORD [Rq(src_reg) + *src_off]
                    ),
                    _ => todo!(),
                },
                IndirectLoc::Mem(src_ro) => match size {
                    1 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rb(dest_reg), BYTE [Rq(dest_reg) + *src_off]
                    ),
                    2 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rw(dest_reg), WORD [Rq(dest_reg) + *src_off]
                    ),
                    4 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rd(dest_reg), DWORD [Rq(dest_reg) + *src_off]
                    ),
                    8 => dynasm!(self.asm
                        ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.off]
                        ; mov Rq(dest_reg), QWORD [Rq(dest_reg) + *src_off]
                    ),
                    _ => todo!(),
                },
            },
            (
                Location::Indirect {
                    ptr: dest_indloc,
                    off: dest_off,
                },
                Location::Const { val: src_cval, .. },
            ) => {
                let src_i64 = src_cval.i64_cast();
                match dest_indloc {
                    IndirectLoc::Register(dest_reg) => match size {
                        1 => dynasm!(self.asm
                            ; mov BYTE [Rq(dest_reg) + *dest_off], src_i64 as i8
                        ),
                        2 => dynasm!(self.asm
                            ; mov WORD [Rq(dest_reg) + *dest_off], src_i64 as i16
                        ),
                        4 => dynasm!(self.asm
                            ; mov DWORD [Rq(dest_reg) + *dest_off], src_i64 as i32
                        ),
                        8 => {
                            let (hi, lo) = split_i64(src_i64);
                            dynasm!(self.asm
                                ; mov DWORD [Rq(dest_reg) + *dest_off], lo as i32
                                ; mov DWORD [Rq(dest_reg) + *dest_off + 4], hi as i32
                            );
                        }
                        _ => todo!(),
                    },
                    IndirectLoc::Mem(dest_ro) => {
                        debug_assert!(dest_ro.reg != *TEMP_REG);
                        match size {
                            1 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov BYTE [Rq(*TEMP_REG) + *dest_off], src_i64 as i8
                                );
                            }
                            2 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov WORD [Rq(*TEMP_REG) + *dest_off], src_i64 as i16
                                );
                            }
                            4 => {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov DWORD [Rq(*TEMP_REG) + *dest_off], src_i64 as i32
                                );
                            }
                            8 => {
                                let (hi, lo) = split_i64(src_i64);
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                                    ; mov DWORD [Rq(*TEMP_REG) + *dest_off], lo as i32
                                    ; mov DWORD [Rq(*TEMP_REG) + *dest_off + 4], hi as i32
                                );
                            }
                            _ => todo!(),
                        }
                    }
                }
            }
            (
                Location::Indirect {
                    ptr: dest_indloc,
                    off: dest_off,
                },
                Location::Register(src_reg),
            ) => match dest_indloc {
                IndirectLoc::Register(dest_reg) => match size {
                    1 => dynasm!(self.asm
                        ; mov BYTE [Rq(dest_reg) + *dest_off], Rb(src_reg)
                    ),
                    2 => dynasm!(self.asm
                        ; mov WORD [Rq(dest_reg) + *dest_off], Rw(src_reg)
                    ),
                    4 => dynasm!(self.asm
                        ; mov DWORD [Rq(dest_reg) + *dest_off], Rd(src_reg)
                    ),
                    8 => dynasm!(self.asm
                        ; mov QWORD [Rq(dest_reg) + *dest_off], Rq(src_reg)
                    ),
                    _ => todo!(),
                },
                IndirectLoc::Mem(dest_ro) => {
                    debug_assert!(*src_reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov BYTE [Rq(*TEMP_REG) + *dest_off], Rb(src_reg)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov WORD [Rq(*TEMP_REG) + *dest_off], Rw(src_reg)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov DWORD [Rq(*TEMP_REG) + *dest_off], Rd(src_reg)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(dest_ro.reg) + dest_ro.off]
                            ; mov QWORD [Rq(*TEMP_REG) + *dest_off], Rq(src_reg)
                        ),
                        _ => todo!(),
                    }
                }
            },
            (
                Location::Mem(dest_ro),
                Location::Indirect {
                    ptr: src_ind,
                    off: src_off,
                },
            ) => match src_ind {
                IndirectLoc::Mem(src_ro) => {
                    debug_assert!(src_ro.reg != *TEMP_REG);
                    match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov WORD [Rq(dest_ro.reg) + dest_ro.off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov DWORD [Rq(dest_ro.reg) + dest_ro.off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD  [Rq(src_ro.reg) + src_ro.off]
                            ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + *src_off]
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => todo!(),
                    }
                }
                IndirectLoc::Register(src_reg) => {
                    debug_assert!(*src_reg != *TEMP_REG);
                    match size {
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_reg) + *src_off]
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.off], Rq(*TEMP_REG)
                        ),
                        _ => todo!(),
                    }
                }
            },
            (
                Location::Indirect {
                    ptr: dest_ind,
                    off: dest_off,
                },
                Location::Mem(src_ro),
            ) => {
                debug_assert!(src_ro.reg != *TEMP_REG);
                match dest_ind {
                    IndirectLoc::Register(dest_reg) => match size {
                        1 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov BYTE [Rq(dest_reg) + *dest_off], Rb(*TEMP_REG)
                        ),
                        2 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov WORD [Rq(dest_reg) + *dest_off], Rw(*TEMP_REG)
                        ),
                        4 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov DWORD [Rq(dest_reg) + *dest_off], Rd(*TEMP_REG)
                        ),
                        8 => dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                            ; mov QWORD [Rq(dest_reg) + *dest_off], Rq(*TEMP_REG)
                        ),
                        _ => {
                            let dest_ro = RegAndOffset {
                                reg: *dest_reg,
                                off: 0,
                            };
                            self.copy_memory(&dest_ro, src_ro, size);
                        }
                    },
                    IndirectLoc::Mem(_) => todo!(),
                }
            }
            (Location::NotLive, _) | (_, Location::NotLive) => unreachable!(),
            _ => todo!(),
        }
    }

    /// Compile a guard in the trace, emitting code to abort execution in case the guard fails.
    fn c_guard(&mut self, guard: &Guard) {
        match guard {
            Guard {
                val,
                kind: GuardKind::OtherInteger(v),
            } => match self.iplace_to_location(val) {
                Location::Register(reg) => {
                    for c in v {
                        self.cmp_reg_const(reg, *c, SIR.ty(&val.ty()).size());
                        dynasm!(self.asm
                            ; je ->guardfail
                        );
                    }
                }
                _ => todo!(),
            },
            Guard {
                val,
                kind: GuardKind::Integer(c),
            } => match self.iplace_to_location(val) {
                Location::Register(reg) => {
                    self.cmp_reg_const(reg, *c, SIR.ty(&val.ty()).size());
                    dynasm!(self.asm
                        ; jne ->guardfail
                    );
                }
                Location::Mem(..) => todo!(),
                Location::Indirect { ptr, off } => match ptr {
                    IndirectLoc::Register(reg) => {
                        dynasm!(self.asm
                            ; mov Rq(*TEMP_REG), QWORD [Rq(reg) + off]
                        );
                        self.cmp_reg_const(*TEMP_REG, *c, SIR.ty(&val.ty()).size());
                        dynasm!(self.asm
                            ; jne ->guardfail
                        );
                    }
                    IndirectLoc::Mem(_ro) => todo!(),
                },
                _ => todo!(),
            },
            _ => todo!(),
        }
    }

    fn cmp_reg_const(&mut self, reg: u8, c: u128, size: u64) {
        match size {
            1 => {
                dynasm!(self.asm
                    ; cmp Rb(reg), i8::try_from(c).unwrap()
                );
            }
            2 => {
                dynasm!(self.asm
                    ; cmp Rw(reg), i16::try_from(c).unwrap()
                );
            }
            4 => {
                dynasm!(self.asm
                    ; cmp Rd(reg), i32::try_from(c).unwrap()
                );
            }
            8 => {
                dynasm!(self.asm
                    ; cmp Rq(reg), i32::try_from(c).unwrap()
                );
            }
            _ => todo!(),
        }
    }

    /// Print information about the state of the compiler and exit.
    fn crash_dump(self, e: Option<CompileError>) -> ! {
        eprintln!("\nThe trace compiler crashed!\n");

        if let Some(e) = e {
            eprintln!("Reason: {}.\n", e);
        } else {
            eprintln!("Reason: unknown");
        }

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
            ; mov rax, 1
            ; ->cleanup:
            ; add rsp, soff as i32
            ; pop rbp
            ; ret
            ; ->guardfail:
            ; mov rax, 0
            ; jmp ->cleanup
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
    fn finish(self, debug: bool) -> dynasmrt::ExecutableBuffer {
        let buf = self.asm.finalize().unwrap();
        if debug {
            // In debug mode the memory section which contains the compiled trace is marked as
            // writeable, which enables gdb/lldb to set breakpoints within the compiled code.
            unsafe {
                let ptr = buf.ptr(dynasmrt::AssemblyOffset(0)) as *mut libc::c_void;
                let len = buf.len();
                let alignment = ptr as usize % libc::sysconf(libc::_SC_PAGESIZE) as usize;
                let ptr = ptr.offset(-(alignment as isize));
                let len = len + alignment;
                libc::mprotect(ptr, len, libc::PROT_EXEC | libc::PROT_WRITE);
            }
        }
        buf
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
            mc: tc.finish(false),
            _pd: PhantomData,
        };
        (ct, spills)
    }

    /// Compile a TIR trace, returning executable code.
    pub fn compile(tt: TirTrace) -> CompiledTrace<TT> {
        let tc = TraceCompiler::<TT>::_compile(tt);
        CompiledTrace::<TT> {
            mc: tc.finish(false),
            _pd: PhantomData,
        }
    }

    fn _compile(tt: TirTrace) -> Self {
        let assembler = dynasmrt::x64::Assembler::new().unwrap();

        // Make the TirTrace mutable so we can drain it into the TraceCompiler.
        let mut tt = tt;
        let mut tc = TraceCompiler::<TT> {
            asm: assembler,
            register_content_map: REG_POOL.iter().map(|r| (*r, RegAlloc::Free)).collect(),
            variable_location_map: HashMap::new(),
            local_decls: tt.local_decls.clone(),
            stack_builder: StackBuilder::default(),
            addr_map: tt.addr_map.drain().into_iter().collect(),
            _pd: PhantomData,
        };

        tc.init();

        for i in 0..tt.len() {
            let res = match unsafe { tt.op(i) } {
                TirOp::Statement(st) => tc.c_statement(st),
                TirOp::Guard(g) => {
                    tc.c_guard(g);
                    Ok(())
                }
            };

            // FIXME -- Later errors should not be fatal. We should be able to abort trace
            // compilation and carry on.
            match res {
                Ok(_) => (),
                Err(e) => tc.crash_dump(Some(e)),
            }
        }
        tc.ret();
        tc
    }

    /// Returns a pointer to the static symbol `sym`, or an error if it cannot be found.
    fn find_symbol(sym: &str) -> Result<*mut c_void, CompileError> {
        let sym_arg = CString::new(sym).unwrap();
        let addr = unsafe { dlsym(RTLD_DEFAULT, sym_arg.into_raw()) };

        if addr.is_null() {
            Err(CompileError::UnknownSymbol(sym.to_owned()))
        } else {
            Ok(addr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompileError, HashMap, Local, Location, RegAlloc, TraceCompiler, REG_POOL};
    use crate::stack_builder::StackBuilder;
    use fm::FMBuilder;
    use libc::{abs, c_void, getuid};
    use regex::Regex;
    use std::marker::PhantomData;
    use yktrace::sir::SIR;
    use yktrace::tir::TirTrace;
    use yktrace::{start_tracing, TracingKind};

    extern "C" {
        fn add6(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) -> u64;
    }
    extern "C" {
        fn add_some(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
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

    #[test]
    fn test_simple() {
        struct IO(u8);

        #[interp_step]
        #[inline(never)]
        fn simple(io: &mut IO) {
            let x = 13;
            io.0 = x;
        }

        let th = start_tracing(TracingKind::HardwareTracing);
        simple(&mut IO(0));
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 13);
    }

    // Repeatedly fetching the register for the same local should yield the same register and
    // should not exhaust the allocator.
    #[ignore] // Broken because we don't know what type IDs to put in local_decls.
    #[test]
    fn reg_alloc_same_local() {
        struct IO(u8);
        let mut tc = TraceCompiler::<IO> {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: REG_POOL
                .iter()
                .cloned()
                .map(|r| (r, RegAlloc::Free))
                .collect(),
            variable_location_map: HashMap::new(),
            local_decls: HashMap::default(),
            stack_builder: StackBuilder::default(),
            addr_map: HashMap::new(),
            _pd: PhantomData,
        };

        for _ in 0..32 {
            assert_eq!(
                tc.local_to_location(Local(1)),
                tc.local_to_location(Local(1))
            );
        }
    }

    // Locals should be allocated to different registers.
    #[ignore] // Broken because we don't know what type IDs to put in local_decls.
    #[test]
    fn reg_alloc() {
        let local_decls = HashMap::new();
        struct IO(u8);
        let mut tc = TraceCompiler::<IO> {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: REG_POOL
                .iter()
                .cloned()
                .map(|r| (r, RegAlloc::Free))
                .collect(),
            variable_location_map: HashMap::new(),
            local_decls,
            stack_builder: StackBuilder::default(),
            addr_map: HashMap::new(),
            _pd: PhantomData,
        };

        let mut seen: Vec<Location> = Vec::new();
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

    #[test]
    fn test_function_call_simple() {
        struct IO(u8);

        #[interp_step]
        #[inline(never)]
        fn fcall(io: &mut IO) {
            io.0 = farg(13);
            let _z = farg(14);
        }

        let mut io = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        fcall(&mut io);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 13);
    }

    #[test]
    fn test_function_call_nested() {
        struct IO(u8);

        fn fnested3(i: u8, _j: u8) -> u8 {
            let c = i;
            c
        }

        fn fnested2(i: u8) -> u8 {
            fnested3(i, 10)
        }

        #[interp_step]
        fn fnested(io: &mut IO) {
            io.0 = fnested2(20);
        }

        let mut io = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        fnested(&mut io);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 20);
    }

    // Test finding a symbol in a shared object.
    #[test]
    fn find_symbol_shared() {
        struct IO(());
        assert!(TraceCompiler::<IO>::find_symbol("printf") == Ok(libc::printf as *mut c_void));
    }

    // Test finding a symbol in the main binary.
    // For this to work the binary must have been linked with `--export-dynamic`, which ykrustc
    // appends to the linker command line.
    #[test]
    #[no_mangle]
    fn find_symbol_main() {
        struct IO(());
        assert!(
            TraceCompiler::<IO>::find_symbol("find_symbol_main")
                == Ok(find_symbol_main as *mut c_void)
        );
    }

    // Check that a non-existent symbol cannot be found.
    #[test]
    fn find_nonexistent_symbol() {
        struct IO(());
        assert_eq!(
            TraceCompiler::<IO>::find_symbol("__xxxyyyzzz__"),
            Err(CompileError::UnknownSymbol("__xxxyyyzzz__".to_owned()))
        );
    }

    // A trace which contains a call to something which we don't have SIR for should emit a TIR
    // call operation.
    #[test]
    fn call_symbol_tir() {
        struct IO(());
        #[interp_step]
        fn interp_step(_: &mut IO) {
            let _ = unsafe { add6(1, 1, 1, 1, 1, 1) };
        }

        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut IO(()));
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        assert_tir(
            "...\n\
            ops:\n\
              ...
              %a = call(add6, [1u64, 1u64, 1u64, 1u64, 1u64, 1u64])\n\
              ...
              dead(%a)\n\
              ...",
            &tir_trace,
        );
    }

    /// Execute a trace which calls a symbol accepting no arguments, but which does return a value.
    #[test]
    fn exec_call_symbol_no_args() {
        struct IO(u32);
        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 = unsafe { getuid() };
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, args.0);
    }

    /// Execute a trace which calls a symbol accepting arguments and returns a value.
    #[test]
    fn exec_call_symbol_with_arg() {
        struct IO(i32);
        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 = unsafe { abs(io.0) };
        }

        let mut inputs = IO(-56);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(-56);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, args.0);
    }

    /// The same as `exec_call_symbol_args_with_rv`, just using a constant argument.
    #[test]
    fn exec_call_symbol_with_const_arg() {
        struct IO(i32);
        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 = unsafe { abs(-123) };
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn exec_call_symbol_with_many_args() {
        struct IO(u64);
        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 = unsafe { add6(1, 2, 3, 4, 5, 6) };
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, 21);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn exec_call_symbol_with_many_args_some_ignored() {
        struct IO(u64);
        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 = unsafe { add_some(1, 2, 3, 4, 5) };
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(args.0, 7);
        assert_eq!(args.0, inputs.0);
    }

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_spilling_simple() {
        struct IO(u64);

        #[interp_step]
        fn many_locals(io: &mut IO) {
            let _a = 1;
            let _b = 2;
            let _c = 3;
            let _d = 4;
            let _e = 5;
            let _f = 6;
            let h = 7;
            let _g = true;
            io.0 = h;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        many_locals(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 7);
        assert_eq!(spills, 3); // Three u8s.
    }

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_spilling_u64() {
        struct IO(u64);

        fn u64value() -> u64 {
            // We need an extra function here to avoid SIR optimising this by assigning assigning the
            // constant directly to the return value (which is a register).
            4294967296 + 8
        }

        #[inline(never)]
        #[interp_step]
        fn spill_u64(io: &mut IO) {
            let _a = 1;
            let _b = 2;
            let _c = 3;
            let _d = 4;
            let _e = 5;
            let _f = 6;
            let _g = 7;
            let h: u64 = u64value();
            io.0 = h;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        spill_u64(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 4294967296 + 8);
        assert_eq!(spills, 2 * 8);
    }

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_mov_register_to_stack() {
        struct IO(u8, u8);

        #[interp_step]
        fn register_to_stack(io: &mut IO) {
            let _a = 1;
            let _b = 2;
            let _c = 3;
            let _d = 4;
            let _e = 5;
            let _f = 6;
            let _g = 7;
            let h = io.0;
            io.1 = h;
        }

        let mut inputs = IO(8, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        register_to_stack(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(8, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, inputs.1);
        assert_eq!(spills, 9); // f, g: i32, h:  u8.
    }

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_mov_stack_to_register() {
        struct IO(u8);

        #[interp_step]
        fn stack_to_register(io: &mut IO) {
            let _a = 1;
            let _b = 2;
            let c = 3;
            let _d = 4;
            // When returning from `farg` all registers are full, so `e` needs to be allocated on the
            // stack. However, after we have returned, anything allocated during `farg` is freed. Thus
            // returning `e` will allocate a new local in a (newly freed) register, resulting in a `mov
            // reg, [rbp]` instruction.
            let e = farg(c);
            io.0 = e;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        stack_to_register(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 3);
        assert_eq!(spills, 1); // Just one u8.
    }

    #[test]
    fn ext_call_and_spilling() {
        struct IO(u64);

        #[interp_step]
        fn ext_call(io: &mut IO) {
            let a = 1;
            let b = 2;
            let c = 3;
            let d = 4;
            let e = 5;
            // When calling `add_some` argument `a` is loaded from a register, while the remaining
            // arguments are loaded from the stack.
            let expect = unsafe { add_some(a, b, c, d, e) };
            io.0 = expect;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        ext_call(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, 7);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn test_binop_add_simple() {
        #[derive(Eq, PartialEq, Debug)]
        struct IO(u64, u64, u64);

        #[interp_step]
        fn interp_stepx(io: &mut IO) {
            io.2 = io.0 + io.1 + 3;
        }

        let mut inputs = IO(5, 2, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_stepx(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(5, 2, 0);
        ct.execute(&mut args);
        assert_eq!(args, IO(5, 2, 10));
    }

    #[test]
    fn test_binop_other() {
        #[derive(Eq, PartialEq, Debug)]
        struct IO(u64, u64, u64);

        #[interp_step]
        fn interp_stepx(io: &mut IO) {
            io.2 = io.0 * 3 - 5;
            io.1 = io.2 / 2;
        }

        let mut inputs = IO(5, 2, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_stepx(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(5, 2, 0);
        ct.execute(&mut args);
        assert_eq!(args, IO(5, 5, 10));
    }

    #[test]
    fn test_ref_deref_simple() {
        #[derive(Debug)]
        struct IO(u64);

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let mut x = 9;
            let y = &mut x;
            *y = 10;
            io.0 = *y;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 10);
    }

    #[test]
    fn test_ref_deref_double_xxx() {
        #[derive(Debug)]
        struct IO(u64);

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let mut x = 9;
            let y = &mut &mut x;
            **y = 4;
            io.0 = x;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 4);
    }

    #[test]
    fn test_ref_deref_double_and_field() {
        #[derive(Debug)]
        struct IO(u64);

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let five = 5;
            let mut s = (4u64, &five);
            let y = &mut s;
            io.0 = *y.1;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 5);
    }

    #[test]
    fn test_ref_deref_stack() {
        struct IO(u64);

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let _a = 1;
            let _b = 2;
            let _c = 3;
            let _d = 4;
            let _e = 5;
            let _f = 6;
            let mut x = 9;
            let y = &mut x;
            *y = 10;
            let z = *y;
            io.0 = z
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 10);
    }

    /// Dereferences a variable that lives on the stack and stores it in a register.
    #[test]
    fn test_deref_stack_to_register() {
        fn deref1(arg: u64) -> u64 {
            let a = &arg;
            return *a;
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let _a = 1;
            let _b = 2;
            let _c = 3;
            let f = 6;
            io.0 = deref1(f);
        }

        struct IO(u64);
        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 6);
    }

    #[test]
    fn test_deref_register_to_stack() {
        struct IO(u64);

        fn deref2(arg: u64) -> u64 {
            let a = &arg;
            let _b = 2;
            let _c = 3;
            let _d = 4;
            return *a;
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let f = 6;
            io.0 = deref2(f);
        }

        // This test dereferences a variable that lives on the stack and stores it in a register.
        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 6);
    }

    #[test]
    fn test_do_not_trace() {
        struct IO(u8);

        #[do_not_trace]
        fn dont_trace_this(a: u8) -> u8 {
            let b = 2;
            let c = a + b;
            c
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 = dont_trace_this(io.0);
        }

        let mut inputs = IO(1);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();

        assert_tir(
            "
            local_decls:
              ...
            ops:
              ...
              %s1 = call(...
              ...",
            &tir_trace,
        );

        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(1);
        ct.execute(&mut args);
        assert_eq!(args.0, 3);
    }

    #[test]
    fn test_do_not_trace_stdlib() {
        struct IO<'a>(&'a mut Vec<u64>);

        #[interp_step]
        fn dont_trace_stdlib(io: &mut IO) {
            io.0.push(3);
        }

        let mut vec: Vec<u64> = Vec::new();
        let mut inputs = IO(&mut vec);
        let th = start_tracing(TracingKind::HardwareTracing);
        dont_trace_stdlib(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut argv: Vec<u64> = Vec::new();
        let mut args = IO(&mut argv);
        ct.execute(&mut args);
        assert_eq!(argv.len(), 1);
        assert_eq!(argv[0], 3);
    }

    #[test]
    fn test_projection_chain() {
        #[derive(Debug)]
        struct IO((usize, u8, usize), u8, S, usize);

        #[derive(Debug, PartialEq)]
        struct S {
            x: usize,
            y: usize,
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.1 = (io.0).1;
            io.3 = io.2.y;
        }

        let s = S { x: 5, y: 6 };
        let t = (1, 2, 3);
        let mut inputs = IO(t, 0u8, s, 0usize);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);

        let t2 = (1, 2, 3);
        let s2 = S { x: 5, y: 6 };
        let mut args = IO(t2, 0u8, s2, 0usize);
        ct.execute(&mut args);
        assert_eq!(args.0, (1usize, 2u8, 3usize));
        assert_eq!(args.1, 2u8);
        assert_eq!(args.2, S { x: 5, y: 6 });
        assert_eq!(args.3, 6);
    }

    #[test]
    fn test_projection_lhs() {
        struct IO((u8, u8), u8);

        #[interp_step]
        fn interp_step(io: &mut IO) {
            (io.0).1 = io.1;
        }

        let t = (1u8, 2u8);
        let mut inputs = IO(t, 3u8);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let t2 = (1u8, 2u8);
        let mut args = IO(t2, 3u8);
        ct.execute(&mut args);
        assert_eq!((args.0).1, 3);
    }

    #[test]
    fn test_array() {
        struct IO<'a>(&'a mut [u8; 3], u8);

        #[interp_step]
        #[inline(never)]
        fn array(io: &mut IO) {
            let z = io.0[1];
            // 1 = &io.0
            // 2 = 1
            // 3 = 2 * size
            // 4 = 1 + 3
            io.1 = z;
        }

        let mut a = [3, 4, 5];
        let mut inputs = IO(&mut a, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        array(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        assert_eq!(inputs.1, 4);
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut a2 = [3, 4, 5];
        let mut args = IO(&mut a2, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, 4);
    }

    #[test]
    fn test_array_nested() {
        struct IO<'a>(&'a mut [[u8; 3]; 2], u8);

        #[interp_step]
        #[inline(never)]
        fn array(io: &mut IO) {
            let z = io.0[1][2];
            io.1 = z;
        }

        let mut a = [[3, 4, 5], [6, 7, 8]];
        let mut inputs = IO(&mut a, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        array(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        assert_eq!(inputs.1, 8);
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut a2 = [[3, 4, 5], [6, 7, 8]];
        let mut args = IO(&mut a2, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, 8);
    }

    #[test]
    fn test_array_nested_mad() {
        struct S([u16; 4]);
        struct IO<'a>(&'a mut [S; 3], u16);

        #[interp_step]
        #[inline(never)]
        fn array(io: &mut IO) {
            let z = io.0[2].0[2];
            io.1 = z;
        }

        let mut a = [S([3, 4, 5, 6]), S([7, 8, 9, 10]), S([11, 12, 13, 14])];
        let mut inputs = IO(&mut a, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        array(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        assert_eq!(inputs.1, 13);
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut a2 = [S([3, 4, 5, 6]), S([7, 8, 9, 10]), S([11, 12, 13, 14])];
        let mut args = IO(&mut a2, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, 13);
    }

    /// Test codegen of field access on a struct ref on the right-hand side.
    #[test]
    fn rhs_struct_ref_field() {
        struct IO(u8);

        #[interp_step]
        fn add1(io: &mut IO) {
            io.0 = io.0 + 1
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        add1(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);

        let mut args = IO(10);
        ct.execute(&mut args);
        assert_eq!(args.0, 11);
    }

    /// Test codegen of indexing a struct ref on the left-hand side.
    #[test]
    fn mut_lhs_struct_ref() {
        struct IO(u8);

        #[interp_step]
        fn set100(io: &mut IO) {
            io.0 = 100;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        set100(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);

        let mut args = IO(10);
        ct.execute(&mut args);
        assert_eq!(args.0, 100);
    }

    /// Test codegen of copying something which doesn't fit in a register.
    #[test]
    fn place_larger_than_reg() {
        #[derive(Debug, Eq, PartialEq)]
        struct S(u64, u64, u64);
        struct IO(S);

        #[interp_step]
        fn ten(io: &mut IO) {
            io.0 = S(10, 10, 10);
        }

        let mut inputs = IO(S(0, 0, 0));
        let th = start_tracing(TracingKind::HardwareTracing);
        ten(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        assert_eq!(inputs.0, S(10, 10, 10));

        let mut args = IO(S(1, 1, 1));
        ct.execute(&mut args);
        assert_eq!(args.0, S(10, 10, 10));
    }

    #[test]
    #[ignore] // FIXME Broken during new trimming scheme. Seg faults.
    fn test_rvalue_len() {
        struct IO<'a>(&'a [u8], u8);

        fn matchthis(inputs: &IO, pc: usize) -> u8 {
            let x = match inputs.0[pc] as char {
                'a' => 1,
                'b' => 2,
                _ => 0,
            };
            x
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let x = matchthis(&io, 0);
            io.1 = x;
        }

        let a = "abc".as_bytes();
        let mut inputs = IO(&a, 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut a2 = "abc".as_bytes();
        let mut args = IO(&mut a2, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, 1);
    }

    // Only `interp_step` annotated functions and their callees should remain after trace trimming.
    #[test]
    fn trim_junk() {
        struct IO(u8);

        #[interp_step]
        fn interp_step(io: &mut IO) {
            io.0 += 1;
        }

        let mut inputs = IO(0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        inputs.0 = 0; // Should get trimmed.
        interp_step(&mut inputs);
        inputs.0 = 0; // Should get trimmed
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);

        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 3);
    }

    #[test]
    fn test_comparison() {
        struct IO(u8, bool);

        fn checks(i: u8) -> bool {
            let a = i == 0;
            let b = i > 1;
            let c = i < 1;
            if a && b || c {
                true
            } else {
                false
            }
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let x = checks(io.0);
            io.1 = x;
        }

        let mut inputs = IO(0, false);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0, false);
        ct.execute(&mut args);
        assert_eq!(args.1, true);
    }

    #[test]
    fn test_guard() {
        struct IO(u8, u8);

        fn guard(i: u8) -> u8 {
            if i != 3 {
                9
            } else {
                10
            }
        }

        #[interp_step]
        fn interp_step(io: &mut IO) {
            let x = guard(io.0);
            io.1 = x;
        }

        let mut inputs = IO(std::hint::black_box(|i| i)(0), 0);
        let th = start_tracing(TracingKind::HardwareTracing);
        interp_step(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0, 0);
        let cr = ct.execute(&mut args);
        assert_eq!(cr, true);
        assert_eq!(args.1, 9);
        // Execute trace with input that fails the guard.
        let mut args = IO(3, 0);
        let cr = ct.execute(&mut args);
        assert_eq!(cr, false);
    }

    #[test]
    fn test_match() {
        struct IO(u8);

        #[interp_step]
        #[inline(never)]
        fn matchthis(io: &mut IO) {
            let x = match io.0 {
                1 => 2,
                2 => 3,
                _ => 0,
            };
            io.0 = x;
        }

        let th = start_tracing(TracingKind::HardwareTracing);
        matchthis(&mut IO(1));
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(1);
        let cr = ct.execute(&mut args);
        assert_eq!(cr, true);
        assert_eq!(args.0, 2);
    }

    #[test]
    fn test_cast() {
        struct IO(u16, u8);

        #[interp_step]
        #[inline(never)]
        fn matchthis(io: &mut IO) {
            let y = match io.1 as char {
                'a' => 1,
                'b' => 2,
                _ => 3,
            };
            io.0 = y;
        }

        let mut io = IO(0, 97);
        let th = start_tracing(TracingKind::HardwareTracing);
        matchthis(&mut io);
        let sir_trace = th.stop_tracing().unwrap();
        assert_eq!(io.0, 1);
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0, 97);
        let cr = ct.execute(&mut args);
        assert_eq!(cr, true);
        assert_eq!(args.0, 1);
    }

    #[test]
    fn test_vec_add() {
        struct IO {
            ptr: usize,
            cells: Vec<u8>,
        }

        #[interp_step]
        #[inline(never)]
        fn vec_add(io: &mut IO) {
            io.cells[io.ptr] = io.cells[io.ptr].wrapping_add(1);
        }

        let cells = vec![0, 1, 2];
        let mut io = IO { ptr: 1, cells };
        let th = start_tracing(TracingKind::HardwareTracing);
        vec_add(&mut io);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let cells = vec![1, 2, 3];
        let mut args = IO { ptr: 1, cells };
        let cr = ct.execute(&mut args);
        assert_eq!(cr, true);
        assert_eq!(args.cells, vec![1, 3, 3]);
        let cr = ct.execute(&mut args);
        assert_eq!(cr, true);
        assert_eq!(args.cells, vec![1, 4, 3]);
    }
}
