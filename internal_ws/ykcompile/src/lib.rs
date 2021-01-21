//! The Yorick TIR trace compiler.

#![cfg_attr(test, feature(test))]

#[macro_use]
extern crate dynasmrt;
#[macro_use]
extern crate lazy_static;
#[cfg(test)]
extern crate test;

mod stack_builder;
mod store;

use dynasmrt::{x64::Rq::*, Register};
use libc::{c_void, dlsym, RTLD_DEFAULT};
use stack_builder::StackBuilder;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::ffi::CString;
use std::fmt::{self, Display, Formatter};
use std::mem;
use std::process::Command;
use ykpack::{IPlace, OffT, SignedIntTy, Ty, TyKind, TypeId, UnsignedIntTy};
use yktrace::tir::{
    BinOp, CallOperand, Constant, Guard, GuardKind, Local, Statement, TirOp, TirTrace,
};
use yktrace::{sir::SIR, INTERP_STEP_ARG};

use dynasmrt::{DynasmApi, DynasmLabelApi};

lazy_static! {
    // Registers that are caller-save as per the Sys-V ABI.
    // Note that R11 is also caller-save, but it's our temproary register and we never want to
    // preserve its value across calls.
    static ref CALLER_SAVED_REGS: [u8; 8] = [RAX.code(), RDI.code(), RSI.code(), RDX.code(),
                                            RCX.code(), R8.code(), R9.code(), R10.code()];

    // Registers that are callee-save as per the Sys-V ABI.
    // Note that RBP is also callee-save, but is handled specially.
    static ref CALLEE_SAVED_REGS: [u8; 5] = [RBX.code(), R12.code(), R13.code(),
                                            R14.code(), R15.code()];

    // The register partitioning. These arrays must not overlap.
    static ref TEMP_REG: u8 = R11.code();
    pub static ref REG_POOL: [u8; 11] = [RAX.code(), RCX.code(), RDX.code(), R8.code(), R9.code(),
                                     R10.code(), RBX.code(), R12.code(), R13.code(), R14.code(),
                                     R15.code()];

    // The trace inputs/outputs are always allocated to this reserved register.
    // This register should not appear in REG_POOL.
    static ref TIO_REG: u8 = RDI.code();

    static ref TEMP_LOC: Location = Location::Reg(*TEMP_REG);
    static ref PTR_SIZE: u64 = u64::try_from(mem::size_of::<usize>()).unwrap();
}

/// Generates functions for add/sub-style operations.
/// The first operand must be in a register.
macro_rules! binop_add_sub {
    ($name: ident, $op:expr) => {
        fn $name(&mut self, opnd1_reg: u8, opnd2: &IPlace) {
            let size = SIR.ty(&opnd2.ty()).size();
            let opnd2_loc = self.iplace_to_location(opnd2);
            match opnd2_loc {
                Location::Reg(r) => match size {
                    1 => {
                        dynasm!(self.asm
                            ; $op Rb(opnd1_reg), Rb(r)
                        );
                    }
                    2 => {
                        dynasm!(self.asm
                            ; $op Rw(opnd1_reg), Rw(r)
                        );
                    }
                    4 => {
                        dynasm!(self.asm
                            ; $op Rd(opnd1_reg), Rd(r)
                        );
                    }
                    8 => {
                        dynasm!(self.asm
                            ; $op Rq(opnd1_reg), Rq(r)
                        );
                    }
                    _ => unreachable!(format!("{}", SIR.ty(&opnd2.ty()))),
                },
                Location::Mem(..) => todo!(),
                Location::Const { val, .. } => {
                    let val = val.i64_cast();
                    match size {
                        1 => {
                            dynasm!(self.asm
                                ; $op Rb(opnd1_reg), val as i8
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; $op Rw(opnd1_reg), val as i16
                            );
                        }
                        4 => {
                            dynasm!(self.asm
                                ; $op Rd(opnd1_reg), val as i32
                            );
                        }
                        8 => {
                            if i32::try_from(val).is_err() {
                                // FIXME Work around x86_64 encoding limitations (no imm64 operands).
                                todo!();
                            } else {
                                dynasm!(self.asm
                                    ; $op Rq(opnd1_reg), val as i32
                                );
                            }
                        }
                        _ => unreachable!(format!("{}", SIR.ty(&opnd2.ty()))),
                    }
                }
                Location::Indirect { .. } => todo!(),
            }
        }
    }
}

/// Generates functions for mul/div-style operations.
/// The first operand must be in a register.
macro_rules! binop_mul_div {
    ($name: ident, $op:expr) => {
        fn $name(&mut self, opnd1_reg: u8, opnd2: &IPlace) {
            // mul and div overwrite RAX, RDX, so save them first.
            dynasm!(self.asm
                ; push rax
                ; push rdx
                ; xor rdx, rdx
            );
            dynasm!(self.asm
                ; mov rax, Rq(opnd1_reg)
            );
            let size = SIR.ty(&opnd2.ty()).size();
            let src_loc = self.iplace_to_location(opnd2);
            match src_loc {
                Location::Reg(r) => match size {
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
                    _ => unreachable!(format!("{}", SIR.ty(&opnd2.ty()))),
                },
                Location::Mem(..) => todo!(),
                Location::Const { val, .. } => {
                    // It's safe to use TEMP_REG here, because opnd2 isn't in a register and if
                    // opnd1_reg was TEMP_REG then we've already moved it into RAX.
                    let val = val.i64_cast();
                    match size {
                        1 => {
                            dynasm!(self.asm
                                ; mov Rb(*TEMP_REG), val as i8
                                ; $op Rb(*TEMP_REG)
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; mov Rw(*TEMP_REG), val as i16
                                ; $op Rw(*TEMP_REG)
                            );
                        }
                        4 => {
                            dynasm!(self.asm
                                ; mov Rd(*TEMP_REG), val as i32
                                ; $op Rd(*TEMP_REG)
                            );
                        }
                        8 => {
                            if i32::try_from(val).is_err() {
                                // FIXME Work around x86_64 encoding limitations (no imm64 operands).
                                todo!();
                            } else {
                                dynasm!(self.asm
                                    ; mov Rq(*TEMP_REG), val as i32
                                    ; $op Rq(*TEMP_REG)
                                );
                            }
                        }
                        _ => unreachable!(format!("{}", SIR.ty(&opnd2.ty()))),
                    }
                }
                Location::Indirect { .. } => todo!(),
            }

            // Restore RAX, RDX
            dynasm!(self.asm
                ; mov Rq(opnd1_reg), rax
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
        Location::Reg(r) => match r {
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

/// Compile a TIR trace, returning executable code.
pub fn compile_trace(tt: TirTrace) -> CompiledTrace {
    CompiledTrace {
        mc: TraceCompiler::_compile(tt, false),
    }
}

/// A compiled `SIRTrace`.
pub struct CompiledTrace {
    /// A compiled trace.
    mc: dynasmrt::ExecutableBuffer,
}

impl CompiledTrace {
    /// Execute the trace by calling (not jumping to) the first instruction's address.
    pub unsafe fn execute<TT>(&self, args: &mut TT) -> bool {
        let func: fn(&mut TT) -> bool = mem::transmute(self.mc.ptr(dynasmrt::AssemblyOffset(0)));
        self.exec_trace::<TT>(func, args)
    }

    /// Actually call the code. This is a separate function making it easier to set a debugger
    /// breakpoint right before entering the trace.
    fn exec_trace<TT>(&self, t_fn: fn(&mut TT) -> bool, args: &mut TT) -> bool {
        t_fn(args)
    }

    /// Return a pointer to the mmap'd block of memory containing the trace. The underlying data is
    /// guaranteed never to move in memory.
    pub fn ptr(&self) -> *const u8 {
        self.mc.ptr(dynasmrt::AssemblyOffset(0))
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
    Reg(u8),
    /// There's a pointer in memory somewhere.
    Mem(RegAndOffset),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    /// A value in a register.
    Reg(u8),
    /// A statically known memory location relative to a register.
    Mem(RegAndOffset),
    /// A location that contains a pointer to some underlying storage.
    Indirect { ptr: IndirectLoc, off: OffT },
    /// A statically known constant.
    Const { val: Constant, ty: TypeId },
}

impl Location {
    /// Creates a new memory location from a register and an offset.
    fn new_mem(reg: u8, off: OffT) -> Self {
        Self::Mem(RegAndOffset { reg, off })
    }

    /// If `self` is a `Mem` then unwrap it, otherwise panic.
    fn unwrap_mem(&self) -> &RegAndOffset {
        if let Location::Mem(ro) = self {
            ro
        } else {
            panic!("tried to unwrap a Mem location when it wasn't a Mem");
        }
    }

    /// Returns which register (if any) is used in addressing this location.
    fn uses_reg(&self) -> Option<u8> {
        match self {
            Location::Reg(reg) => Some(*reg),
            Location::Mem(RegAndOffset { reg, .. }) => Some(*reg),
            Location::Indirect {
                ptr: IndirectLoc::Reg(reg),
                ..
            }
            | Location::Indirect {
                ptr: IndirectLoc::Mem(RegAndOffset { reg, .. }),
                ..
            } => Some(*reg),
            Location::Const { .. } => None,
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
            Location::Reg(..) | Location::Const { .. } => todo!("offsetting a constant"),
        }
    }

    /// Converts a direct place to an indirect place for use as a pointer.
    fn to_indirect(&self) -> Self {
        let ptr = match self {
            Location::Reg(r) => IndirectLoc::Reg(*r),
            Location::Mem(ro) => IndirectLoc::Mem(ro.clone()),
            _ => unreachable!(),
        };
        Location::Indirect { ptr, off: 0 }
    }
}

/// Allocation of one of the REG_POOL. Temporary registers are tracked separately.
#[derive(Debug)]
enum RegAlloc {
    Local(Local),
    Free,
}

use ykpack::LocalDecl;

/// The `TraceCompiler` takes a `SIRTrace` and compiles it to machine code. Returns a `CompiledTrace`.
pub struct TraceCompiler {
    /// The dynasm assembler which will do all of the heavy lifting of the assembly.
    asm: dynasmrt::x64::Assembler,
    /// Stores the content of each register.
    register_content_map: HashMap<u8, RegAlloc>,
    /// Maps trace locals to their location (register, stack).
    variable_location_map: HashMap<Local, Location>,
    /// Local decls of the tir trace.
    pub local_decls: HashMap<Local, LocalDecl>,
    /// Stack builder for allocating objects on the stack.
    stack_builder: StackBuilder,
    /// Stores the memory addresses of local functions.
    addr_map: HashMap<String, u64>,
}

impl TraceCompiler {
    pub fn new(local_decls: HashMap<Local, LocalDecl>, addr_map: HashMap<String, u64>) -> Self {
        let mut tc = TraceCompiler {
            asm: dynasmrt::x64::Assembler::new().unwrap(),
            register_content_map: REG_POOL.iter().map(|r| (*r, RegAlloc::Free)).collect(),
            variable_location_map: HashMap::new(),
            local_decls,
            stack_builder: StackBuilder::default(),
            addr_map,
        };

        // At the start of the trace, jump to the label that allocates stack space.
        dynasm!(tc.asm
            ; jmp ->reserve
            ; ->crash:
            ; ud2
            ; ->main:
        );
        tc
    }

    fn can_live_in_register(decl: &LocalDecl) -> bool {
        if decl.referenced {
            // We must allocate it on the stack so that we can reference it.
            return false;
        }

        // FIXME: optimisation: small structs and tuples etc. could actually live in a register.
        let ty = &*SIR.ty(&decl.ty);
        match &ty.kind {
            TyKind::UnsignedInt(ui) => !matches!(ui, UnsignedIntTy::U128),
            TyKind::SignedInt(si) => !matches!(si, SignedIntTy::I128),
            TyKind::Array { .. } => false,
            TyKind::Slice(_) => false,
            TyKind::Ref(_) | TyKind::Bool | TyKind::Char => true,
            TyKind::Struct(..) | TyKind::Tuple(..) => false,
            TyKind::Unimplemented(..) => todo!("{}", ty),
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
            e => todo!("{}", e),
        }
    }

    /// Given a local, returns the register allocation for it, or, if there is no allocation yet,
    /// performs one.
    pub fn local_to_location(&mut self, l: Local) -> Location {
        if l == INTERP_STEP_ARG {
            // There is a register set aside for trace inputs.
            Location::Reg(*TIO_REG)
        } else if let Some(location) = self.variable_location_map.get(&l) {
            // We already have a location for this local.
            location.clone()
        } else {
            let decl = &self.local_decls[&l];
            if Self::can_live_in_register(&decl) {
                // Find a free register to store this local.
                let loc = if let Some(reg) = self.get_free_register() {
                    self.register_content_map.insert(reg, RegAlloc::Local(l));
                    Location::Reg(reg)
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

    fn local_live(&mut self, local: &Local) {
        // Assign a Location to this Local.
        self.local_to_location(*local);
    }

    /// Notifies the register allocator that a local has died and that its storage may be freed.
    pub fn local_dead(&mut self, local: &Local) -> Result<(), CompileError> {
        match self
            .variable_location_map
            .get(local)
            .expect("freeing unallocated register")
        {
            Location::Reg(reg) => {
                // This local is currently stored in a register, so free the register.
                //
                // Note that if we are marking the reserved TIO_REG free then this actually adds a
                // new register key to the map (as opposed to marking a pre-existing entry free).
                // This is safe since if we are freeing TIO_REG, then the trace inputs local must
                // not be used for the remainder of the trace.
                self.register_content_map.insert(*reg, RegAlloc::Free);
            }
            Location::Mem { .. } | Location::Indirect { .. } => {}
            Location::Const { .. } => unreachable!(),
        }
        self.variable_location_map.remove(local);
        Ok(())
    }

    /// Copy bytes from one memory location to another.
    fn copy_memory(&mut self, dest: &RegAndOffset, src: &RegAndOffset, size: u64) {
        // We use memmove(3), as it's not clear if MIR (and therefore SIR) could cause copies
        // involving overlapping buffers.
        let sym = Self::find_symbol("memmove").unwrap();
        self.save_regs(&*CALLER_SAVED_REGS);
        dynasm!(self.asm
            ; push rax
            ; xor rax, rax
        );

        dynasm!(self.asm
            ; lea rdi, [Rq(dest.reg) + dest.off]
        );
        if src.reg == RDI.code() {
            // If the second argument lived in RDI then we've just overwritten it, so we load it
            // back from when we callee-saved it.
            let mut rdi_stackpos = CALLER_SAVED_REGS
                .iter()
                .rev()
                .position(|r| *r == RDI.code())
                .unwrap();
            // We've also pushed RAX in the meantime.
            rdi_stackpos += 1;
            dynasm!(self.asm
                ; mov rsi, [rsp + i32::try_from(rdi_stackpos).unwrap() * 8]
            );
        } else {
            dynasm!(self.asm
                ; lea rsi, [Rq(src.reg) + src.off]
            );
        };
        dynasm!(self.asm
            ; mov rdx, size as i32
            ; mov r11, QWORD sym as i64
            ; call r11
            ; pop rax
        );
        self.restore_regs(&*CALLER_SAVED_REGS);
    }

    /// Emit a NOP operation.
    fn _nop(&mut self) {
        dynasm!(self.asm
            ; nop
        );
    }

    /// Push the specified registers to the stack in order.
    fn save_regs(&mut self, regs: &[u8]) {
        for reg in regs.iter() {
            dynasm!(self.asm
                ; push Rq(reg)
            );
        }
    }
    /// Pop the specified registers from the stack in reverse order.
    fn restore_regs(&mut self, regs: &[u8]) {
        for reg in regs.iter().rev() {
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
        // OPTIMISE: Only save registers in use by the register allocator.
        let mut save_regs = CALLER_SAVED_REGS.iter().cloned().collect::<Vec<u8>>();
        if let Some(d) = dest {
            let dest_loc = self.iplace_to_location(d);
            if let Location::Reg(dest_reg) = dest_loc {
                // If the result of the call is destined for one of the caller-save registers, then
                // there's no point in saving the register.
                save_regs.retain(|r| *r != dest_reg);
            }
        }
        self.save_regs(&*save_regs);

        // Helper function to find the index of a caller-save register previously pushed to the
        // stack. The first register pushed is at the highest stack offset (from the stack
        // pointer), hence reversing the order of `save_regs`. Returns `None` if `reg` was never
        // saved during caller-save.
        let saved_stack_index = |reg: u8| -> Option<i32> {
            save_regs
                .iter()
                .rev()
                .position(|&r| r == reg)
                .map(|i| i32::try_from(i).unwrap())
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
                Location::Reg(reg) => {
                    if let Some(idx) = saved_stack_index(reg) {
                        // We saved this register to the stack during caller-save. Since there is
                        // overlap between caller-save registers and argument registers, we may
                        // have overwritten the value in the meantime. So we should load the value
                        // back from the stack.
                        dynasm!(self.asm
                            ; mov Rq(arg_reg), [rsp + idx * 8]
                        );
                    } else {
                        // We didn't save this register, so it remains intact.
                        dynasm!(self.asm
                            ; mov Rq(arg_reg), Rq(reg)
                        );
                    }
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
            }
        }

        let sym_addr = if let Some(addr) = self.addr_map.get(sym) {
            *addr as i64
        } else {
            TraceCompiler::find_symbol(sym)? as i64
        };
        dynasm!(self.asm
            // In Sys-V ABI, `al` is a hidden argument used to specify the number of vector args
            // for a vararg call. We don't support this right now, so set it to zero.
            ; xor rax, rax
            ; mov Rq(*TEMP_REG), QWORD sym_addr
            ; call Rq(*TEMP_REG)
            // Stash return value. We do this because restore_regs() below may clobber RAX.
            ; mov Rq(*TEMP_REG), rax
        );

        // Restore caller-save registers.
        self.restore_regs(&save_regs);

        if let Some(d) = dest {
            let dest_loc = self.iplace_to_location(d);
            self.store_raw(&dest_loc, &Location::Reg(*TEMP_REG), SIR.ty(&d.ty()).size());
        }

        Ok(())
    }

    /// Load an IPlace into the given register. Panic if it doesn't fit.
    fn load_reg_iplace(&mut self, reg: u8, src_ip: &IPlace) -> Location {
        let dest_loc = Location::Reg(reg);
        let src_loc = self.iplace_to_location(src_ip);
        self.store_raw(&dest_loc, &src_loc, SIR.ty(&src_ip.ty()).size());
        dest_loc
    }

    fn c_binop(&mut self, dest: &IPlace, op: BinOp, opnd1: &IPlace, opnd2: &IPlace, checked: bool) {
        let opnd1_ty = SIR.ty(&opnd1.ty());
        debug_assert!(opnd1_ty == SIR.ty(&opnd2.ty()));

        // For now this whole function assumes we are operating on integers.
        if !opnd1_ty.is_int() {
            todo!("binops for non-integers");
        }

        match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                return self.c_condition(dest, &op, opnd1, opnd2);
            }
            _ => {}
        }

        // We do this in three stages.
        // 1) Copy the first operand into the temp register.
        self.load_reg_iplace(*TEMP_REG, opnd1);

        // 2) Perform arithmetic.
        match op {
            BinOp::Add => self.c_binop_add(*TEMP_REG, opnd2),
            BinOp::Sub => self.c_binop_sub(*TEMP_REG, opnd2),
            BinOp::Mul => {
                if opnd1_ty.is_signed_int() {
                    todo!("signed mul"); // use IMUL
                } else {
                    self.c_binop_mul(*TEMP_REG, opnd2);
                }
            }
            BinOp::Div => {
                if opnd1_ty.is_signed_int() {
                    todo!("signed div"); // use IDIV
                } else {
                    self.c_binop_div(*TEMP_REG, opnd2);
                }
            }
            _ => todo!(),
        }

        // 3) Move the result to where it is supposed to live.
        let dest_loc = self.iplace_to_location(dest);
        let size = opnd1_ty.size();
        if checked {
            // If it is a checked operation, then we have to build a (value, overflow-flag) tuple.
            // Let's do the flag first, so as to read EFLAGS closest to where they are set.
            let dest_ro = dest_loc.unwrap_mem();
            let sir_ty = SIR.ty(&dest.ty());
            let tty = sir_ty.unwrap_tuple();
            let flag_off = i32::try_from(tty.fields.offsets[1]).unwrap();

            if opnd1_ty.is_signed_int() {
                dynasm!(self.asm
                    ; jo >overflow
                );
            } else {
                dynasm!(self.asm
                    ; jc >overflow
                );
            }
            dynasm!(self.asm
                ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off + flag_off], 0
                ; jmp >done
                ; overflow:
                ; mov BYTE [Rq(dest_ro.reg) + dest_ro.off + flag_off], 1
                ; done:
            );
        }
        self.store_raw(&dest_loc, &*TEMP_LOC, size);
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
            Location::Reg(reg) => match ty.size() {
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
            Location::Reg(..) => todo!(),
            Location::Mem(..) => todo!(),
            Location::Indirect { ptr, off } => match ptr {
                IndirectLoc::Reg(..) => todo!(),
                IndirectLoc::Mem(ind_ro) => {
                    dynasm!(self.asm
                        ; mov Rq(*TEMP_REG), [Rq(ind_ro.reg) + ind_ro.off]
                        ; add Rq(*TEMP_REG), off
                        ; jo ->crash
                    );
                }
            },
            Location::Const { .. } => todo!(),
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
            Statement::StorageLive(l) => self.local_live(l),
            Statement::StorageDead(l) => self.local_dead(l)?,
            Statement::Call(target, args, dest) => self.c_call(target, args, dest)?,
            Statement::Cast(dest, src) => self.c_cast(dest, src),
            Statement::Nop | Statement::Debug(..) => {}
            Statement::Unimplemented(s) => todo!("{:?}", s),
        }

        Ok(())
    }

    fn c_mkref(&mut self, dest: &IPlace, src: &IPlace) {
        let src_loc = self.iplace_to_location(src);
        match src_loc {
            Location::Reg(..) => {
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
                    IndirectLoc::Reg(reg) => {
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
        }
        let dest_loc = self.iplace_to_location(dest);
        debug_assert_eq!(SIR.ty(&dest.ty()).size(), *PTR_SIZE);
        self.store_raw(&dest_loc, &*TEMP_LOC, *PTR_SIZE);
    }

    fn c_cast(&mut self, dest: &IPlace, src: &IPlace) {
        let src_loc = self.iplace_to_location(src);
        let ty = &*SIR.ty(&src.ty()); // Type of the source.
        let cty = SIR.ty(&dest.ty()); // Type of the cast (same as dest type).
        match ty.kind {
            TyKind::UnsignedInt(_) => self.c_cast_uint(src_loc, &ty, &cty),
            _ => todo!(),
        }
        let dest_loc = self.iplace_to_location(dest);
        self.store_raw(&dest_loc, &*TEMP_LOC, SIR.ty(&dest.ty()).size());
    }

    fn c_cast_uint(&mut self, src: Location, ty: &Ty, cty: &Ty) {
        match src {
            Location::Reg(reg) => {
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
        }
    }

    fn c_istore(&mut self, dest: &IPlace, src: &IPlace) {
        self.store(dest, src);
    }

    /// Compile a guard in the trace, emitting code to abort execution in case the guard fails.
    fn c_guard(&mut self, guard: &Guard) {
        // FIXME some of the terminators from which we build these guards can have cleanup blocks.
        // Currently we don't run any cleanup, but should we?
        match guard {
            Guard {
                val,
                kind: GuardKind::OtherInteger(v),
                ..
            } => match self.iplace_to_location(val) {
                Location::Reg(reg) => {
                    for c in v {
                        self.cmp_reg_const(reg, *c, SIR.ty(&val.ty()).size());
                        dynasm!(self.asm
                            ; je ->guardfail
                        );
                    }
                }
                Location::Mem(ro) => {
                    dynasm!(self.asm
                        ; mov Rq(*TEMP_REG), QWORD [Rq(ro.reg) + ro.off]
                    );
                    for c in v {
                        self.cmp_reg_const(*TEMP_REG, *c, SIR.ty(&val.ty()).size());
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
                ..
            } => match self.iplace_to_location(val) {
                Location::Reg(reg) => {
                    self.cmp_reg_const(reg, *c, SIR.ty(&val.ty()).size());
                    dynasm!(self.asm
                        ; jne ->guardfail
                    );
                }
                Location::Mem(ro) => {
                    dynasm!(self.asm
                        ; mov Rq(*TEMP_REG), QWORD [Rq(ro.reg) + ro.off]
                    );
                    self.cmp_reg_const(*TEMP_REG, *c, SIR.ty(&val.ty()).size());
                    dynasm!(self.asm
                        ; jne ->guardfail
                    );
                }
                Location::Indirect { ptr, off } => {
                    match ptr {
                        IndirectLoc::Reg(reg) => {
                            dynasm!(self.asm
                                ; mov Rq(*TEMP_REG), QWORD [Rq(reg) + off]
                            );
                        }
                        IndirectLoc::Mem(src_ro) => {
                            dynasm!(self.asm
                                ; mov Rq(*TEMP_REG), QWORD [Rq(src_ro.reg) + src_ro.off]
                                ; mov Rq(*TEMP_REG), QWORD [Rq(*TEMP_REG) + off]
                            );
                        }
                    }
                    self.cmp_reg_const(*TEMP_REG, *c, SIR.ty(&val.ty()).size());
                    dynasm!(self.asm
                        ; jne ->guardfail
                    );
                }
                _ => todo!(),
            },
            Guard {
                val,
                kind: GuardKind::Boolean(expect),
                ..
            } => match self.iplace_to_location(val) {
                Location::Reg(reg) => {
                    dynasm!(self.asm
                        ; cmp Rb(reg), *expect as i8
                        ; jne ->guardfail
                    );
                }
                Location::Mem(ro) => {
                    dynasm!(self.asm
                        ; cmp BYTE [Rq(ro.reg) + ro.off], *expect as i8
                        ; jne ->guardfail
                    );
                }
                _ => todo!(),
            },
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
            ; mov rax, 1 // Signifies that there were no guard failures.
            ; ->cleanup:
        );
        self.restore_regs(&*CALLEE_SAVED_REGS);
        dynasm!(self.asm
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
        );
        self.save_regs(&*CALLEE_SAVED_REGS);
        dynasm!(self.asm
            ; jmp ->main
        );
    }

    fn _compile(mut tt: TirTrace, debug: bool) -> dynasmrt::ExecutableBuffer {
        let mut tc: Self = TraceCompiler::new(
            tt.local_decls.clone(),
            tt.addr_map.drain().into_iter().collect(),
        );

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
        let buf = tc.asm.finalize().unwrap();
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

    /// Returns a pointer to the static symbol `sym`, or an error if it cannot be found.
    pub fn find_symbol(sym: &str) -> Result<*mut c_void, CompileError> {
        let sym_arg = CString::new(sym).unwrap();
        let addr = unsafe { dlsym(RTLD_DEFAULT, sym_arg.into_raw()) };

        if addr.is_null() {
            Err(CompileError::UnknownSymbol(sym.to_owned()))
        } else {
            Ok(addr)
        }
    }
}
