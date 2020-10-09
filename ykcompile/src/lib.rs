//! The Yorick TIR trace compiler.

#![feature(proc_macro_hygiene)]
#![feature(test)]
#![feature(core_intrinsics)]

#[macro_use]
extern crate dynasmrt;
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
use ykpack::{SignedIntTy, Ty, TypeId, UnsignedIntTy};
use yktrace::sir::SIR;
use yktrace::tir::{
    BinOp, CallOperand, Constant, ConstantInt, Guard, Local, Operand, Place, Projection, Rvalue,
    Statement, TirOp, TirTrace,
};

use dynasmrt::{DynasmApi, DynasmLabelApi};

const CALLER_SAVE_REGS: [dynasmrt::x64::Rq; 8] = [RDI, RSI, RDX, RCX, R8, R9, R10, R11];

// Register partitioning. These arrays must not overlap.
// FIXME add callee save registers to the pool. Trace code will need to save/restore them.
const TEMP_REGS: [dynasmrt::x64::Rq; 2] = [R10, R11];
const LOCAL_REGS: [dynasmrt::x64::Rq; 4] = [R9, R8, RDX, RCX];

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
    pub fn execute(&self, args: &mut TT) {
        let func: fn(&mut TT) = unsafe { mem::transmute(self.mc.ptr(dynasmrt::AssemblyOffset(0))) };
        self.exec_trace(func, args);
    }

    /// Actually call the code. This is a separate function making it easier to set a debugger
    /// breakpoint right before entering the trace.
    fn exec_trace(&self, t_fn: fn(&mut TT), args: &mut TT) {
        t_fn(args);
    }
}

/// Represents a memory location using a register and an offset.
#[derive(Debug, Clone, PartialEq)]
pub struct RegAndOffset {
    reg: u8,
    offs: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    /// A value in a register.
    Register(u8),
    /// A statically known memory location relative to a register.
    Mem(RegAndOffset),
    /// A runtime memory location stored in a register.
    Addr(u8),
    /// A non-live location. Used by the register allocator.
    NotLive,
}

impl Location {
    /// Creates a new memory location from a register and an offset.
    fn new_mem(reg: u8, offs: i32) -> Self {
        Self::Mem(RegAndOffset { reg, offs })
    }

    /// If `self` is a `Mem` then unwrap it, otherwise panic.
    fn unwrap_mem(&self) -> &RegAndOffset {
        if let Location::Mem(ro) = self {
            ro
        } else {
            panic!("tried to unwrap a Mem location when it wasn't a Mem");
        }
    }

    /// If `self` is a `Register` then unwrap it, otherwise panic.
    fn unwrap_reg(&self) -> u8 {
        if let Location::Register(reg) = self {
            *reg
        } else {
            panic!("tried to unwrap a Register location when it wasn't a Register");
        }
    }
}

/// Allocation of one of the LOCAL_REGS. Temporary registers are tracked separately.
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
    /// Available temproary registers.
    temp_regs: Vec<u8>,
    /// Local referencing the input arguments to the trace.
    trace_inputs_local: Option<Local>,
    /// Local decls of the tir trace.
    local_decls: HashMap<Local, LocalDecl>,
    /// Stack builder for allocating objects on the stack.
    stack_builder: StackBuilder,
    addr_map: HashMap<String, u64>,
    _pd: PhantomData<TT>,
}

impl<TT> TraceCompiler<TT> {
    fn can_live_in_register(tyid: &TypeId) -> bool {
        // FIXME: optimisation: small structs and tuples etc. could actually live in a register.
        let ty = SIR.ty(tyid);
        match ty {
            Ty::UnsignedInt(ui) => match ui {
                UnsignedIntTy::U128 => false,
                _ => true,
            },
            Ty::SignedInt(si) => match si {
                SignedIntTy::I128 => false,
                _ => true,
            },
            Ty::Array(_) => false,
            Ty::Slice(_) => false,
            Ty::Ref(_) | Ty::Bool => true,
            Ty::Struct(..) | Ty::Tuple(..) => false,
            Ty::Unimplemented(..) => todo!("{}", ty),
        }
    }

    /// Determine if the type needs to be copied when it is being dereferenced.
    fn is_copyable(tyid: &TypeId) -> bool {
        let ty = SIR.ty(tyid);
        match ty {
            Ty::UnsignedInt(ui) => match ui {
                UnsignedIntTy::U128 => false,
                _ => true,
            },
            Ty::SignedInt(si) => match si {
                SignedIntTy::I128 => false,
                _ => true,
            },
            // An array is copyable if its elements are.
            Ty::Array(ety) => Self::is_copyable(ety),
            Ty::Slice(ety) => Self::is_copyable(ety),
            Ty::Ref(_) | Ty::Bool => true,
            // FIXME A struct is copyable if it implements the Copy trait.
            Ty::Struct(..) => false,
            // FIXME A tuple is copyable if all its elements are.
            Ty::Tuple(..) => false,
            Ty::Unimplemented(..) => todo!("{}", ty),
        }
    }

    fn place_to_location(&mut self, p: &Place, store: bool) -> (Location, Ty) {
        if !p.projection.is_empty() {
            self.resolve_projection(p, store)
        } else {
            let ty = self.place_ty(&Place::from(p.local)).clone();
            (self.local_to_location(p.local), ty)
        }
    }

    /// Takes a `Place`, resolves all projections, and returns a `Location` containing the result.
    fn resolve_projection(&mut self, p: &Place, store: bool) -> (Location, Ty) {
        let mut curloc = self.local_to_location(p.local);
        let mut ty = self.place_ty(&Place::from(p.local)).clone();
        let mut iter = p.projection.iter().peekable();
        while let Some(proj) = iter.next() {
            match proj {
                Projection::Field(idx) => match ty {
                    Ty::Struct(sty) => match curloc {
                        Location::Mem(ro) => {
                            let offs = sty.fields.offsets[usize::try_from(*idx).unwrap()];
                            ty = SIR
                                .ty(&sty.fields.tys[usize::try_from(*idx).unwrap()])
                                .clone();
                            curloc =
                                Location::new_mem(ro.reg, ro.offs + i32::try_from(offs).unwrap());
                        }
                        _ => unreachable!("{:?}", curloc),
                    },
                    Ty::Tuple(tty) => match curloc {
                        Location::Mem(ro) => {
                            let offs = tty.fields.offsets[usize::try_from(*idx).unwrap()];
                            ty = SIR
                                .ty(&tty.fields.tys[usize::try_from(*idx).unwrap()])
                                .clone();
                            curloc =
                                Location::new_mem(ro.reg, ro.offs + i32::try_from(offs).unwrap());
                        }
                        _ => unreachable!("{:?}", curloc),
                    },
                    Ty::Ref(tyid) => match SIR.ty(&tyid) {
                        Ty::Struct(sty) => {
                            let offs = sty.fields.offsets[usize::try_from(*idx).unwrap()];
                            ty = SIR
                                .ty(&sty.fields.tys[usize::try_from(*idx).unwrap()])
                                .clone();
                            let temp = self.create_temporary();
                            match &curloc {
                                Location::Mem(ro) => {
                                    dynasm!(self.asm
                                        ; lea Rq(temp), [Rq(ro.reg) + ro.offs + i32::try_from(offs).unwrap()]
                                    );
                                }
                                Location::Register(reg) => {
                                    dynasm!(self.asm
                                        ; lea Rq(temp), [Rq(reg) + i32::try_from(offs).unwrap()]
                                    );
                                }
                                _ => unreachable!(),
                            }
                            self.free_if_temp(curloc);
                            curloc = if store {
                                Location::Addr(temp)
                            } else {
                                dynasm!(self.asm
                                    ; mov Rq(temp), [Rq(temp)]
                                );
                                Location::Register(temp)
                            };
                        }
                        Ty::Tuple(_tty) => todo!(),
                        _ => unreachable!(),
                    },
                    _ => todo!("{:?}", ty),
                },
                Projection::Deref => {
                    // FIXME We currently assume Deref is only called on Refs.

                    // Are we dereferencing a reference, if so, what's its type.
                    let tyid = match ty {
                        Ty::Ref(rty) => rty.clone(),
                        _ => todo!(),
                    };

                    // Special case: If the `Deref` is followed by an `Index` or `Field`
                    // projection, we defer resolution to them and don't copy the value.
                    // FIXME Do we need to check all remaining projections?
                    let copy = match iter.peek() {
                        Some(Projection::Index(_)) => false,
                        Some(Projection::Field(_)) => false,
                        _ => true,
                    };
                    if Self::is_copyable(&tyid) && copy {
                        match SIR.ty(&tyid) {
                            Ty::Array(_) | Ty::Tuple(_) | Ty::Struct(_) => todo!(),
                            _ => {}
                        }
                        // Copy referenced value into a temporary.
                        let temp = self.create_temporary();
                        match &curloc {
                            Location::Mem(ro) => {
                                // Deref value and copy it.
                                dynasm!(self.asm
                                    ; mov Rq(temp), [Rq(ro.reg) + ro.offs]
                                );
                            }
                            Location::Register(reg) | Location::Addr(reg) => {
                                dynasm!(self.asm
                                    ; mov Rq(temp), Rq(reg)
                                );
                            }
                            _ => unreachable!(),
                        };
                        self.free_if_temp(curloc);
                        curloc = if store {
                            Location::Addr(temp)
                        } else {
                            dynasm!(self.asm
                                ; mov Rq(temp), [Rq(temp)]
                            );
                            Location::Register(temp)
                        };
                        ty = SIR.ty(&tyid).clone();
                    }
                }
                Projection::Index(local) => {
                    // Get the type of the array elements.
                    let elem_ty = match ty {
                        Ty::Array(_) => {
                            // FIXME Since we can't compile array construction yet, we can assume
                            // we are always dealing with references here. Once we can, the
                            // assembler instructions below also need updating.
                            todo!()
                        }
                        Ty::Ref(tyid) => match SIR.ty(&tyid) {
                            Ty::Array(ety) => SIR.ty(&ety),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    // Compute the offset of this index.
                    let temp = self.create_temporary();
                    match self.local_to_location(*local) {
                        Location::Register(reg) => {
                            dynasm!(self.asm
                                ; imul Rq(temp), Rq(reg), elem_ty.size() as i32
                            );
                        }
                        Location::Mem(ro) => {
                            dynasm!(self.asm
                                ; imul Rq(temp), [Rq(ro.reg) + ro.offs], elem_ty.size() as i32
                            );
                        }
                        _ => todo!(),
                    }
                    // Add together the index and the array address and retrieve its value.
                    match &curloc {
                        Location::Mem(ro) => {
                            dynasm!(self.asm
                                ; add Rq(temp), [Rq(ro.reg) + ro.offs]
                            );
                        }
                        Location::Register(reg) => {
                            dynasm!(self.asm
                                ; add Rq(temp), Rq(reg)
                            );
                        }
                        _ => unreachable!(),
                    }
                    self.free_if_temp(curloc);
                    curloc = if store {
                        Location::Addr(temp)
                    } else {
                        dynasm!(self.asm
                            ; mov Rq(temp), [Rq(temp)]
                        );
                        Location::Register(temp)
                    };
                    ty = elem_ty.clone();
                }
                _ => todo!("{}", p),
            }
        }
        (curloc, ty)
    }

    /// Given a local, returns the register allocation for it, or, if there is no allocation yet,
    /// performs one.
    fn local_to_location(&mut self, l: Local) -> Location {
        if Some(l) == self.trace_inputs_local {
            // If the local references `trace_inputs` return its location on the stack, which is
            // stored in the first argument of the executed trace.
            Location::new_mem(RDI.code(), 0 as i32)
        } else if let Some(location) = self.variable_location_map.get(&l) {
            // We already have a location for this local.
            location.clone()
        } else {
            let tyid = self.local_decls[&l].ty;
            if Self::can_live_in_register(&tyid) {
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
                let ty = SIR.ty(&tyid);
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

    /// Find a free register to be used as a temporary. If no free register can be found, a
    /// register containing a Local is selected and its content spilled to the stack.
    fn create_temporary(&mut self) -> u8 {
        self.temp_regs
            .pop()
            .unwrap_or_else(|| panic!("Exhausted temporary registers!"))
    }

    /// Free the temporary register so it can be re-used.
    fn free_if_temp(&mut self, loc: Location) {
        match loc {
            Location::Register(reg) | Location::Addr(reg) => {
                // FIXME cache the collected list we are searching here.
                if TEMP_REGS
                    .iter()
                    .map(|r| r.code())
                    .collect::<Vec<u8>>()
                    .contains(&reg)
                {
                    debug_assert!(!self.temp_regs.contains(&reg), "double free temp reg");
                    self.temp_regs.push(reg);
                }
            }
            Location::Mem { .. } => {}
            _ => unreachable!(),
        }
    }

    /// Notifies the register allocator that the register allocated to `local` may now be re-used.
    fn free_register(&mut self, local: &Local) -> Result<(), CompileError> {
        match self.variable_location_map.get(local) {
            Some(Location::Register(reg)) | Some(Location::Addr(reg)) => {
                debug_assert!(!TEMP_REGS
                    .iter()
                    .map(|r| r.code())
                    .collect::<Vec<u8>>()
                    .contains(reg));
                // If this local is currently stored in a register, free it.
                self.register_content_map.insert(*reg, RegAlloc::Free);
            }
            Some(Location::Mem { .. }) => {}
            Some(Location::NotLive) => unreachable!(),
            None => unreachable!("freeing unallocated register"),
        }
        self.variable_location_map.insert(*local, Location::NotLive);
        Ok(())
    }

    /// Returns whether the register content map contains any temporaries. This is used as a sanity
    /// check at the end of a trace to make sure we haven't forgotten to free temporaries at the
    /// end of an operation.
    fn check_temporaries(&self) -> bool {
        self.temp_regs.len() == TEMP_REGS.len()
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
            ; lea rdi, [Rq(dest.reg) + dest.offs]
            ; lea rsi, [Rq(src.reg) + src.offs]
            ; mov rdx, size as i32
            ; mov r11, QWORD sym as i64
            ; call r11
            ; pop rax
        );
        self.caller_save_restore();
    }

    /// Get the type of a place.
    fn place_ty(&self, p: &Place) -> &Ty {
        SIR.ty(&self.local_decls[&p.local].ty)
    }

    /// Codegen a `Place` into a `Location`.
    fn c_place(&mut self, p: &Place) -> Location {
        self.place_to_location(p, false).0
    }

    /// Codegen a reference into a `Location`.
    fn c_ref(&mut self, p: &Place) -> Location {
        // Deal with the special case `&*`, i.e. referencing a `Deref` on a reference just returns
        // the reference.
        // FIXME Make sure the special case is only triggered for `&` on Refs and nothing else,
        // e.g. `&*`.
        if let Some(pj) = p.projection.get(0) {
            if matches!(pj, Projection::Deref)
                && matches!(SIR.ty(&self.local_decls[&p.local].ty), Ty::Ref(_))
            {
                // Clone the projection while removing the `Deref` from the end.
                let mut newproj = Vec::new();
                for p in p.projection.iter().take(p.projection.len() - 1) {
                    newproj.push(p.clone());
                }
                let np = Place {
                    local: p.local,
                    projection: newproj,
                };
                let (rloc, _) = self.place_to_location(&np, false);
                let reg = self.create_temporary();
                match rloc {
                    Location::Register(reg2) => {
                        dynasm!(self.asm
                            ; mov Rq(reg), Rq(reg2)
                        );
                    }
                    _ => todo!(),
                }
                self.free_if_temp(rloc);
                return Location::Register(reg);
            }
        }

        // We can only reference Locals living on the stack. So move it there if it doesn't.
        let reg = self.create_temporary();
        let rloc = match self.place_to_location(p, false) {
            (Location::Register(reg2), _) => {
                let loc = self.stack_builder.alloc(8, 8);
                let ro = loc.unwrap_mem();
                dynasm!(self.asm
                    ; mov [Rq(ro.reg) + ro.offs], Rq(reg2)
                );
                // This Local lives now on the stack...
                self.variable_location_map.insert(p.local, loc.clone());
                // ...so we can free its old register.
                debug_assert!(!TEMP_REGS
                    .iter()
                    .map(|r| r.code())
                    .collect::<Vec<u8>>()
                    .contains(&reg2));
                self.register_content_map.insert(reg2, RegAlloc::Free);
                loc
            }
            (loc, _) => loc,
        };
        // Now create the reference.
        match &rloc {
            Location::Mem(ro) => {
                dynasm!(self.asm
                    ; lea Rq(reg), [Rq(ro.reg) + ro.offs]
                );
            }
            _ => unreachable!(),
        };
        self.free_if_temp(rloc);
        Location::Register(reg)
    }

    fn c_len(&mut self, p: &Place) -> Location {
        let (loc, _) = self.place_to_location(p, true);
        let dst = self.create_temporary();
        match loc {
            Location::Addr(src) => {
                // A slice &[T] is a fat pointer with its length in the last 8 bytes.
                dynasm!(self.asm
                    ; mov Rq(dst), [Rq(src) + 8]
                );
            }
            // FIXME Can `Len` be called on non-references?
            _ => unreachable!(),
        }
        self.free_if_temp(loc);
        Location::Register(dst)
    }

    /// Emit a NOP operation.
    fn nop(&mut self) {
        dynasm!(self.asm
            ; nop
        );
    }

    /// Codegen a constant integer into a `Location`.
    fn c_constint(&mut self, constant: &ConstantInt) -> Location {
        let reg = self.create_temporary();
        let c_val = constant.i64_cast();
        dynasm!(self.asm
            ; mov Rq(reg), QWORD c_val
        );
        Location::Register(reg)
    }

    /// Codegen a Boolean into a `Location`.
    fn c_bool(&mut self, b: bool) -> Location {
        let reg = self.create_temporary();
        dynasm!(self.asm
            ; mov Rq(reg), QWORD b as i64
        );
        Location::Register(reg)
    }

    /// Compile the entry into an inlined function call.
    fn c_enter(&mut self, args: &Vec<Operand>, off: u32) {
        // Move call arguments into registers.
        for (op, i) in args.iter().zip(1..) {
            let loc = match op {
                Operand::Place(p) => self.c_place(p),
                Operand::Constant(c) => match c {
                    Constant::Int(ci) => self.c_constint(ci),
                    Constant::Bool(b) => self.c_bool(*b),
                    c => todo!("{}", c),
                },
            };
            let arg_idx = Place::from(Local(i + off));
            self.store(&arg_idx, loc.clone());
            self.free_if_temp(loc);
        }
    }

    /// Push all of the caller-save registers to the stack.
    fn caller_save(&mut self) {
        for reg in CALLER_SAVE_REGS.iter() {
            dynasm!(self.asm
                ; push Rq(reg.code())
            );
        }
    }

    /// Restore caller-save registers from the stack.
    fn caller_save_restore(&mut self) {
        for reg in CALLER_SAVE_REGS.iter().rev() {
            dynasm!(self.asm
                ; pop Rq(reg.code())
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
        args: &Vec<Operand>,
        dest: &Option<Place>,
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
                    .position(|&r| r.code() == reg)
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
            // `unwrap()` must succeed, as we checked there are no more than 6 args above.
            let arg_reg = arg_regs.pop().unwrap();

            match arg {
                Operand::Place(place) => {
                    // Load argument back from the stack.
                    let (loc, _) = self.place_to_location(place, false);
                    match &loc {
                        Location::Register(reg) => {
                            let off = stack_index(*reg) * 8;
                            dynasm!(self.asm
                                ; mov Rq(arg_reg), [rsp + off]
                            );
                        }
                        Location::Mem(ro) => {
                            dynasm!(self.asm
                                ; mov Rq(arg_reg), [Rq(ro.reg) + ro.offs]
                            );
                        }
                        Location::Addr(_) | Location::NotLive => unreachable!(),
                    };
                    self.free_if_temp(loc);
                }
                Operand::Constant(c) => {
                    dynasm!(self.asm
                        ; mov Rq(arg_reg), QWORD c.i64_cast()
                    );
                }
            };
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
            self.store(d, Location::Register(RAX.code()));
        }

        Ok(())
    }

    /// Compile a checked binary operation into a `Location`.
    fn c_checked_binop(&mut self, binop: &BinOp, op1: &Operand, op2: &Operand) -> Location {
        // Move `op1` into `dest`.
        let dest_loc = match op1 {
            Operand::Place(p) => self.c_place(&p),
            Operand::Constant(Constant::Int(ci)) => self.c_constint(&ci),
            Operand::Constant(Constant::Bool(_b)) => unreachable!(),
            Operand::Constant(c) => todo!("{}", c),
        };
        let dest = dest_loc.unwrap_reg();
        // Add together `dest` and `op2`.
        match op2 {
            Operand::Place(p) => {
                let (rloc, _) = self.place_to_location(&p, false);
                match binop {
                    BinOp::Add => self.c_checked_add_place(dest, &rloc),
                    _ => todo!(),
                }
                self.free_if_temp(rloc);
            }
            Operand::Constant(Constant::Int(ci)) => match binop {
                BinOp::Add => self.c_checked_add_const(dest, ci),
                _ => todo!(),
            },
            Operand::Constant(Constant::Bool(_b)) => todo!(),
            Operand::Constant(c) => todo!("{}", c),
        };
        // In the future this will set the overflow flag of the tuple in `lloc`, which will be
        // checked by a guard, allowing us to return from the trace more gracefully.
        dynasm!(self.asm
            ; jc ->crash
        );
        dest_loc
    }

    // FIXME Use a macro to generate funcs for all of the different binary operations.
    // Code-gen the addition of a `Location` to the value in the register `dest_reg`.
    fn c_checked_add_place(&mut self, dest_reg: u8, src_loc: &Location) {
        match src_loc {
            Location::Register(reg) => {
                dynasm!(self.asm
                    ; add Rq(dest_reg), Rq(reg)
                );
            }
            Location::Mem(ro) => {
                dynasm!(self.asm
                    ; add Rq(dest_reg), [Rq(ro.reg) + ro.offs]
                );
            }
            _ => unreachable!(),
        }
    }

    // Code-gen the addition of a constant integer to the value in the register `dest_reg`.
    fn c_checked_add_const(&mut self, dest_reg: u8, src_const: &ConstantInt) {
        let c_val = src_const.i64_cast();
        if c_val <= u32::MAX.into() {
            dynasm!(self.asm
                ; add Rq(dest_reg), c_val as u32 as i32
            );
        } else {
            dynasm!(self.asm
                ; mov rax, QWORD c_val
                ; add Rq(dest_reg), rax
            );
        }
    }

    /// Compile a TIR statement.
    fn c_statement(&mut self, stmt: &Statement) -> Result<(), CompileError> {
        match stmt {
            Statement::Assign(l, r) => {
                let rloc = match r {
                    Rvalue::Use(Operand::Place(p)) => self.c_place(p),
                    Rvalue::Use(Operand::Constant(c)) => match c {
                        Constant::Int(ci) => self.c_constint(ci),
                        Constant::Bool(b) => self.c_bool(*b),
                        c => todo!("{}", c),
                    },
                    Rvalue::CheckedBinaryOp(binop, op1, op2) => {
                        let rloc = self.c_checked_binop(binop, op1, op2);
                        // FIXME deal with overflow
                        let mut l = l.clone();
                        l.projection.push(Projection::Field(0));
                        // FIXME dedup code
                        self.store(&l, rloc.clone());
                        self.free_if_temp(rloc);
                        return Ok(());
                    }
                    Rvalue::Ref(p) => self.c_ref(p),
                    Rvalue::Len(p) => self.c_len(p),
                    unimpl => todo!("{}", unimpl),
                };
                self.store(&l, rloc.clone());
                self.free_if_temp(rloc);
            }
            Statement::Enter(_, args, _dest, off) => self.c_enter(args, *off),
            Statement::Leave => {}
            Statement::StorageDead(l) => self.free_register(l)?,
            Statement::Call(target, args, dest) => self.c_call(target, args, dest)?,
            Statement::Nop => {}
            Statement::Unimplemented(s) => todo!("{:?}", s),
        }

        Ok(())
    }

    /// Store the value in `src_loc` into `dest_plc`.
    fn store(&mut self, dest_plc: &Place, src_loc: Location) {
        let (dest_loc, ty) = self.place_to_location(dest_plc, true);
        match (&dest_loc, &src_loc) {
            (Location::Addr(dest_reg), Location::Register(src_reg)) => {
                // If the lhs is a projection that results in a memory address (e.g.
                // `(*$1).0`), then the value in `dest_reg` is a pointer to store into.
                match ty.size() {
                    0 => (), // ZST.
                    1 => {
                        dynasm!(self.asm
                            ; mov [Rq(dest_reg)], Rb(src_reg)
                        );
                    }
                    2 => {
                        dynasm!(self.asm
                            ; mov [Rq(dest_reg)], Rw(src_reg)
                        );
                    }
                    4 => {
                        dynasm!(self.asm
                            ; mov [Rq(dest_reg)], Rd(src_reg)
                        );
                    }
                    8 => {
                        dynasm!(self.asm
                            ; mov [Rq(dest_reg)], Rq(src_reg)
                        );
                    }
                    _ => unreachable!(),
                }
            }
            (Location::Register(dest_reg), Location::Register(src_reg)) => {
                dynasm!(self.asm
                    ; mov Rq(dest_reg), Rq(src_reg)
                );
            }
            (Location::Mem(dest_ro), Location::Register(src_reg)) => {
                match ty.size() {
                    0 => (), // ZST.
                    1 => {
                        dynasm!(self.asm
                            ; mov BYTE [Rq(dest_ro.reg) + dest_ro.offs], Rb(src_reg)
                        );
                    }
                    2 => {
                        dynasm!(self.asm
                            ; mov WORD [Rq(dest_ro.reg) + dest_ro.offs], Rw(src_reg)
                        );
                    }
                    4 => {
                        dynasm!(self.asm
                            ; mov DWORD [Rq(dest_ro.reg) + dest_ro.offs], Rd(src_reg)
                        );
                    }
                    8 => {
                        dynasm!(self.asm
                            ; mov QWORD [Rq(dest_ro.reg) + dest_ro.offs], Rq(src_reg)
                        );
                    }
                    _ => unreachable!(),
                }
            }
            (Location::Mem(dest_ro), Location::Mem(src_ro)) => {
                if ty.size() <= 8 {
                    let temp = self.create_temporary();
                    match ty.size() {
                        0 => (), // ZST.
                        1 => {
                            dynasm!(self.asm
                                ; mov Rb(temp), BYTE [Rq(src_ro.reg) + src_ro.offs]
                                ; mov BYTE [Rq(dest_ro.reg) + dest_ro.offs], Rb(temp)
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; mov Rw(temp), WORD [Rq(src_ro.reg) + src_ro.offs]
                                ; mov WORD [Rq(dest_ro.reg) + dest_ro.offs], Rw(temp)
                            );
                        }
                        4 => {
                            dynasm!(self.asm
                                ; mov Rd(temp), DWORD [Rq(src_ro.reg) + src_ro.offs]
                                ; mov DWORD [Rq(dest_ro.reg) + dest_ro.offs], Rd(temp)
                            );
                        }
                        8 => {
                            dynasm!(self.asm
                                ; mov Rq(temp), QWORD [Rq(src_ro.reg) + src_ro.offs]
                                ; mov QWORD [Rq(dest_ro.reg) + dest_ro.offs], Rq(temp)
                            );
                        }
                        _ => unreachable!(),
                    }
                    self.free_if_temp(Location::Register(temp));
                } else {
                    self.copy_memory(dest_ro, src_ro, ty.size());
                }
            }
            //(Location::Register(dest_reg, dest_is_ptr), Location::Mem(src_ro)) => {
            (Location::Addr(dest_reg), Location::Mem(src_ro)) => {
                if ty.size() <= 8 {
                    let temp = self.create_temporary();
                    match ty.size() {
                        0 => (), // ZST.
                        1 => {
                            dynasm!(self.asm
                                ; mov Rb(temp), BYTE [Rq(src_ro.reg) + src_ro.offs]
                                ; mov BYTE [Rq(dest_reg)], Rb(temp)
                            );
                        }
                        2 => {
                            dynasm!(self.asm
                                ; mov Rw(temp), WORD [Rq(src_ro.reg) + src_ro.offs]
                                ; mov WORD [Rq(dest_reg)], Rw(temp)
                            );
                        }
                        4 => {
                            dynasm!(self.asm
                                ; mov Rd(temp), DWORD [Rq(src_ro.reg) + src_ro.offs]
                                ; mov DWORD [Rq(dest_reg)], Rd(temp)
                            );
                        }
                        8 => {
                            dynasm!(self.asm
                                ; mov Rq(temp), QWORD [Rq(src_ro.reg) + src_ro.offs]
                                ; mov QWORD [Rq(dest_reg)], Rq(temp)
                            );
                        }
                        _ => unreachable!(),
                    }
                    self.free_if_temp(Location::Register(temp));
                } else {
                    self.copy_memory(
                        &RegAndOffset {
                            reg: *dest_reg,
                            offs: 0,
                        },
                        src_ro,
                        ty.size(),
                    );
                }
            }
            (Location::Register(dest_reg), Location::Mem(src_ro)) => {
                match ty.size() {
                    0 => (), // ZST.
                    1 => {
                        dynasm!(self.asm
                            ; mov Rb(dest_reg), BYTE [Rq(src_ro.reg) + src_ro.offs]
                        );
                    }
                    2 => {
                        dynasm!(self.asm
                            ; mov Rw(dest_reg), WORD [Rq(src_ro.reg) + src_ro.offs]
                        );
                    }
                    4 => {
                        dynasm!(self.asm
                            ; mov Rd(dest_reg), DWORD [Rq(src_ro.reg) + src_ro.offs]
                        );
                    }
                    8 => {
                        dynasm!(self.asm
                            ; mov Rq(dest_reg), QWORD [Rq(src_ro.reg) + src_ro.offs]
                        );
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
        self.free_if_temp(dest_loc);
    }

    /// Compile a guard in the trace, emitting code to abort execution in case the guard fails.
    fn c_guard(&mut self, _grd: &Guard) {
        self.nop(); // FIXME compile guards
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

        // Make the TirTrace mutable so we can drain it into the TraceCompiler.
        let mut tt = tt;
        let mut tc = TraceCompiler::<TT> {
            asm: assembler,
            temp_regs: TEMP_REGS.iter().map(|r| r.code()).collect::<Vec<u8>>(),
            register_content_map: LOCAL_REGS
                .iter()
                .map(|r| (r.code(), RegAlloc::Free))
                .collect(),
            variable_location_map: HashMap::new(),
            trace_inputs_local: tt.inputs().map(|t| t.clone()),
            local_decls: tt.local_decls.clone(),
            stack_builder: StackBuilder::default(),
            addr_map: tt.addr_map.drain().into_iter().collect(),
            _pd: PhantomData,
        };

        tc.init();

        for i in 0..tt.len() {
            let res = match unsafe { tt.op(i) } {
                TirOp::Statement(st) => tc.c_statement(st),
                TirOp::Guard(g) => Ok(tc.c_guard(g)),
            };

            // FIXME -- Later errors should not be fatal. We should be able to abort trace
            // compilation and carry on.
            match res {
                Ok(_) => (),
                Err(e) => tc.crash_dump(Some(e)),
            }
        }

        // Make sure we didn't forget to free some temporaries.
        assert!(tc.check_temporaries());
        tc.ret();
        tc
    }

    /// Returns a pointer to the static symbol `sym`, or an error if it cannot be found.
    fn find_symbol(sym: &str) -> Result<*mut c_void, CompileError> {
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
    use super::{
        CompileError, HashMap, Local, Location, RegAlloc, TraceCompiler, LOCAL_REGS, TEMP_REGS,
    };
    use crate::stack_builder::StackBuilder;
    use dynasmrt::Register;
    use fm::FMBuilder;
    use libc::{abs, c_void, getuid};
    use regex::Regex;
    use std::marker::PhantomData;
    use yktrace::sir::SIR;
    use yktrace::tir::TirTrace;
    use yktrace::{start_tracing, trace_inputs, TracingKind};

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
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = simple();
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
            register_content_map: LOCAL_REGS
                .iter()
                .cloned()
                .map(|r| (r.code(), RegAlloc::Free))
                .collect(),
            temp_regs: TEMP_REGS.iter().map(|r| r.code()).collect::<Vec<u8>>(),
            variable_location_map: HashMap::new(),
            trace_inputs_local: None,
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
            register_content_map: LOCAL_REGS
                .iter()
                .cloned()
                .map(|r| (r.code(), RegAlloc::Free))
                .collect(),
            temp_regs: TEMP_REGS.iter().map(|r| r.code()).collect::<Vec<u8>>(),
            variable_location_map: HashMap::new(),
            trace_inputs_local: None,
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

    #[inline(never)]
    fn fcall() -> u8 {
        let y = farg(13); // assigns 13 to $1
        let _z = farg(14); // overwrites $1 within the call
        y // returns $1
    }

    #[test]
    fn test_function_call_simple() {
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = fcall();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
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
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = fnested();
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
        let th = start_tracing(TracingKind::HardwareTracing);
        let _ = unsafe { add6(1, 1, 1, 1, 1, 1) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        assert_tir(
            "...\n\
            ops:\n\
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
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = unsafe { getuid() };
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
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let v = -56;
        inputs.0 = unsafe { abs(v) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, args.0);
    }

    /// The same as `exec_call_symbol_args_with_rv`, just using a constant argument.
    #[test]
    fn exec_call_symbol_with_const_arg() {
        struct IO(i32);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = unsafe { abs(-123) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn exec_call_symbol_with_many_args() {
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = unsafe { add6(1, 2, 3, 4, 5, 6) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, 21);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn exec_call_symbol_with_many_args_some_ignored() {
        extern "C" {
            fn add_some(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
        }

        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = unsafe { add_some(1, 2, 3, 4, 5) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
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

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_spilling_simple() {
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = many_locals();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 7);
        assert_eq!(spills, 3); // Three u8s.
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

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_spilling_u64() {
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = spill_u64();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
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

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_mov_register_to_stack() {
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = register_to_stack(8);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 8);
        assert_eq!(spills, 9); // f, g: i32, h:  u8.
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

    #[ignore] // FIXME: It has become hard to test spilling.
    #[test]
    fn test_mov_stack_to_register() {
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = stack_to_register();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let (ct, spills) = TraceCompiler::<IO>::test_compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 3);
        assert_eq!(spills, 1); // Just one u8.
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
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = ext_call();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let mut args = IO(0);
        TraceCompiler::<IO>::compile(tir_trace).execute(&mut args);
        assert_eq!(inputs.0, 7);
        assert_eq!(inputs.0, args.0);
    }

    #[test]
    fn test_trace_inputs() {
        struct IO(u64, u64, u64);
        let mut inputs = trace_inputs(IO(1, 2, 3));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = unsafe { add6(inputs.0, inputs.1, inputs.2, 4, 5, 6) };
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(1, 2, 3);
        ct.execute(&mut args);
        assert_eq!(inputs.0, 21);
        assert_eq!(inputs.0, args.0);
        // Execute once more with different arguments.
        let mut args2 = IO(7, 8, 9);
        ct.execute(&mut args2);
        assert_eq!(args2.0, 39);
    }

    #[inline(never)]
    fn add(a: u8) -> u8 {
        let x = a + 3; // x = a; add x, 3
        let y = a + x;
        y
    }

    fn add64(a: u64) -> u64 {
        let x = a + 8589934592;
        x
    }

    #[test]
    fn test_binop_add() {
        struct IO(u8, u64, u8, u8);
        let mut inputs = trace_inputs(IO(0, 0, 0, 0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = add(13);
        inputs.1 = add64(1);
        inputs.2 = inputs.0 + 2;
        inputs.3 = inputs.0 + inputs.0;
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0, 0, 0, 0);
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
        struct IO(u8, u64);
        let mut inputs = trace_inputs(IO(0, 0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _d = 6;
        inputs.0 = add(13);
        inputs.1 = add64(1);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0, 0);
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

        struct IO(());
        let _ = trace_inputs(IO(()));
        let th = start_tracing(TracingKind::HardwareTracing);
        let s = S { _x: 100, y: 200 };
        let _expect = get_y(s);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();

        // %s1: Initial s in the outer function
        // %s2: A copy of s. Uninteresting.
        // %s3: s inside the function.
        // %res: the result of the call.
        assert_tir("
            local_decls:
              ...
              %s1: (%cgu, %tid1) => StructTy { offsets: [0, 8], tys: [(%cgu, %tid2), (%cgu, %tid2)], align: 8, size: 16 }
              ...
              %res: (%cgu, %tid2) => u64
              ...
              %s2: (%cgu, %tid1)...
              ...
              %s3: (%cgu, %tid1)...
              ...
            ops:
              ...
              (%s1).0 = 100u64
              (%s1).1 = 200u64
              ...
              %s2 = %s1
              ...
              enter(...
              ...
              %res = (%s3).1
              ...
              leave
              ...", &tir_trace);
    }

    fn ref_deref() -> u64 {
        let mut x = 9;
        let y = &mut x;
        *y = 10;
        let z = *y;
        z
    }

    #[test]
    #[ignore] // FIXME: Need to type our projections.
    fn test_ref_deref() {
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = ref_deref();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 10);
    }

    #[test]
    #[ignore] // FIXME: Need to type our projections.
    fn test_ref_deref_stack() {
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        let _e = 5;
        let _f = 6;
        inputs.0 = ref_deref();
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 10);
    }

    fn deref1(arg: u64) -> u64 {
        let a = &arg;
        return *a;
    }

    #[test]
    fn test_deref_stack_to_register() {
        // This test dereferences a variable that lives on the stack and stores it in a register.
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let _a = 1;
        let _b = 2;
        let _c = 3;
        let f = 6;
        inputs.0 = deref1(f);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 6);
    }

    fn deref2(arg: u64) -> u64 {
        let a = &arg;
        let _b = 2;
        let _c = 3;
        let _d = 4;
        return *a;
    }

    #[test]
    fn test_deref_register_to_stack() {
        // This test dereferences a variable that lives on the stack and stores it in a register.
        struct IO(u64);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let f = 6;
        inputs.0 = deref2(f);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 6);
    }

    #[do_not_trace]
    fn dont_trace_this(a: u8) -> u8 {
        let b = 2;
        let c = a + b;
        c
    }

    #[test]
    fn test_do_not_trace() {
        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.0 = dont_trace_this(1);
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
        let mut args = IO(0);
        ct.execute(&mut args);
        assert_eq!(args.0, 3);
    }

    fn dont_trace_stdlib(a: &mut Vec<u64>) -> u64 {
        a.push(3);
        3
    }

    #[test]
    fn test_do_not_trace_stdlib() {
        let mut vec: Vec<u64> = Vec::new();
        struct IO<'a>(&'a mut Vec<u64>);
        let inputs = trace_inputs(IO(&mut vec));
        let th = start_tracing(TracingKind::HardwareTracing);
        let v = inputs.0;
        dont_trace_stdlib(v);
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
        #[derive(Debug, PartialEq)]
        struct S {
            x: usize,
            y: usize,
        }
        let s = S { x: 5, y: 6 };
        let t = (1usize, 2u8, 3usize);
        struct IO((usize, u8, usize), u8, S, usize);
        let mut inputs = trace_inputs(IO(t, 0u8, s, 0usize));
        let th = start_tracing(TracingKind::HardwareTracing);
        inputs.1 = (inputs.0).1;
        inputs.3 = inputs.2.y;
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);

        let t2 = (1usize, 2u8, 3usize);
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
        let t = (1u8, 2u8);
        struct IO((u8, u8), u8);
        let mut inputs = trace_inputs(IO(t, 3u8));
        let th = start_tracing(TracingKind::HardwareTracing);
        (inputs.0).1 = inputs.1;
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let t2 = (1u8, 2u8);
        let mut args = IO(t2, 3u8);
        ct.execute(&mut args);
        assert_eq!((args.0).1, 3);
    }

    #[inline(never)]
    fn array(a: &mut [u8; 3]) -> u8 {
        let z = a[1];
        z
    }

    #[test]
    fn test_array() {
        struct IO<'a>(&'a mut [u8; 3], u8);
        let mut a = [3u8, 4u8, 5u8];
        let mut inputs = trace_inputs(IO(&mut a, 0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let tmp = inputs.0;
        inputs.1 = array(tmp);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut a2 = [3u8, 4u8, 5u8];
        let mut args = IO(&mut a2, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, 4);
    }

    /// Test codegen of field access on a struct ref on the right-hand side.
    #[test]
    fn rhs_struct_ref_field() {
        fn add1(io: &mut IO) -> u8 {
            io.0 + 1
        }

        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let x = add1(&mut inputs);
        inputs.0 = x;
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
        // FIXME return value not necessary, but we can't codegen returning nothing just yet.
        fn set100(io: &mut IO) -> u8 {
            io.0 = 100;
            0
        }

        struct IO(u8);
        let mut inputs = trace_inputs(IO(0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let _ = set100(&mut inputs);
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
        // FIXME return value not necessary, but we can't codegen returning nothing just yet.
        fn ten(io: &mut IO) -> u8 {
            io.0 = S(10, 10, 10);
            0
        }

        #[derive(Debug, Eq, PartialEq)]
        struct S(u64, u64, u64);
        struct IO(S);

        let mut inputs = trace_inputs(IO(S(0, 0, 0)));
        let th = start_tracing(TracingKind::HardwareTracing);
        let _ = ten(&mut inputs);
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);

        let mut args = IO(S(1, 1, 1));
        ct.execute(&mut args);
        assert_eq!(args.0, S(10, 10, 10));
    }

    #[test]
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

        let a = "abc".as_bytes();
        let mut inputs = trace_inputs(IO(&a, 0));
        let th = start_tracing(TracingKind::HardwareTracing);
        let x = matchthis(&inputs, 0);
        inputs.1 = x;
        let sir_trace = th.stop_tracing().unwrap();
        let tir_trace = TirTrace::new(&*SIR, &*sir_trace).unwrap();
        let ct = TraceCompiler::<IO>::compile(tir_trace);
        let mut a2 = "abc".as_bytes();
        let mut args = IO(&mut a2, 0);
        ct.execute(&mut args);
        assert_eq!(args.1, 1);
    }
}
