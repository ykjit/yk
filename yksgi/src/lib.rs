use libc::dlsym;
use libffi::middle::{arg as ffi_arg, Arg as FFIArg, Builder as FFIBuilder, CodePtr as FFICodePtr};
use llvm_sys::core::*;
use llvm_sys::target::{LLVMABISizeOfType, LLVMOffsetOfElement};
use llvm_sys::{LLVMOpcode, LLVMTypeKind};
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::ffi::{c_void, CStr};
use std::ptr;
use std::slice;

mod llvmbridge;
use llvmbridge::{get_aot_original, BasicBlock, Module, Type, Value};

static ALWAYS_SKIP_FUNCS: [&str; 5] = [
    "llvm.lifetime",
    "llvm.lifetime.end.p0i8",
    "llvm.dbg.value",
    "llvm.dbg.declare",
    "fprintf",
];
static SKIP_FOR_NOW_FUNCS: [&str; 2] = ["yk_location_drop", "yk_mt_drop"];
static YK_CONTROL_POINT_FUNC: &str = "__ykrt_control_point";

/// Active frames (basic block index, instruction index, function name) in the AOTModule where the
/// guard failure occured. Mirrors the struct defined in ykllvmwrap/jitmodbuilder.cc.
#[derive(Debug)]
#[repr(C)]
pub struct FrameInfo {
    pub bbidx: usize,
    pub instridx: usize,
    pub fname: *const i8,
}

/// Stopgap interpreter values.
#[derive(Clone, Copy, PartialEq)]
pub struct SGValue {
    pub val: u64,
    pub ty: Type,
}

impl SGValue {
    pub fn new(val: u64, ty: Type) -> Self {
        SGValue { val, ty }
    }

    pub fn with_type(&self, ty: Type) -> Self {
        SGValue::new(self.val, ty)
    }
}

/// A frame holding live variables.
struct Frame {
    vars: HashMap<Value, SGValue>,
    pc: Value,
}

impl Frame {
    fn new(pc: Value) -> Frame {
        Frame {
            vars: HashMap::new(),
            pc,
        }
    }

    /// Get the value of the variable `key` in this frame.
    fn get(&self, key: &Value) -> Option<&SGValue> {
        self.vars.get(key)
    }

    /// Add new variable `key` with value `val`.
    fn add(&mut self, key: Value, val: SGValue) {
        self.vars.insert(key, val);
    }
}

// FIXME: Add unit tests.
macro_rules! signext {
    ( $name: ident, $ty:ty ) => {
        fn $name(v: $ty, dstw: u32) -> u64 {
            // Casting a smaller signed integer to a bigger unsigned integer using `as`
            // automatically sign extends in Rust.
            match dstw {
                8 => unreachable!(), // cast from i1 always results in -1 or 0
                16 => (v as u16) as u64,
                32 => (v as u32) as u64,
                64 => v as u64,
                _ => unreachable!(),
            }
        }
    };
}

signext!(sext_i8, i8);
signext!(sext_i16, i16);
signext!(sext_i32, i32);

// This return value tells the caller of the control point that a guard failure has happened but
// the stopgap interpreter managed to recover and reach the state just before the control point
// again.
const TRACE_GUARDFAIL_CONTINUE: u8 = 1;
// Same as above, but means that a guard failure has occured and the stopgap interpreter has exited
// the main interpreter loop and interpreted the calling function's return.
const TRACE_GUARDFAIL_RETURN: u8 = 2;

/// The stopgap interpreter. Used during guard failures to get back to the control point by
/// interpreting LLVM IR.
pub struct SGInterp {
    /// LLVM IR module we are interpreting.
    module: Module,
    /// Current frames.
    frames: Vec<Frame>,
    /// Current instruction being interpreted.
    pc: Value,
    /// Last basic block that was interpreted.
    lastbb: Option<BasicBlock>,
}

impl SGInterp {
    /// Create a new stopgap interpreter and initialise it to start interpretation at the location
    /// given by a basic block index, instruction index, and function name.
    /// FIXME: Support initialisation of multiple frames.
    pub unsafe fn new(activeframes: &[FrameInfo]) -> SGInterp {
        // Get AOT module IR and parse it.
        let module = Module::from_bc();

        // Initialise frames.
        let mut frames = Vec::with_capacity(activeframes.len());
        for frame in activeframes {
            let funcname = std::ffi::CStr::from_ptr(frame.fname);
            let func = module.function(funcname.as_ptr());
            let bb = func.bb(frame.bbidx);
            let instr = bb.instruction(frame.instridx);
            frames.push(Frame::new(instr));
        }
        // Create and initialise stop gap interpreter.
        let current_pc = frames.last().unwrap().pc;
        SGInterp {
            module,
            frames,
            pc: current_pc,
            lastbb: None,
        }
    }

    /// Add a live variable and its value to the current frame.
    pub fn var_init(
        &mut self,
        bbidx: usize,
        instridx: usize,
        fname: &CStr,
        sfidx: usize,
        val: u64,
    ) {
        let func = self.module.function(fname.as_ptr());
        let bb = func.bb(bbidx);
        let instr = bb.instruction(instridx);
        let orgaot = if sfidx == 0 {
            unsafe { get_aot_original(&instr) }
        } else {
            // Only the root stackframe contains the control point call, so for the other frames
            // there's no need to match live variables to their corresponding variables passed into
            // the control point. See `get_aot_original` for more details.
            None
        };
        let ty = instr.get_type();
        let value = SGValue::new(val, ty);
        self.frames.get_mut(sfidx).unwrap().add(instr, value);
        if let Some(v) = orgaot {
            self.frames.get_mut(sfidx).unwrap().add(v, value);
        }
    }

    /// Lookup the value of variable `var` in the current frame.
    fn var_lookup(&self, var: &Value) -> SGValue {
        if var.is_instruction() || var.is_argument() {
            *self.frames.last().unwrap().get(var).unwrap()
        } else if var.is_constant() {
            llvmbridge::llvm_const_to_sgvalue(*var)
        } else {
            // GlobalVariable, Function, etc.
            todo!()
        }
    }

    /// Set the value of variable `var` in the current frame.
    fn var_set(&mut self, var: Value, val: SGValue) {
        if var.is_instruction() {
            self.frames.last_mut().unwrap().add(var, val);
        } else {
            // GlobalVariable, Function, etc.
            todo!()
        }
    }

    /// Interpret LLVM IR from the interpreters initialised position. Returns true if the control
    /// point was reached, or false if the interpreter left the main interpreter loop.
    pub unsafe fn interpret(&mut self) -> u8 {
        // We start interpretation at the branch instruction that was turned into a guard. We
        // need to re-interpret this instruction in order to find out which branch we need to
        // follow.
        loop {
            match self.pc.opcode() {
                LLVMOpcode::LLVMAdd => self.add(),
                LLVMOpcode::LLVMBitCast => self.bitcast(),
                LLVMOpcode::LLVMBr => {
                    self.branch();
                    continue;
                }
                LLVMOpcode::LLVMCall => {
                    if self.call() {
                        // We've reached the control point again, so its safe to continue with the
                        // fake trace stitching loop in mt.rs.
                        return TRACE_GUARDFAIL_CONTINUE;
                    }
                }
                LLVMOpcode::LLVMGetElementPtr => self.gep(),
                LLVMOpcode::LLVMICmp => self.icmp(),
                LLVMOpcode::LLVMLoad => self.load(),
                LLVMOpcode::LLVMPHI => self.phi(),
                LLVMOpcode::LLVMPtrToInt => self.ptrtoint(),
                LLVMOpcode::LLVMRet => {
                    if self.ret() {
                        // We've interpreted the return of the control point's caller, so we need
                        // to tell our caller it also needs to return when the control point
                        // returns.
                        return TRACE_GUARDFAIL_RETURN;
                    }
                }
                LLVMOpcode::LLVMSExt => self.sext(),
                LLVMOpcode::LLVMStore => self.store(),
                LLVMOpcode::LLVMSwitch => {
                    self.switch();
                    continue;
                }
                LLVMOpcode::LLVMSub => self.sub(),
                LLVMOpcode::LLVMSelect => self.select(),
                _ => todo!("{:?}", self.pc.as_str()),
            }
            self.pc = Value::new(LLVMGetNextInstruction(self.pc.get()));
        }
    }

    fn select(&mut self) {
        let cond = self.pc.get_operand(0);
        let op_true = self.pc.get_operand(1);
        let op_false = self.pc.get_operand(2);

        if op_true.get_type().is_vector() || op_false.get_type().is_vector() {
            todo!(
                "vector switches are not implemented: {:?}",
                self.pc.as_str()
            );
        }
        debug_assert!(cond.get_type().get_int_width() == 1); // Only `i1` is valid.

        let chosen = if self.var_lookup(&cond).val == 1 {
            op_true
        } else {
            op_false
        };
        self.var_set(self.pc, self.var_lookup(&chosen));
    }

    // FIMXE: generalise binary operators.
    fn add(&mut self) {
        // FIXME: Handle overflows.
        let op1 = self.pc.get_operand(0);
        let op2 = self.pc.get_operand(1);
        let val1 = self.var_lookup(&op1);
        let val2 = self.var_lookup(&op2);
        let res = val1.val.checked_add(val2.val).unwrap();
        self.var_set(self.pc, SGValue::new(res, self.pc.get_type()));
    }

    fn sub(&mut self) {
        // FIXME: Handle underflows.
        let op1 = self.pc.get_operand(0);
        let op2 = self.pc.get_operand(1);
        let val1 = self.var_lookup(&op1);
        let val2 = self.var_lookup(&op2);
        let res = val1.val.checked_sub(val2.val).unwrap();
        self.var_set(self.pc, SGValue::new(res, self.pc.get_type()));
    }

    /// Cast value to new type of same bit size (without changing any bits).
    fn bitcast(&mut self) {
        // Since we store all integer values as u64 and evaluate them lazily according
        // to the attached type, this operation is a NOP.
        let src = self.pc.get_operand(0);
        let dstty = self.pc.get_type();
        let val = self.var_lookup(&src);
        self.var_set(self.pc, val.with_type(dstty));
    }

    /// Interpret branch instruction `instr`.
    unsafe fn branch(&mut self) {
        debug_assert!(self.pc.is_br());
        let succ = if LLVMIsConditional(self.pc.get()) == 0 {
            BasicBlock::new(LLVMGetSuccessor(self.pc.get(), 0))
        } else {
            let cond = Value::new(LLVMGetCondition(self.pc.get()));
            debug_assert!(cond.get_type().is_integer());
            debug_assert!(cond.get_type().get_int_width() == 1);
            let val = self.var_lookup(&cond);
            if val.val == 1 {
                BasicBlock::new(LLVMGetSuccessor(self.pc.get(), 0))
            } else {
                BasicBlock::new(LLVMGetSuccessor(self.pc.get(), 1))
            }
        };
        self.lastbb = Some(BasicBlock::new(LLVMGetInstructionParent(self.pc.get())));
        self.pc = succ.first();
    }

    /// Implement call instructions. Returns `true` if the control point has been reached,
    /// `false` otherwise.
    unsafe fn call(&mut self) -> bool {
        let func = self.pc.get_called_value();
        if func.is_inline_asm() {
            // FIXME: Implement calls to inline asm. Just skip them for now, as our tests won't run
            // otherwise.
            return false;
        }
        let name = func.get_name();
        if ALWAYS_SKIP_FUNCS.contains(&name) {
            // There's no point calling these functions inside the stopgap interpreter so just skip
            // them.
            false
        } else if SKIP_FOR_NOW_FUNCS.contains(&name) {
            // FIXME: These need to run, but since we can't do calls, just skip them for now.
            false
        } else if name.starts_with("puts") || name.starts_with("printf") {
            // FIXME: Until we can handle function calls, simulate prints to make our tests work.
            // Get format string.
            let op = self.pc.get_operand(0);
            let op2 = op.get_operand(0);
            let op3 = LLVMGetInitializer(op2.get());
            let mut l = 0;
            let s = LLVMGetAsString(op3, &mut l);
            // Get operands
            let mut ops = Vec::new();
            for i in 1..LLVMGetNumOperands(self.pc.get()) - 1 {
                let op = self.pc.get_operand(u32::try_from(i).unwrap());
                let val = self.var_lookup(&op);
                ops.push(val.val);
            }
            // FIXME: Hack in some printf calls with different arguments to improve our testing
            // capabilities. Replace with a more dynamic approach once we've figured out how to do
            // so.
            match ops.len() {
                0 => libc::printf(s),
                1 => libc::printf(s, ops[0]),
                2 => libc::printf(s, ops[0], ops[1]),
                3 => libc::printf(s, ops[0], ops[1], ops[2]),
                _ => todo!(),
            };
            false
        } else if name.starts_with(YK_CONTROL_POINT_FUNC) {
            // FIXME: When we see the control point we are done and can just return.
            true
        } else {
            // We are going to do a native call via libffi, starting with looking up the address of
            // the callee in the virtual address space.
            //
            // OPT: This could benefit from caching, and we may already know the address of the
            // callee if we inlined it when preparing traces:
            // https://github.com/ykjit/yk/issues/544
            debug_assert!(func.is_function());
            let fptr = dlsym(ptr::null_mut(), name.as_ptr() as *const i8);
            if fptr == ptr::null_mut() {
                todo!("couldn't find symbol: {}", name);
            }

            if func.is_vararg_function() {
                todo!("calling a varargs function");
            }

            // Now build the calling interface and collect the values of the callee's arguments.
            let mut builder = FFIBuilder::new();
            let num_args = usize::try_from(self.pc.get_num_arg_operands()).unwrap();
            let arg_lay = Layout::array::<u64>(num_args).unwrap();
            let arg_mem = alloc(Layout::array::<u64>(num_args).unwrap());
            let mut arg_mem_p = arg_mem.cast::<u64>();
            for i in 0..num_args {
                // The first N operands are the first N operands of the function being called.
                let arg = self.pc.get_operand(u32::try_from(i).unwrap());
                builder = builder.arg(arg.get_type().ffi_type());
                *arg_mem_p = self.var_lookup(&arg).val;
                arg_mem_p = arg_mem_p.add(1);
            }
            builder = builder.res(self.pc.get_type().ffi_type());
            let cif = builder.into_cif(); // OPT: cache CIFs for repeated calls to same func sig.

            // Actually do the call.
            let ret_ty = self.pc.get_type();
            let arg_slice = slice::from_raw_parts_mut::<u64>(arg_mem as *mut u64, num_args);
            let ffi_arg_vals: Vec<FFIArg> = arg_slice.iter().map(|a| ffi_arg(a)).collect();
            if ret_ty.is_integer() {
                // FIXME: https://github.com/ykjit/yk/issues/536
                match ret_ty.get_int_width() {
                    32 => {
                        let rv = cif.call::<u32>(FFICodePtr(fptr), &ffi_arg_vals) as u64;
                        self.var_set(self.pc, SGValue::new(rv, ret_ty));
                    }
                    _ => todo!(),
                }
            } else if ret_ty.is_void() {
                cif.call::<()>(FFICodePtr(fptr), &ffi_arg_vals);
            } else {
                todo!("{:?}", ret_ty.as_str());
            };
            // `ffi_arg_vals` contains raw pointers to `arg_mem`, so we had better drop those
            // before we deallocate `arg_mem`.
            drop(ffi_arg_vals);
            dealloc(arg_mem, arg_lay);
            return false;
        }
    }

    unsafe fn gep(&mut self) {
        // FIXME: If the target is a `struct` and it's packed, then we need to get the data layout
        // to compute the correct offsets.
        let ty = self.pc.get_type();
        debug_assert!(ty.is_pointer());

        let layout = self.module.datalayout();
        let aggr = self.pc.get_operand(0);
        let mut curty = aggr.get_type();
        let ptr = self.var_lookup(&aggr).val;
        let numops = LLVMGetNumOperands(self.pc.get());

        let mut offset = 0;
        for i in 1..numops {
            let op = self.pc.get_operand(u32::try_from(i).unwrap());
            let idx = if op.is_constant() {
                LLVMConstIntGetZExtValue(op.get())
            } else {
                self.var_lookup(&op).val
            };

            if curty.is_struct() {
                let off = LLVMOffsetOfElement(layout, curty.get(), u32::try_from(idx).unwrap());
                offset += off;
                let nty = LLVMStructGetTypeAtIndex(curty.get(), u32::try_from(idx).unwrap());
                curty = Type::new(nty);
            } else {
                // Pointer into an array
                curty = curty.get_element_type();
                let size = LLVMABISizeOfType(layout, curty.get());
                offset += size * idx;
            }
        }
        debug_assert!(curty.get() == ty.get_element_type().get());
        let newval = SGValue::new(ptr + offset, ty);
        self.var_set(self.pc, newval);
    }

    unsafe fn icmp(&mut self) {
        let op1 = self.pc.get_operand(0);
        let op2 = self.pc.get_operand(1);
        let val1 = self.var_lookup(&op1);
        let val2 = self.var_lookup(&op2);
        let b = match LLVMGetICmpPredicate(self.pc.get()) {
            llvm_sys::LLVMIntPredicate::LLVMIntULT => val1.val < val2.val,
            llvm_sys::LLVMIntPredicate::LLVMIntUGT => val1.val > val2.val,
            llvm_sys::LLVMIntPredicate::LLVMIntEQ => val1.val == val2.val,
            llvm_sys::LLVMIntPredicate::LLVMIntNE => val1.val != val2.val,
            llvm_sys::LLVMIntPredicate::LLVMIntSGT => match val1.ty.get_int_width() {
                32 => val1.val as i32 > val2.val as i32,
                _ => todo!(),
            },
            llvm_sys::LLVMIntPredicate::LLVMIntSLT => match val1.ty.get_int_width() {
                32 => (val1.val as i32) < (val2.val as i32),
                _ => todo!(),
            },
            _ => todo!(),
        };
        self.var_set(self.pc, SGValue::new(b as u64, self.pc.get_type()));
    }

    unsafe fn load(&mut self) {
        let op = self.pc.get_operand(0);
        let ty = self.pc.get_type();
        let val = self.var_lookup(&op);
        debug_assert!(val.ty.is_pointer());
        let newval = match ty.kind() {
            LLVMTypeKind::LLVMIntegerTypeKind => {
                let width = LLVMGetIntTypeWidth(ty.get());
                match width {
                    8 => ptr::read::<u8>(val.val as *const u8) as u64,
                    16 => ptr::read::<u16>(val.val as *const u16) as u64,
                    32 => ptr::read::<u32>(val.val as *const u32) as u64,
                    64 => ptr::read::<u64>(val.val as *const u64),
                    _ => unreachable!(),
                }
            }
            LLVMTypeKind::LLVMPointerTypeKind => ptr::read::<u64>(val.val as *const u64),
            _ => todo!(),
        };
        self.var_set(self.pc, SGValue::new(newval, ty));
    }

    unsafe fn phi(&mut self) {
        let num = LLVMCountIncoming(self.pc.get());
        for i in 0..num {
            let block = LLVMGetIncomingBlock(self.pc.get(), i);
            if block == self.lastbb.as_ref().unwrap().get() {
                let val = Value::new(LLVMGetIncomingValue(self.pc.get(), i));
                let sgval = self.var_lookup(&val);
                self.var_set(self.pc, sgval);
                return;
            }
        }
        unreachable!();
    }

    /// Interpret return instruction `instr`.
    fn ret(&mut self) -> bool {
        // FIXME: Pass return value back to the control point.
        let numops = unsafe { LLVMGetNumOperands(self.pc.get()) };
        let retval = if numops == 0 {
            None
        } else {
            let op = self.pc.get_operand(0);
            Some(self.var_lookup(&op))
        };
        if self.frames.len() == 1 {
            // We've reached the end of the interpreters main, so just get the return value and
            // exit.
            // FIXME: This assumes that the bottom frame contains the interpreter loop. Is this
            // always the case?
            true
        } else {
            self.frames.pop();
            self.pc = self.frames.last_mut().unwrap().pc;
            debug_assert!(self.pc.is_call());
            if let Some(val) = retval {
                self.var_set(self.pc, val);
            }
            false
        }
    }

    fn ptrtoint(&mut self) {
        let src = self.pc.get_operand(0);
        let srcty = src.get_type();
        debug_assert!(srcty.is_pointer());
        let srcval = self.var_lookup(&src).val;
        self.var_set(self.pc, SGValue::new(srcval, self.pc.get_type()));
    }

    fn sext(&mut self) {
        let src = self.pc.get_operand(0);
        let srcty = src.get_type();
        if srcty.is_vector() {
            todo!()
        }
        let srcval = self.var_lookup(&src);
        let dstty = self.pc.get_type();
        let srcw = srcty.get_int_width();
        let dstw = dstty.get_int_width();
        debug_assert!(srcw < dstw);
        let dstval = match srcw {
            1 => todo!(),
            8 => sext_i8(srcval.val as i8, dstw),
            16 => sext_i16(srcval.val as i16, dstw),
            32 => sext_i32(srcval.val as i32, dstw),
            64 => srcval.val,
            _ => unreachable!(),
        };
        self.var_set(self.pc, SGValue::new(dstval, dstty));
    }

    unsafe fn store(&mut self) {
        let src = self.pc.get_operand(0);
        let dst = self.pc.get_operand(1);
        let srcval = self.var_lookup(&src);
        let dstval = self.var_lookup(&dst);
        let ty = src.get_type();
        match ty.kind() {
            LLVMTypeKind::LLVMIntegerTypeKind => {
                let width = LLVMGetIntTypeWidth(ty.get());
                // FIXME: Can this be generalised? LLVM allows non-byte-sized integer types, so
                // having a case for every possibility will be impractical:
                // https://github.com/ykjit/yk/issues/536
                match width {
                    1 => ptr::write(dstval.val as *mut u8, srcval.val as u8),
                    8 => ptr::write(dstval.val as *mut u8, srcval.val as u8),
                    32 => ptr::write(dstval.val as *mut u32, srcval.val as u32),
                    64 => ptr::write(dstval.val as *mut u64, srcval.val as u64),
                    _ => todo!(),
                }
            }
            LLVMTypeKind::LLVMPointerTypeKind => {
                ptr::write(
                    dstval.val as *mut *const c_void,
                    srcval.val as *const c_void,
                );
            }
            _ => todo!(),
        }
    }

    unsafe fn switch(&mut self) {
        debug_assert!(self.pc.is_switch());
        let cond = self.pc.get_operand(0);
        let val = self.var_lookup(&cond);
        let num_dests = LLVMGetNumSuccessors(self.pc.get());
        let mut succ = None;
        // Iterate over all switch cases to find a match.
        for i in 1..num_dests {
            // Skip the default case.
            let v = llvmbridge::llvm_const_to_sgvalue(self.pc.get_operand(i * 2));
            if v == val {
                succ = Some(BasicBlock::new(LLVMGetSuccessor(self.pc.get(), i)));
                break;
            }
        }
        if succ.is_none() {
            succ = Some(BasicBlock::new(LLVMGetSuccessor(self.pc.get(), 0)));
        }
        self.lastbb = Some(BasicBlock::new(LLVMGetInstructionParent(self.pc.get())));
        self.pc = succ.unwrap().first();
    }
}
