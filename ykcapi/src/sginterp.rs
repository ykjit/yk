use llvm_sys::bit_reader::LLVMParseBitcodeInContext2;
use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMModuleRef, LLVMValueRef};
use llvm_sys::{LLVMOpcode};
use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::MaybeUninit;

use crate::llvmapihelper::{self};
use crate::llvmwrap::{LocalVar};

/// Stopgap interpreter values.
#[derive(Debug)]
pub enum SGValue {
    U32(u32),
    U64(u64),
}

/// A frame holding live variables.
struct Frame {
    vars: HashMap<LocalVar, SGValue>,
}

impl Frame {
    fn new() -> Frame {
        Frame {
            vars: HashMap::new(),
        }
    }

    /// Get the value of the variable `key` in this frame.
    fn get(&self, key: &LocalVar) -> Option<&SGValue> {
        self.vars.get(key)
    }

    /// Insert new variable into this frame.
    fn add(&mut self, key: LocalVar, val: SGValue) {
        self.vars.insert(key, val);
    }
}

/// The stopgap interpreter. Used during guard failures to get back to the control point by
/// interpreting LLVM IR.
pub struct SGInterp {
    /// LLVM IR module we are interpreting.
    module: LLVMModuleRef,
    /// Current frames.
    frames: Vec<Frame>,
    /// Current instruction being interpreted.
    pc: LLVMValueRef,
}

impl SGInterp {
    /// Create a new stopgap interpreter and initialise it to start interpretation at the location
    /// given by a basic block index, instruction index, and function name.
    /// FIXME: Support initalisation of multiple frames.
    pub unsafe fn new(bbidx: u32, instridx: u32, fname: &CStr) -> SGInterp {
        // Get AOT module IR and parse it.
        let (addr, size) = ykutil::obj::llvmbc_section();
        let membuf = LLVMCreateMemoryBufferWithMemoryRange(
            addr as *const i8,
            size,
            "".as_ptr() as *const i8,
            0,
        );
        let context = LLVMContextCreate();
        let mut module: MaybeUninit<LLVMModuleRef> = MaybeUninit::uninit();
        let module = {
            LLVMParseBitcodeInContext2(context, membuf, module.as_mut_ptr());
            module.assume_init()
        };
        // Create and initialise stop gap interpreter.
        let func = LLVMGetNamedFunction(module, fname.as_ptr());
        let bb = llvmapihelper::get_basic_block(func, bbidx);
        let instr = llvmapihelper::get_instruction(bb, instridx);
        SGInterp {
            module,
            frames: vec![Frame::new()],
            pc: instr,
        }
    }

    /// Add a live variable and its value to the current frame.
    pub unsafe fn init_live(&mut self, bbidx: u32, instridx: u32, fname: &CStr, value: SGValue) {
        let func = LLVMGetNamedFunction(self.module, fname.as_ptr());
        let bb = llvmapihelper::get_basic_block(func, bbidx);
        let instr = llvmapihelper::get_instruction(bb, instridx);
        self.frames.last_mut().unwrap().add(LocalVar::new(instr), value);
    }

    /// Lookup the value of variable `var` in the current frame.
    fn lookup(&self, var: &LocalVar) -> Option<&SGValue> {
        self.frames.last().unwrap().get(var)
    }

    /// Start interpretation of the initialised interpreter.
    pub unsafe fn interpret(&mut self) {
        // We start interpretation at the branch instruction that was turned into a guard. We need
        // to re-interpret this instruction in order to find out which branch we need to follow.
        loop {
            match LLVMGetInstructionOpcode(self.pc) {
                LLVMOpcode::LLVMBr => self.branch(self.pc),
                LLVMOpcode::LLVMRet => self.ret(self.pc),
                _ => todo!("{:?}", CStr::from_ptr(LLVMPrintValueToString(self.pc))),
            }
        }
    }

    /// Interpret branch instruction `instr`.
    pub unsafe fn branch(&mut self, instr: LLVMValueRef) {
        let cond = LocalVar::new(LLVMGetCondition(instr));
        let val = self.lookup(&cond);
        let res = match val.unwrap() {
            SGValue::U32(v) => *v == 1,
            SGValue::U64(v) => *v == 1,
        };
        let succ = if res {
            LLVMGetSuccessor(instr, 0)
        } else {
            LLVMGetSuccessor(instr, 1)
        };
        self.pc = LLVMGetFirstInstruction(succ);
    }

    /// Interpret return instruction `instr`.
    unsafe fn ret(&mut self, instr: LLVMValueRef) {
        if self.frames.len() == 1 {
            // We've reached the end of the interpreters main, so just get the return value and
            // exit. This is possibly a hack, though I'm not sure what the correct behaviour is.
            let op = LLVMGetOperand(instr, 0);
            let val = if !LLVMIsAConstant(op).is_null() {
                llvmapihelper::llvm_const_to_sgvalue(op)
            } else {
                todo!()
            };
            let ret = match val {
                SGValue::U32(v) => v as i32,
                SGValue::U64(v) => v as i32,
            };
            std::process::exit(ret);
        }
    }
}
