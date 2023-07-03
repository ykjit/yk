use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMBasicBlockRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef};
use llvm_sys::target::{LLVMGetModuleDataLayout, LLVMTargetDataRef};
use llvm_sys::LLVMTypeKind;
use std::{ffi::CStr, fmt};

pub struct Module(LLVMModuleRef);

// Replicates struct of same name in `ykllvmwrap.cc`.
#[repr(C)]
pub struct BitcodeSection {
    data: *const u8,
    len: u64,
}

extern "C" {
    pub fn LLVMGetThreadSafeModule(bs: BitcodeSection) -> LLVMModuleRef;
}

impl Module {
    pub unsafe fn from_bc() -> Self {
        let (data, len) = ykutil::obj::llvmbc_section();
        let module = LLVMGetThreadSafeModule(BitcodeSection { data, len });
        Self(module)
    }

    pub fn function(&self, name: *const i8) -> Function {
        let func = unsafe { LLVMGetNamedFunction(self.0, name) };
        debug_assert!(!func.is_null());
        unsafe { Function::new(func) }
    }

    pub fn datalayout(&self) -> LLVMTargetDataRef {
        unsafe { LLVMGetModuleDataLayout(self.0) }
    }
}

pub struct Function(LLVMValueRef);

impl Function {
    pub unsafe fn new(func: LLVMValueRef) -> Self {
        debug_assert!(!LLVMIsAFunction(func).is_null());
        Self(func)
    }

    pub fn bb(&self, bbidx: usize) -> BasicBlock {
        let mut bb = unsafe { LLVMGetFirstBasicBlock(self.0) };
        for _ in 0..bbidx {
            bb = unsafe { LLVMGetNextBasicBlock(bb) };
        }
        unsafe { BasicBlock::new(bb) }
    }
}

pub struct BasicBlock(LLVMBasicBlockRef);

impl BasicBlock {
    pub unsafe fn new(bb: LLVMBasicBlockRef) -> Self {
        Self(bb)
    }

    pub fn instruction(&self, instridx: usize) -> Value {
        let mut instr = unsafe { LLVMGetFirstInstruction(self.0) };
        for _ in 0..instridx {
            instr = unsafe { LLVMGetNextInstruction(instr) };
        }
        unsafe { Value::new(instr) }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub struct Type(LLVMTypeRef);
impl Type {
    pub fn kind(&self) -> LLVMTypeKind {
        unsafe { LLVMGetTypeKind(self.0) }
    }

    pub fn is_integer(&self) -> bool {
        matches!(self.kind(), LLVMTypeKind::LLVMIntegerTypeKind)
    }

    pub fn get_int_width(&self) -> u32 {
        debug_assert!(self.is_integer());
        unsafe { LLVMGetIntTypeWidth(self.0) }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct Value(LLVMValueRef);
impl Value {
    pub unsafe fn new(vref: LLVMValueRef) -> Self {
        Value(vref)
    }

    pub fn get(&self) -> LLVMValueRef {
        self.0
    }

    pub fn is_instruction(&self) -> bool {
        unsafe { !LLVMIsAInstruction(self.0).is_null() }
    }

    pub fn is_alloca(&self) -> bool {
        unsafe { !LLVMIsAAllocaInst(self.0).is_null() }
    }

    pub fn is_store(&self) -> bool {
        unsafe { !LLVMIsAStoreInst(self.0).is_null() }
    }

    pub fn is_call(&self) -> bool {
        unsafe { !LLVMIsACallInst(self.0).is_null() }
    }

    pub fn is_intrinsic(&self) -> bool {
        unsafe { !LLVMIsAIntrinsicInst(self.0).is_null() }
    }

    pub fn get_type(&self) -> Type {
        unsafe { Type(LLVMTypeOf(self.0)) }
    }

    pub fn get_operand(&self, idx: u32) -> Value {
        unsafe {
            debug_assert!(!LLVMIsAUser(self.0).is_null());
            Value(LLVMGetOperand(self.0, idx))
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", unsafe {
            CStr::from_ptr(LLVMPrintValueToString(self.0))
        })
    }
}
