use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMBasicBlockRef, LLVMValueRef};

#[derive(PartialEq, Eq, Hash)]
pub struct LocalVar(LLVMValueRef);

impl LocalVar {
    pub unsafe fn new(instr: LLVMValueRef) -> Self {
      debug_assert!(!LLVMIsAInstruction(instr).is_null());
      Self(instr)
    }
}
