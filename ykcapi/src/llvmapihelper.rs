use crate::sginterp::SGValue;
use llvm_sys::core::*;
use llvm_sys::prelude::{LLVMBasicBlockRef, LLVMValueRef};
use llvm_sys::{LLVMTypeKind};

pub unsafe fn get_basic_block(func: LLVMValueRef, bbidx: u32) -> LLVMBasicBlockRef {
    let mut bb = LLVMGetFirstBasicBlock(func);
    for _ in 0..bbidx {
        bb = LLVMGetNextBasicBlock(bb);
    }
    bb
}

pub unsafe fn get_instruction(bb: LLVMBasicBlockRef, instridx: u32) -> LLVMValueRef {
    let mut instr = LLVMGetFirstInstruction(bb);
    for _ in 0..instridx {
        instr = LLVMGetNextInstruction(instr);
    }
    instr
}

pub unsafe fn parse_const(c: LLVMValueRef) -> SGValue {
    let ty = LLVMTypeOf(c);
    let kind = LLVMGetTypeKind(ty);
    match kind {
        LLVMTypeKind::LLVMIntegerTypeKind => {
            let width = LLVMGetIntTypeWidth(ty);
            let val = LLVMConstIntGetZExtValue(c) as u64;
            match width {
                32 => SGValue::U32(val as u32),
                64 => SGValue::U64(val),
                _ => todo!(),
            }
        }
        _ => todo!(),
    }
}
