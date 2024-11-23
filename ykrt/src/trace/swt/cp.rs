use crate::aotsmp::AOT_STACKMAPS;
use capstone::prelude::*;
use dynasmrt::{dynasm, x64::Assembler, DynasmApi, ExecutableBuffer};
use std::error::Error;
use std::sync::LazyLock;
use std::{sync::Arc};

// unoptimised (original functions) control point stack map id
const UNOPT_CP_SMID: usize = 0;
// optimised (cloned functions) control point stack map id
const OPT_CP_SMID: usize = 1;

// Example IR:
// call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 13, ptr @__ykrt_control_point, i32 3, ptr %28, ptr %7, i64 1, ptr %6, ptr %7, ptr %8, ptr %9, ptr %28), !dbg !119
//
//  Where - 1 is the control point id
pub(crate) static RETURN_INTO_OPT_CP: LazyLock<Arc<ExecutableBuffer>> = LazyLock::new(|| {
    let (rec, _) = AOT_STACKMAPS.as_ref().unwrap().get(OPT_CP_SMID);
    let mut asm = Assembler::new().unwrap();

    // let target_addr = i64::try_from(rec.offset).unwrap() + CALL_OFFSET;
    let call_offset = calc_after_cp_offset(rec.offset).unwrap();
    let target_addr = i64::try_from(rec.offset).unwrap() + call_offset;
    dynasm!(asm
        ; .arch x64
        // ; int3                          // Insert a breakpoint for GDB
        ; mov rax, QWORD target_addr
        ; mov [rsp], rax
        ; ret
    );
    let buffer = asm.finalize().unwrap();
    Arc::new(buffer)
});

// Example IR:
//  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 13, ptr @__ykrt_control_point, i32 3, ptr %28, ptr %7, i64 0, ptr %6, ptr %7, ptr %8, ptr %9, ptr %28), !dbg !74
//
//  Where - 0 is the control point id
pub(crate) static RETURN_INTO_UNOPT_CP: LazyLock<Arc<ExecutableBuffer>> = LazyLock::new(|| {
    let (rec, _) = AOT_STACKMAPS.as_ref().unwrap().get(UNOPT_CP_SMID);
    let mut asm = Assembler::new().unwrap();
    let call_offset = calc_after_cp_offset(rec.offset).unwrap();
    let target_addr = i64::try_from(rec.offset).unwrap() + call_offset;
    dynasm!(asm
        ; .arch x64
        // ; int3                          // Insert a breakpoint for GDB
        ; mov rax, QWORD target_addr
        ; mov [rsp], rax
        ; ret
    );
    let buffer = asm.finalize().unwrap();
    Arc::new(buffer)
});

// Additional offset to the CP Record offset.
// TODO: calculate this offset somehow instead of hardcoding it.
// Example:
//  CP Record offset points to 0x00000000002023a4 but we want 0x00000000002023b1
//  Which is in offset of 0x00000000002023b1 - 0x00000000002023a4 = 0xD = 13
//  0x00000000002023a4 <+308>:	movabs $0x202620,%r11
//  0x00000000002023ae <+318>:	call   *%r11
//  0x00000000002023b1 <+321>:	jmp    0x2023b3 <main+323>
fn calc_after_cp_offset(rec_offset: u64) -> Result<i64, Box<dyn Error>> {
    // Define the maximum number of bytes to disassemble
    const MAX_CODE_SIZE: usize = 64;
    // Read the machine code starting at rec_offset
    let code_slice = unsafe { std::slice::from_raw_parts(rec_offset as *const u8, MAX_CODE_SIZE) };
    let cs = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .build()
        .unwrap();
    // Disassemble the code
    let instructions = cs.disasm_all(code_slice, rec_offset as u64).unwrap();
    // Initialize the offset accumulator
    let mut offset:i64 = 0;
    for inst in instructions.iter() {
        offset += inst.bytes().len() as i64;
        if inst.mnemonic().unwrap_or("") == "call" {
            return Ok(offset);
        }
    }

    Err(format!("Call instruction not found within the code slice: {}, len:{}", rec_offset, MAX_CODE_SIZE).into())
}



#[cfg(test)]
mod tests {
    use super::*;
    use dynasmrt::{dynasm, x64::Assembler};
    use std::error::Error;

    #[test]
    fn test_calc_after_cp_offset_with_call_instruction() -> Result<(), Box<dyn Error>> {
        // Arrange: Create a buffer with a call instruction
        let mut asm = Assembler::new().unwrap();
        let call_addr:i32 = 0x666;
        dynasm!(asm
            ; .arch x64
            ; nop
            ; call call_addr
            ; ret
        );
        let buffer = asm.finalize().unwrap();
        let code_ptr = buffer.ptr(dynasmrt::AssemblyOffset(0)) as u64;
        // Act: Calculate the call offset
        let offset = calc_after_cp_offset(code_ptr)?;
        // Assert: The offset should be 6 bytes
        assert_eq!(offset, 6, "The call offset should be 6 bytes");
        Ok(())
    }
    #[test]
    fn test_calc_after_cp_offset_with_movabs_and_nops() -> Result<(), Box<dyn Error>> {
        // Arrange: Create a buffer with movabs, multiple nops, and call instruction
        let mut asm = Assembler::new().unwrap();
        dynasm!(asm
            ; .arch x64
            ; nop                         // 1 byte
            ; mov r11, 0x202620           // 10 bytes
            ; call r11                    // 2 bytes
            ; ret                         // 1 byte
        );
        let buffer = asm.finalize().unwrap();
        let code_ptr = buffer.ptr(dynasmrt::AssemblyOffset(0)) as u64;
        // Act: Calculate the call offset
        let offset = calc_after_cp_offset(code_ptr)?;
        // Assert: The offset should be 13 bytes
        assert_eq!(offset, 13, "The call offset should be 13 bytes");
        Ok(())
    }
}