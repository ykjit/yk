use crate::aotsmp::AOT_STACKMAPS;
use capstone::prelude::*;
use dynasmrt::x64::Rq;
use dynasmrt::x64::Rq::{RAX, RBP, RCX, RDI, RDX, RSI, RSP};
use dynasmrt::{dynasm, x64::Assembler, DynasmApi, ExecutableBuffer};
use std::error::Error;
use std::sync::Arc;
use std::sync::LazyLock;
use yksmp::Location::{Constant, Direct, Indirect, LargeConstant, Register};

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
    build_livevars_cp_asm(UNOPT_CP_SMID, OPT_CP_SMID, &mut asm);
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
    build_livevars_cp_asm(OPT_CP_SMID, UNOPT_CP_SMID, &mut asm);

    let call_offset = calc_after_cp_offset(rec.offset).unwrap();
    let target_addr = i64::try_from(rec.offset).unwrap() + call_offset;
    dynasm!(asm
        ; .arch x64
        // ; int3                          // Insert a breakpoint for GDB
        // ; int3                          // Insert a breakpoint for GDB
        ; mov rax, QWORD target_addr
        ; mov [rsp], rax
        ; ret
    );
    let buffer = asm.finalize().unwrap();
    Arc::new(buffer)
});

fn reg_num_to_dynasm_reg(dwarf_reg_num: u16) -> Rq {
    match dwarf_reg_num {
        0 => Rq::RAX,
        1 => Rq::RDX,
        2 => Rq::RCX,
        3 => Rq::RBX,
        4 => Rq::RSI,
        5 => Rq::RDI,
        6 => Rq::RBP,
        7 => Rq::RSP,
        8 => Rq::R8,
        9 => Rq::R9,
        10 => Rq::R10,
        11 => Rq::R11,
        12 => Rq::R12,
        13 => Rq::R13,
        14 => Rq::R14,
        15 => Rq::R15,
        _ => panic!("Unsupported register"),
    }
}

#[cfg(tracer_swt)]
fn reg_num_stack_offset(dwarf_reg_num: u16) -> i32 {
    match dwarf_reg_num {
        0 => 0,    // rax
        1 => 8,    // rdx
        2 => 16,   // rcx
        3 => 24,   // rbx
        4 => 40,   // rsi
        5 => 32,   // rdi
        // rbp is not saved
        // rsp is not saved
        8 => 64,   // r8
        9 => 72,   // r9
        10 => 80,  // r10
        11 => 88,  // r11
        12 => 96,  // r12
        13 => 104, // r13
        14 => 112, // r14
        15 => 120, // r15
        _ => panic!("Unsupported register {}", dwarf_reg_num),
    }
}

#[cfg(tracer_swt)]
fn build_livevars_cp_asm(src_smid: usize, dst_smid: usize, asm: &mut Assembler) {
    let (src_rec, _) = AOT_STACKMAPS.as_ref().unwrap().get(src_smid);
    let (dst_rec, _) = AOT_STACKMAPS.as_ref().unwrap().get(dst_smid);

    // Save all the registers to the stack
    dynasm!(asm
        ; .arch x64
        ; push r15    // 15 - offset 120
        ; push r14    // 14 - offset 112
        ; push r13    // 13 - offset 104
        ; push r12    // 12 - offset 96
        ; push r11    // 11 - offset 88
        ; push r10    // 10 - offset 80
        ; push r9     // 9 - offset 72
        ; push r8     // 8 - offset 64
        ; sub rsp, 16 // Allocates 16 bytes of padding for rsp and rbp
        ; push rsi    // 5 - offset 40
        ; push rdi    // 4 - offset 32
        ; push rbx    // 3 - offset 24
        ; push rcx    // 2 - offset 16
        ; push rdx    // 1 - offset 8
        ; push rax    // 0 - offset 0
    );

    // TODO: remove this temporary break instruction
    dynasm!(asm; int3);

    for (index, src_var) in src_rec.live_vars.iter().enumerate() {
        let dst_var = &dst_rec.live_vars[index];
        if src_var.len() > 1 || dst_var.len() > 1 {
            todo!("Deal with multi register locations");
        }

        let src_location = &src_var.get(0).unwrap();
        let dst_location = &dst_var.get(0).unwrap();
        println!("@@ dst_location: {:?}, src_location: {:?}", dst_location, src_location);

        match dst_location {
            Indirect(dst_reg_num, dst_off, dst_val_size) => {
                match src_location {
                    Register(src_reg_num, src_val_size, src_add_locs, _src_add_loc_reg) => {
                        assert!(
                            dst_val_size == src_val_size,
                            "Indirect to Register - src and dst val size must match. got src: {} and dst: {}",
                            src_val_size, dst_val_size
                        );
                        assert!(*src_add_locs == 0, "deal with additional info");
                        println!(
                            "@@ Indirect to Register - from {:?} to {:?}",
                            src_reg_num, dst_reg_num
                        );
                        let src_reg = u8::try_from(*src_reg_num).unwrap();
                        let src_offset = reg_num_stack_offset(*src_reg_num);
                        let dst_reg = u8::try_from(*dst_reg_num).unwrap();
                        match *src_val_size {
                            1 => dynasm!(asm
                                ; mov al, BYTE [rsp + src_offset]
                                ; mov BYTE [Rq(dst_reg) + *dst_off], al
                            ),
                            2 => dynasm!(asm
                                ; mov ax, WORD [rsp + src_offset]
                                ; mov WORD [Rq(dst_reg) + *dst_off], ax
                            ),
                            4 => dynasm!(asm
                                ; mov eax, DWORD [rsp + src_offset]
                                ; mov DWORD [Rq(dst_reg) + *dst_off], eax
                            ),
                            8 => dynasm!(asm
                                ; mov rax, QWORD [rsp + src_offset]
                                ; mov QWORD [Rq(dst_reg) + *dst_off], rax
                            ),
                            //mov QWORD [rbp - i32::try_from(off_dst).unwrap()], Rq(reg.code())
                            _ => panic!("Unsupported source value size: {}", src_val_size),
                        }
                    },
                    Constant(_val) => {
                        // copy from constant to indirect
                    },
                    Indirect(src_reg_num, src_off, src_val_size) => {
                        println!(
                            "@@ Indirect to Indirect - from {:?} to {:?}",
                            src_reg_num, dst_reg_num
                        );
                        assert!(
                            src_val_size == dst_val_size,
                            "Value sizes must match, got src: {} and dst: {}",
                            src_val_size,
                            dst_val_size
                        );
                        // let src_reg = u8::try_from(*src_reg_num).unwrap();
                        // let dst_reg = u8::try_from(*dst_reg_num).unwrap();

                        match *src_val_size {
                            1 => dynasm!(asm
                                ; mov al, BYTE [rsp + *src_off]
                                ; mov BYTE [rbp + *dst_off], al
                            ),
                            2 => dynasm!(asm
                                ; mov ax, WORD [rsp + *src_off]
                                ; mov WORD [rbp + *dst_off], ax
                            ),
                            4 => dynasm!(asm
                                ; mov eax, DWORD [rsp + *src_off]
                                ; mov DWORD [rbp + *dst_off], eax
                            ),
                            8 => dynasm!(asm
                                ; mov rax, QWORD [rsp + *src_off]
                                ; mov QWORD [rbp + *dst_off], rax
                            ),
                            _ => panic!("Unsupported source value size: {}", src_val_size),
                        }
                    },
                    _ => panic!("Unsupported source location: {:?}", src_location),
                }
            }
            Register(dst_reg_num, dst_val_size, dst_add_locs, _dst_add_loc_reg) => {
                match src_location {
                    Register(src_reg_num, src_val_size, src_add_locs, _src_add_loc_reg) => {
                        println!(
                            "@@ Register to Register - from {:?} to {:?}",
                            src_reg_num, dst_reg_num
                        );
                        assert!(
                            *src_add_locs == 0 && *dst_add_locs == 0,
                            "deal with additional info"
                        );
                        assert!(
                            dst_val_size == src_val_size,
                            "src and dst val size must match"
                        );
                        // skip copying to the same register with the same value size
                        if src_reg_num == dst_reg_num && src_val_size == dst_val_size {
                            continue;
                        }
                        let src_offset = reg_num_stack_offset(*src_reg_num);
                        let dest_reg = u8::try_from(*dst_reg_num).unwrap();
                        match *src_val_size {
                            1 => todo!("implement reg to reg 1 byte"),
                            2 => todo!("implement reg to reg 2 bytes"),
                            4 => todo!("implement reg to reg 4 bytes"),
                            8 => dynasm!(asm; mov Rq(dest_reg), QWORD [rsp + src_offset]),
                            _ => todo!("implement Register to Register value size {}", src_val_size),
                        }
                    },
                    Indirect(src_reg_num, src_off, src_val_size) => {
                        println!(
                            "@@ Register to Indirect - from {:?} to {:?}",
                            src_reg_num, dst_reg_num
                        );

                        assert!(
                            dst_val_size == src_val_size,
                            "Register to Indirect - src and dst val size must match. got src: {} and dst: {}",
                            src_val_size, dst_val_size
                        );
                        println!(
                            "@@ Register to Indirect - from {:?} to {:?}",
                            src_reg_num, dst_reg_num
                        );
                        // let src_reg = u8::try_from(*src_reg_num).unwrap();
                        // let src_offset = reg_num_stack_offset(*src_reg_num);
                        let dst_reg = u8::try_from(*dst_reg_num).unwrap();

                        match *src_val_size {
                            // 1 => dynasm!(asm
                            //     ; mov Rq(dst_reg), BYTE [Rq(src_reg) + *src_off]
                            // ),
                            // 2 => dynasm!(asm
                            //     ; mov Rq(dst_reg), WORD [Rq(src_reg) + *src_off]
                            // ),
                            // 4 => dynasm!(asm
                            //     ; mov Rq(dst_reg), DWORD [Rq(src_reg) + *src_off]
                            // ),
                            8 => dynasm!(asm
                                ; mov Rq(dst_reg), QWORD [rbp + *src_off]
                            ),
                            // 8 => dynasm!(asm; mov Rq(dest_reg), QWORD [rsp + src_offset]),
                            _ => panic!("Unsupported source value size: {}", src_val_size),
                        }
                    },
                    _ => panic!("Unsupported source location: {:?}", src_location),
                }
            }
            Direct(_dst_reg_num, _dst_off, _dst_val_size) => {
                // Direct locations are read-only, so it doesn't make sense to write to
                // them. This is likely a case where the direct value has been moved
                // somewhere else (register/normal stack) so dst and src no longer
                // match. But since the value can't change we can safely ignore this.
            }
            _ => panic!("unexpectd dst location: {:?}", dst_location),
        }

    }

    // Assembly code to restore registers
    // dynasm!(asm
    //     // Restore registers by popping them in reverse order
    //     ; pop rax     // Corresponds to push rax
    //     ; pop rdx     // Corresponds to push rdx
    //     ; pop rcx     // Corresponds to push rcx
    //     ; pop rbx     // Corresponds to push rbx
    //     ; pop rdi     // Corresponds to push rdi
    //     ; pop rsi     // Corresponds to push rsi
    //     ; add rsp, 16 // Reverse sub rsp, 16
    //     ; pop r8      // Corresponds to push r8
    //     ; pop r9      // Corresponds to push r9
    //     ; pop r10     // Corresponds to push r10
    //     ; pop r11     // Corresponds to push r11
    //     ; pop r12     // Corresponds to push r12
    //     ; pop r13     // Corresponds to push r13
    //     ; pop r14     // Corresponds to push r14
    //     ; pop r15     // Corresponds to push r15
    //     ; ret
    // );

    // asm.finalize().unwrap()
}

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
    let mut offset: i64 = 0;
    for inst in instructions.iter() {
        offset += inst.bytes().len() as i64;
        if inst.mnemonic().unwrap_or("") == "call" {
            return Ok(offset);
        }
    }

    Err(format!(
        "Call instruction not found within the code slice: {}, len:{}",
        rec_offset, MAX_CODE_SIZE
    )
    .into())
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
        let call_addr: i32 = 0x666;
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
