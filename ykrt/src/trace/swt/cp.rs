use crate::aotsmp::AOT_STACKMAPS;
use capstone::prelude::*;
use dynasmrt::{dynasm, x64::Assembler, DynasmApi};
use std::error::Error;
use yksmp::Location::{Constant, Direct, Indirect, LargeConstant, Register};

use std::{ffi::c_void};

/// The size of a 64-bit register in bytes.
pub(crate) static REG64_BYTESIZE: u64 = 8;

#[repr(usize)]
#[derive(Debug, Clone, Copy)]
pub enum ControlPointStackMapId {
    // unoptimised (original functions) control point stack map id
    UnOpt = 0,
    // optimised (cloned functions) control point stack map id
    Opt = 1,
}
// Example IR:
// call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 13, ptr @__ykrt_control_point, i32 3, ptr %28, ptr %7, i64 1, ptr %6, ptr %7, ptr %8, ptr %9, ptr %28), !dbg !119
//
//  Where - 1 is the control point id
// pub(crate) static RETURN_INTO_OPT_CP: LazyLock<Arc<ExecutableBuffer>> = LazyLock::new(|| {
//     let mut asm = Assembler::new().unwrap();
//     build_livevars_cp_asm(UnOpt, Opt, &mut asm);
//     let buffer = asm.finalize().unwrap();
//     Arc::new(buffer)
// });

// Example IR:
//  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 0, i32 13, ptr @__ykrt_control_point, i32 3, ptr %28, ptr %7, i64 0, ptr %6, ptr %7, ptr %8, ptr %9, ptr %28), !dbg !74
//
//  Where - 0 is the control point id
// pub(crate) static RETURN_INTO_UNOPT_CP: LazyLock<Arc<ExecutableBuffer>> = LazyLock::new(|| {
// pub(crate) static RETURN_INTO_UNOPT_CP: LazyLock<Arc<ExecutableBuffer>> = LazyLock::new(|| {
//     let mut asm = Assembler::new().unwrap();
//     build_livevars_cp_asm(Opt, UnOpt, &mut asm);
//     let buffer = asm.finalize().unwrap();
//     Arc::new(buffer)
// });

pub unsafe fn you_can_do_it(from: ControlPointStackMapId, to: ControlPointStackMapId, frameaddr: *mut c_void) {
    // println!("@@ you_can_do_it from: {:x} to: {:x} frameaddr: {:x}", from as usize, to as usize, frameaddr as usize);
    let mut asm = Assembler::new().unwrap();

    // let frameaddr_i64: i64 = frameaddr as i64;
    // println!("@@ frameaddr_i64: {:x}, {}, original: {:x}, {}", frameaddr_i64, frameaddr_i64, frameaddr as usize, frameaddr as usize);

    build_livevars_cp_asm(from as usize, to as usize, &mut asm, frameaddr as usize);
    let buffer = asm.finalize().unwrap();
    let func: unsafe fn() = std::mem::transmute(buffer.as_ptr());
    func();
}

#[cfg(tracer_swt)]
fn reg_num_stack_offset(dwarf_reg_num: u16) -> i32 {
    match dwarf_reg_num {
        0 => 0,  // rax
        1 => 8,  // rdx
        2 => 16, // rcx
        3 => 24, // rbx
        4 => 32, // rsi
        5 => 40, // rdi
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
fn build_livevars_cp_asm(src_smid: usize, dst_smid: usize, asm: &mut Assembler, frameaddr: usize) {
    let verbose = false;
    // TODO: find the pushed registers in the control point

    let (src_rec, _) = AOT_STACKMAPS.as_ref().unwrap().get(src_smid);
    let (dst_rec, dst_rec_pinfo) = AOT_STACKMAPS.as_ref().unwrap().get(dst_smid);

    // TODO:
    // 1. memcopy the stack or allocate another stack frame
    // 2. The registers here are not the actual registers we need... We need to take them from the control point

    let mut dest_rsp = dst_rec.size;
    if dst_rec_pinfo.hasfp {
        dest_rsp -= REG64_BYTESIZE;
    }

    // TODO: figure out how to not hardcode this!
    dest_rsp += 112; // Adjusting the stack that is extended by the __ykrt_control_point_real call.
    dynasm!(asm
        ; .arch x64
        ; mov rbp, QWORD frameaddr as i64 // reset rbp
        ; mov rsp, QWORD frameaddr as i64 // reset rsp
        ; sub rsp, (dest_rsp).try_into().unwrap()
        // ; int3
    );

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

    if verbose {
        println!(
            "@@ from {} to {} live var count: {}",
            src_smid,
            dst_smid,
            dst_rec.live_vars.len()
        );
    }

    for (index, src_var) in src_rec.live_vars.iter().enumerate() {
        let dst_var = &dst_rec.live_vars[index];
        if src_var.len() > 1 || dst_var.len() > 1 {
            todo!("Deal with multi register locations");
        }
        assert!(
            src_rec.live_vars.len() == dst_rec.live_vars.len(),
            "Expected single register location"
        );

        let src_location = &src_var.get(0).unwrap();
        let dst_location = &dst_var.get(0).unwrap();
        if verbose {
            println!(
                "@@ dst_location: {:?}, src_location: {:?}",
                dst_location, src_location
            );
        }
        // breakpoint for each location
        // dynasm!(asm; int3);
        match dst_location {
            Indirect(_dst_reg_num, dst_off, dst_val_size) => {
                match src_location {
                    Register(src_reg_num, src_val_size, src_add_locs, _src_add_loc_reg) => {
                        assert!(
                            dst_val_size == src_val_size,
                            "Indirect to Register - src and dst val size must match. got src: {} and dst: {}",
                            src_val_size, dst_val_size
                        );
                        assert!(*src_add_locs == 0, "deal with additional info");
                        let src_offset = reg_num_stack_offset(*src_reg_num);
                        match *src_val_size {
                            1 => dynasm!(asm
                                ; mov al, BYTE [rsp + src_offset]
                                ; mov BYTE [rbp + *dst_off], al
                            ),
                            2 => dynasm!(asm
                                ; mov ax, WORD [rsp + src_offset]
                                ; mov WORD [rbp + *dst_off], ax
                            ),
                            4 => dynasm!(asm
                                ; mov eax, DWORD [rsp + src_offset]
                                ; mov DWORD [rbp + *dst_off], eax
                            ),
                            8 => dynasm!(asm
                                ; mov rax, QWORD [rsp + src_offset]
                                ; mov QWORD [rbp + *dst_off], rax
                            ),
                            _ => panic!(
                                "Unexpected Indirect to Register value size: {}",
                                src_val_size
                            ),
                        }
                    }
                    Constant(_val) => {
                        todo!("implement Indirect to Constant")
                    }
                    LargeConstant(_val) => {
                        todo!("implement Indirect to LargeConstant")
                    }
                    Indirect(_src_reg_num, src_off, src_val_size) => {
                        // TODO: understand what to do where the size value is different
                        let min_size = src_val_size.min(dst_val_size);
                        match min_size {
                            // based on ykrt/src/compile/jitc_yk/codegen/x64/mod.rs
                            1 => dynasm!(asm
                                // TODO: this is problematic cause of read and writes at the sames time
                                // 1. memcopy the whole stack and then copy to the right rbp
                                ; mov al, BYTE [rbp + i32::try_from(*src_off).unwrap()]
                                ; mov BYTE [rbp + i32::try_from(*dst_off).unwrap()], al
                            ),
                            2 => dynasm!(asm
                                ; mov ax, WORD [rbp + i32::try_from(*src_off).unwrap()]
                                ; mov WORD [rbp + i32::try_from(*dst_off).unwrap()], ax
                            ),
                            4 => dynasm!(asm
                                ; mov eax, DWORD [rbp + i32::try_from(*src_off).unwrap()]
                                ; mov DWORD [rbp + i32::try_from(*dst_off).unwrap()], eax
                            ),
                            8 => dynasm!(asm
                                ; mov rax, QWORD [rbp + i32::try_from(*src_off).unwrap()]
                                ; mov QWORD [rbp + i32::try_from(*dst_off).unwrap()], rax
                            ),
                            _ => panic!("Unexpected Indirect to Indirect value size: {}", min_size),
                        }
                    }
                    _ => panic!("Unsupported source location: {:?}", src_location),
                }
            }
            Register(dst_reg_num, dst_val_size, dst_add_locs, _dst_add_loc_reg) => {
                match src_location {
                    Register(src_reg_num, src_val_size, src_add_locs, _src_add_loc_reg) => {
                        assert!(
                            *src_add_locs == 0 && *dst_add_locs == 0,
                            "Register to Register - deal with additional info"
                        );
                        assert!(
                            dst_val_size == src_val_size,
                            "Register to Register - src and dst val size must match. Got src: {} and dst: {}",
                            src_val_size, dst_val_size
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
                            _ => {
                                todo!("unexpect Register to Register value size {}", src_val_size)
                            }
                        }
                    }
                    Indirect(src_reg_num, src_off, src_val_size) => {
                        assert!(*src_reg_num == 6, "Indirect register is expected to be rbp");
                        let dst_reg = u8::try_from(*dst_reg_num).unwrap();
                        // let dst_reg_rd = map_to_rd(*dst_reg_num);
                        match *dst_val_size {
                            1 => todo!("implement reg to indirect 1 byte"),
                            2 => todo!("implement reg to indirect 2 bytes"),
                            4 => todo!("implement reg to indirect 4 bytes"),
                            // 4 => dynasm!(asm; mov Rq(dst_reg), DWORD [rsp + i32::try_from(*src_off).unwrap()]),
                            8 => {
                                dynasm!(asm; mov Rq(dst_reg), QWORD [rbp + i32::try_from(*src_off).unwrap()])
                            }
                            _ => panic!("Unsupported source value size: {}", src_val_size),
                        }
                    }
                    _ => panic!("Unsupported source location: {:?}", src_location),
                }
            }
            Direct(_dst_reg_num, _dst_off, _dst_val_size) => {
                // Direct locations are read-only, so it doesn't make sense to write to
                // them. This is likely a case where the direct value has been moved
                // somewhere else (register/normal stack) so dst and src no longer
                // match. But since the value can't change we can safely ignore this.
            }
            _ => panic!("unexpect dst location: {:?}", dst_location),
        }
    }

    if verbose {
        println!("@@ dst_size: 0x{:x}, dst_rbp: 0x{:x}, dst addr: 0x{:x}", dst_rec.size as i64, frameaddr as i64, dst_rec.offset);
        println!("@@ src_size: 0x{:x}, src_rbp: 0x{:x}, src addr: 0x{:x}", src_rec.size as i64, frameaddr as i64, src_rec.offset);
    }

    let call_offset = calc_after_cp_offset(dst_rec.offset).unwrap();
    let dst_target_addr = i64::try_from(dst_rec.offset).unwrap() + call_offset;

    dynasm!(asm
        ; .arch x64
        // ; int3
        ; sub rsp, 16 // reserves 16 bytes of space on the stack.
        ; mov [rsp], rax // save rsp
        ; mov rax, QWORD dst_target_addr // loads the target address into rax
        ; mov [rsp + 8], rax // stores the target address into rsp+8
        ; pop rax // restores the original rax at rsp
        // ; int3 // breakpoint
        ; ret // loads 8 bytes from rsp and jumps to it
    );
}

// This function calculates the offset to the call instruction just
// after the given offset.
// Example:
//  CP Record offset points to 0x00000000002023a4, we want to find the
//  instruction at 0x00000000002023b1.
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
