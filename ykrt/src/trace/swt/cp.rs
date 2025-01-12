use crate::aotsmp::AOT_STACKMAPS;
use capstone::prelude::*;
use dynasmrt::{dynasm, x64::Assembler, DynasmApi};
use std::error::Error;
use std::ffi::c_void;
use yksmp::Location::{Constant, Direct, Indirect, LargeConstant, Register};

/// The size of a 64-bit register in bytes.
pub(crate) static REG64_BYTESIZE: i32 = 8;

// Feature flags
pub static CP_TRANSITION_DEBUG_MODE: bool = true;
pub static STACK_SANDWITCH: bool = false;

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlPointStackMapId {
    // unoptimised (original functions) control point stack map id
    Opt = 0,
    // optimised (cloned functions) control point stack map id
    UnOpt = 1,
}

pub struct ControlPointTransition {
    pub src_smid: ControlPointStackMapId,
    pub dst_smid: ControlPointStackMapId,
    pub frameaddr: *const c_void,
    pub rsp: *const c_void,
    pub trace_addr: *const c_void,
    pub exec_trace: bool,
    pub exec_trace_fn: ExecTraceFn,
}

// We use the registers saved by the control point.
// __ykrt_control_point:
// "push rax",
// "push rcx",
// "push rbx",
// "push rdi",
// "push rsi",
// ....
// "push r8",
// "push r9",
// "push r10",
// "push r11",
// "push r12",
// "push r13",
// "push r14",
// "push r15",
#[cfg(tracer_swt)]
fn reg_num_to__ykrt_control_point_stack_offset(dwarf_reg_num: u16) -> i32 {
    let offset = match dwarf_reg_num {
        0 => 0x60, // rax
        // 1 => 8,  // rdx - is not saved
        2 => 0x58, // rcx
        3 => 0x50, // rbx
        // Question: why rsi and rdi are not at their index?
        5 => 0x48, // rdi
        4 => 0x40, // rsi
        // 6 => 0x48 - not saved
        // 7 => 0x40 - not saved
        8 => 0x38,  // r8
        9 => 0x30,  // r9
        10 => 0x28, // r10
        11 => 0x20, // r11
        12 => 0x18, // r12
        13 => 0x10, // r13
        14 => 0x8,  // r14
        15 => 0x0,  // r15
        _ => panic!("Unsupported register {}", dwarf_reg_num),
    };
    return offset;
}

pub(crate) type ExecTraceFn = unsafe extern "C" fn(
    frameaddr: *const c_void,
    rsp: *const c_void,
    trace_addr: *const c_void,
) -> !;

pub unsafe fn control_point_transition(transition: ControlPointTransition) {
    let ControlPointTransition {
        src_smid,
        dst_smid,
        frameaddr,
        rsp,
        trace_addr,
        exec_trace,
        exec_trace_fn,
    } = transition;
    let frameaddr = frameaddr as usize;
    let mut asm = Assembler::new().unwrap();
    let (src_rec, _) = AOT_STACKMAPS.as_ref().unwrap().get(src_smid as usize);
    let (dst_rec, dst_rec_pinfo) = AOT_STACKMAPS.as_ref().unwrap().get(dst_smid as usize);

    let (unopt_rec, unopt_pinfo) = AOT_STACKMAPS
        .as_ref()
        .unwrap()
        .get(ControlPointStackMapId::UnOpt as usize);
    let (opt_rec, opt_pinfo) = AOT_STACKMAPS
        .as_ref()
        .unwrap()
        .get(ControlPointStackMapId::Opt as usize);

    let mut unopt_frame_size = unopt_rec.size;
    if unopt_pinfo.hasfp {
        unopt_frame_size -= REG64_BYTESIZE as u64;
    }
    let mut opt_frame_size = opt_rec.size;
    if opt_pinfo.hasfp {
        opt_frame_size -= REG64_BYTESIZE as u64;
    }

    if CP_TRANSITION_DEBUG_MODE {
        // debug breakpoint
        dynasm!(asm ; .arch x64 ; int3 );
        println!(
            "@@ unopt_frame_size: 0x{:x}, opt_frame_size: 0x{:x}",
            unopt_frame_size, opt_frame_size
        );
    }

        // We use src_rsp_offset to find the registers saved by the control
    // point.
    // If the copied valies are wrong - check that the REG64_BYTESIZE * X offsets correct
    // wrt the __ykrt_control_point rsp.
    let mut src_rsp_offset: i32 = 0;
    if src_smid == ControlPointStackMapId::Opt {
        src_rsp_offset = i32::try_from(unopt_frame_size + REG64_BYTESIZE as u64 * 4).unwrap();
    } else if src_smid == ControlPointStackMapId::UnOpt {
        // TODO: verify that this is correct!
        src_rsp_offset = i32::try_from(opt_frame_size + REG64_BYTESIZE as u64 * 6).unwrap();
    }


    let mut src_rbp: i64 = 0;
    let mut src_rsp: i64 = 0;
    let mut dst_rbp: i64 = 0;

    // Stack Diagram:
    // +--------------------------------------------+
    // | optimised frame                            |
    // +--------------------------------------------+
    // | saved optimised rbp                        |
    // +--------------------------------------------+
    // | unoptimised rbp                            |
    // +--------------------------------------------+
    // | unoptimised Frame (rbp - unopt frame size) |
    // +--------------------------------------------+
    match (src_smid, dst_smid) {
        (ControlPointStackMapId::Opt, ControlPointStackMapId::UnOpt) => {
            src_rbp = frameaddr as i64;
            dst_rbp = src_rbp - REG64_BYTESIZE as i64 - opt_frame_size as i64;
            src_rsp = frameaddr as i64 - opt_frame_size as i64 - REG64_BYTESIZE as i64 - REG64_BYTESIZE as i64 - src_rsp_offset as i64;
            dynasm!(asm
                ; .arch x64
                ; mov rbp, QWORD frameaddr as i64 // set rbp to optimised rbp
                ; mov rsp, QWORD frameaddr as i64 // set rsp to optimised rbp
                ; sub rsp, (opt_frame_size).try_into().unwrap() // adjust rsp to optimised frame size
                ; push rbp // save optimised rbp
                ; mov rbp, rsp // set unoptimised rbp
                ; sub rsp, (unopt_frame_size).try_into().unwrap() // set rsp to unoptimised frame size
            );
        }
        (ControlPointStackMapId::UnOpt, ControlPointStackMapId::Opt) => {
            dst_rbp = frameaddr as i64;
            src_rbp = frameaddr as i64 - REG64_BYTESIZE as i64 - unopt_frame_size as i64;
            src_rsp = frameaddr as i64 - REG64_BYTESIZE as i64 - unopt_frame_size as i64;
            dynasm!(asm
                ; .arch x64
                ; add rsp, (unopt_frame_size).try_into().unwrap() // set rsp to optimised rsp
                ; pop rbp // restore optimised rbp
            );
        }
        _ => panic!(
            "Unsupported stack map transition: {:?} -> {:?}",
            src_smid, dst_smid
        ),
    }
    if CP_TRANSITION_DEBUG_MODE {
        println!("@@ frameaddr: 0x{:x}, src_rbp: 0x{:x}, dst_rbp: 0x{:x}", frameaddr as i64, src_rbp, dst_rbp);
    }

    // Original
    // let mut dest_rsp_frame_size = opt_frame_size;
    // if dst_smid == ControlPointStackMapId::UnOpt {
    //     dest_rsp_frame_size = unopt_frame_size;
    // }
    // if CP_TRANSITION_DEBUG_MODE {
    //     println!("@@ rbp: 0x{:x}, rsp: 0x{:x}", frameaddr as i64, frameaddr as i64 - dest_rsp_frame_size as i64);
    // }
    // dynasm!(asm
    //     ; .arch x64
    //     ; mov rbp, QWORD frameaddr as i64 // reset rbp
    //     ; mov rsp, QWORD frameaddr as i64 // reset rsp
    //     ; sub rsp, (dest_rsp_frame_size).try_into().unwrap() // adjust rsp
    // );


    if CP_TRANSITION_DEBUG_MODE {
        println!("--------------------------------");
        println!("@@ src live vars - smid: {:?}", src_smid);
        for (index, src_var) in src_rec.live_vars.iter().enumerate() {
            let src_location = &src_var.get(0).unwrap();
            println!("{:?}", src_location);
        }
        println!("--------------------------------");
        println!("@@ dst live vars - smid: {:?}", dst_smid);
        for (index, dst_var) in dst_rec.live_vars.iter().enumerate() {
            let dst_location = &dst_var.get(0).unwrap();
            println!("{:?}", dst_location);
        }
        println!("--------------------------------");
    }

    for (index, src_var) in src_rec.live_vars.iter().enumerate() {
        let dst_var = &dst_rec.live_vars[index];
        if src_var.len() > 1 || dst_var.len() > 1 {
            todo!("Deal with multi register locations");
        }
        assert!(
            src_rec.live_vars.len() == dst_rec.live_vars.len(),
            "Expected single register location, got src: {} and dst: {}",
            src_rec.live_vars.len(),
            dst_rec.live_vars.len()
        );

        let src_location = &src_var.get(0).unwrap();
        let dst_location = &dst_var.get(0).unwrap();

        match src_location {
            Register(src_reg_num, src_val_size, src_add_locs) => {
                match dst_location {
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        assert!(
                            src_add_locs.len() == 0 && dst_add_locs.len() == 0,
                            "Register to Register - deal with additional info"
                        );
                        assert!(
                            dst_val_size == src_val_size,
                            "Register to Register - src and dst val size must match. Got src: {} and dst: {}",
                            src_val_size, dst_val_size
                        );
                        if CP_TRANSITION_DEBUG_MODE {
                            println!(
                                "@@ Reg2Reg src: {:?}, dst: {:?}",
                                src_location, dst_location
                            );
                        }
                        // skip copying to the same register with the same value size
                        if src_reg_num == dst_reg_num && src_val_size == dst_val_size {
                            continue;
                        }
                        todo!("implement Register to Register");
                        // let src_offset = reg_num_to__ykrt_control_point_stack_offset(*src_reg_num)
                        // let dest_reg = u8::try_from(*dst_reg_num).unwrap();
                        // match *src_val_size {
                        //     1 => dynasm!(asm; mov Rb(dest_reg), BYTE [rsp + src_offset]),
                        //     2 => dynasm!(asm; mov Rw(dest_reg), WORD [rsp + src_offset]),
                        //     4 => dynasm!(asm; mov Rd(dest_reg), DWORD [rsp + src_offset]),
                        //     8 => dynasm!(asm; mov Rq(dest_reg), QWORD [rsp + src_offset]),
                        //     _ => {
                        //         todo!("unexpect Register to Register value size {}", src_val_size)
                        //     }
                        // }
                    }
                    Indirect(_dst_reg_num, dst_off, dst_val_size) => {
                        assert!(
                            dst_val_size == src_val_size,
                            "Indirect to Register - src and dst val size must match. got src: {} and dst: {}",
                            src_val_size, dst_val_size
                        );
                        assert!(src_add_locs.len() == 0, "deal with additional info");
                        if CP_TRANSITION_DEBUG_MODE {
                            println!(
                                "@@ Reg2Ind src: {:?}, dst: {:?}",
                                src_location, dst_location
                            );
                        }
                        let src_offset = reg_num_to__ykrt_control_point_stack_offset(*src_reg_num)
                            - src_rsp_offset;
                        // let src_addr = i32::try_from(src_rbp as i32 + src_offset).unwrap();
                        // let dst_addr = i32::try_from(dst_rbp as i32 + *dst_off).unwrap();
                        // match *src_val_size {
                        //     1 => dynasm!(asm
                        //         ; mov al, BYTE [rsp + src_offset]
                        //         ; mov BYTE [dst_addr], al
                        //     ),
                        //     2 => dynasm!(asm
                        //         ; mov ax, WORD [rsp + src_offset]
                        //         ; mov WORD [dst_addr], ax
                        //     ),
                        //     4 => dynasm!(asm
                        //         ; mov eax, DWORD [rsp + src_offset]
                        //         ; mov DWORD [dst_addr], eax
                        //     ),
                        //     8 => dynasm!(asm
                        //         ; mov rax, QWORD [rsp + src_offset]
                        //         ; mov QWORD [dst_addr], rax
                        //     ),
                        //     _ => panic!(
                        //         "Unexpected Indirect to Register value size: {}",
                        //         src_val_size
                        //     ),
                        // }
                        todo!("implement Indirect to Register");
                    }
                    Constant(_val) => {
                        todo!("implement Indirect to Constant")
                    }
                    LargeConstant(_val) => {
                        todo!("implement Indirect to LargeConstant")
                    }
                    Direct(_dst_reg_num, dst_off, _dst_val_size) => {
                        // We can't write to a direct location cause its
                        // a constant reference to a value on the stack
                        // so we can't copy it to another location
                        let src_offset = reg_num_to__ykrt_control_point_stack_offset(*src_reg_num);
                        let rsp_addr = src_rsp as i64 + src_offset as i64;
                        if CP_TRANSITION_DEBUG_MODE {
                            println!("Reg2Dir src: {:?}, dst: {:?}, rbp_offset: 0x{:x}, src_rsp: 0x{:x}, rsp_addr: 0x{:x}, rsp_addr_i32: 0x{:x}", src_location, dst_location, src_rsp_offset, src_rsp, rsp_addr, rsp_addr as i32);
                        }
                        let dst_rbp_addr = i32::try_from(dst_rbp as i32 + *dst_off).unwrap();
                        match *src_val_size {
                            1 => todo!(),
                            2 => todo!(),
                            4 => todo!(),
                            8 => dynasm!(asm
                                ; mov rbx, rsp_addr
                                ; mov rax, QWORD [rbx] // Load the pointer (e.g. 0x00007ffff6e4b020)
                                ; mov rax, QWORD [rax] // Dereference the pointer to load the value (0x5)
                                ; mov [dst_rbp_addr], rax // Store the actual value (0x5) to the destination
                            ),
                            _ => panic!(
                                "Unexpected Indirect to Register value size: {}",
                                src_val_size
                            ),
                        }
                    }
                    _ => panic!("Unsupported dst location: {:?}", dst_location),
                }
            }
            Indirect(src_reg_num, src_off, src_val_size) => {
                match dst_location {
                    Indirect(_dst_reg_num, dst_off, dst_val_size) => {
                        if CP_TRANSITION_DEBUG_MODE {
                            println!(
                                "@@ Ind2Ind src: {:?}, dst: {:?}",
                                src_location, dst_location
                            );
                        }
                        // TODO: understand what to do where the size value is different
                        let min_size = src_val_size.min(dst_val_size);
                        let src_addr = i32::try_from(src_rbp as i32 + src_off).unwrap();
                        let dst_addr = i32::try_from(dst_rbp as i32 + *dst_off).unwrap();
                        match min_size {
                            // based on ykrt/src/compile/jitc_yk/codegen/x64/mod.rs
                            1 => dynasm!(asm
                                // TODO: this is problematic cause of read and writes at the sames time
                                // 1. memcopy the whole stack and then copy to the right rbp
                                ; mov al, BYTE [src_addr]
                                ; mov BYTE [dst_addr], al
                            ),
                            2 => dynasm!(asm
                                ; mov ax, WORD [src_addr]
                                ; mov WORD [dst_addr], ax
                            ),
                            4 => dynasm!(asm
                                ; mov eax, DWORD [src_addr]
                                ; mov DWORD [dst_addr], eax
                            ),
                            8 => dynasm!(asm
                                ; mov rax, QWORD [src_addr]
                                ; mov QWORD [dst_addr], rax
                            ),
                            _ => panic!("Unexpected Indirect to Indirect value size: {}", min_size),
                        }
                    }
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        if CP_TRANSITION_DEBUG_MODE {
                            println!(
                                "@@ Ind2Reg src: {:?}, dst: {:?}",
                                src_location, dst_location
                            );
                        }
                        assert!(*src_reg_num == 6, "Indirect register is expected to be rbp");
                        let src_offset = i32::try_from(*src_off).unwrap();
                        let src_rbp_addr = i32::try_from(src_rbp as i32 + src_offset).unwrap();
                        let dst_reg = u8::try_from(*dst_reg_num).unwrap();
                        match *dst_val_size {
                            1 => dynasm!(asm; mov Rb(dst_reg), BYTE [src_rbp_addr]),
                            2 => dynasm!(asm; mov Rw(dst_reg), WORD [src_rbp_addr]),
                            4 => dynasm!(asm; mov Rd(dst_reg), DWORD [src_rbp_addr]),
                            8 => dynasm!(asm; mov Rq(dst_reg), QWORD [src_rbp_addr]),
                            _ => panic!("Unsupported source value size: {}", src_val_size),
                        }
                    }
                    _ => panic!("Unsupported dst location: {:?}", dst_location),
                }
            }
            Direct(src_reg_num, src_off, src_val_size) => {
                match dst_location {
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        if CP_TRANSITION_DEBUG_MODE {
                            println!(
                                "@@ Dir2Reg src: {:?}, dst: {:?}",
                                src_location, dst_location
                            );
                        }
                        let src_offset = i32::try_from(*src_off).unwrap();
                        let src_addr = i32::try_from(src_rbp as i32 + src_offset).unwrap();
                        let dst_reg = u8::try_from(*dst_reg_num).unwrap();
                        match *dst_val_size {
                            // 1 => dynasm!(asm; mov Rb(dst_reg), BYTE [src_addr]),
                            // 2 => dynasm!(asm; mov Rw(dst_reg), WORD [src_addr]),
                            // 4 => dynasm!(asm; mov Rd(dst_reg), DWORD [src_addr]),
                            8 => dynasm!(asm; lea Rq(dst_reg), [src_addr]),
                            _ => panic!("Unsupported source value size: {}", src_val_size),
                        }
                    }
                    Direct(_dst_reg_num, dst_off, _dst_val_size) => {
                        if CP_TRANSITION_DEBUG_MODE {
                            println!(
                                "[SKIPPED] Dir2Dir src: {:?}, dst: {:?}",
                                src_location, dst_location
                            );
                        }
                        // Direct locations are read-only, so it doesn't make sense to write to
                        // them. This is likely a case where the direct value has been moved
                        // somewhere else (register/normal stack) so dst and src no longer
                        // match. But since the value can't change we can safely ignore this.

                        // println!("Dir2Dir src: {:?}, dst: {:?}", src_location, dst_location);
                        // match *src_val_size {
                        //     1 => todo!(),
                        //     2 => todo!(),
                        //     4 => todo!(),
                        //     8 => dynasm!(asm
                        //         ; mov rax, QWORD [rbp+*src_off]   // Load the pointer (e.g. 0x00007ffff6e4b020)
                        //         ; mov rax, QWORD [rax]            // Dereference the pointer to load the value (0x5)
                        //         ; mov [rbp + *dst_off], rax       // Store the actual value (0x5) to the destination
                        //     ),
                        //     _ => panic!(
                        //         "Unexpected Indirect to Register value size: {}",
                        //         src_val_size
                        //     ),
                        // }
                    }
                    _ => panic!("Unsupported dst location: {:?}", dst_location),
                }
            }
            _ => panic!("Unsupported source location: {:?}", src_location),
        }
    }

    if CP_TRANSITION_DEBUG_MODE {
        println!(
            "@@ dst_size: 0x{:x}, dst_rbp: 0x{:x}, dst_addr: 0x{:x}",
            dst_rec.size as i64, frameaddr as i64, dst_rec.offset
        );
    }
    if exec_trace {
        if CP_TRANSITION_DEBUG_MODE {
            println!("@@ calling exec_trace");
        }
        // Move the arguments into the appropriate registers
        dynasm!(asm
            ; .arch x64
            ; mov rdi, QWORD frameaddr as i64                   // First argument
            ; mov rsi, QWORD rsp as i64    // Second argument
            ; mov rdx, QWORD trace_addr as i64          // Third argument
            ; mov rcx, QWORD exec_trace_fn as i64         // Move function pointer to rcx
            ; call rcx // Call the function - we don't care about rcx because its overriden in the exec_trace_fn
        );
    } else {
        let call_offset = calc_after_cp_offset(dst_rec.offset).unwrap();
        let dst_target_addr = i64::try_from(dst_rec.offset).unwrap() + call_offset;
        if CP_TRANSITION_DEBUG_MODE {
            println!("@@ transitioning to 0x{:x}", dst_target_addr);
        }
        dynasm!(asm
            ; .arch x64
            // ; int3
            // ; add rsp, TOTAL_STACK_ADJUSTMENT // reserves 128 bytes of space on the stack.
            ; sub rsp, 16 // reserves 16 bytes of space on the stack.
            ; mov [rsp], rax // save rsp
            ; mov rax, QWORD dst_target_addr // loads the target address into rax
            ; mov [rsp + REG64_BYTESIZE as i32], rax // stores the target address into rsp+8
            ; pop rax // restores the original rax at rsp
            // ; int3 // breakpoint
            ; ret // loads 8 bytes from rsp and jumps to it
        );
    }
    let buffer = asm.finalize().unwrap();
    let func: unsafe fn() = std::mem::transmute(buffer.as_ptr());
    func();
}

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
