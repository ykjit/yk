use std::collections::HashMap;
use std::env;
use std::sync::LazyLock;

use crate::aotsmp::AOT_STACKMAPS;
use crate::trace::swt::live_vars::{copy_live_vars_to_temp_buffer, set_destination_live_vars};
use capstone::prelude::*;
use dynasmrt::{DynasmApi, ExecutableBuffer, dynasm, x64::Assembler};

use std::error::Error;
use std::ffi::c_void;

use crate::log::stats::Stats;

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlPointStackMapId {
    // unoptimised (original functions) control point stack map id
    Opt = 0,
    // optimised (cloned functions) control point stack map id
    UnOpt = 1,
}

/// The size of a 64-bit register in bytes.
pub(crate) static REG64_BYTESIZE: u64 = 8;

// Flag for verbose logging
pub static YKD_SWT_VERBOSE: LazyLock<bool> = LazyLock::new(|| {
    env::var("YKD_SWT_VERBOSE")
        .map(|v| v == "1")
        .unwrap_or(false)
});

// Flag for verbose logging of asm
pub static YKD_SWT_VERBOSE_ASM: LazyLock<bool> = LazyLock::new(|| {
    env::var("YKD_SWT_VERBOSE_ASM")
        .map(|v| v == "1")
        .unwrap_or(false)
});

// Flag for control point asm break point instruction (int3)
pub(crate) static YKD_SWT_CP_BREAK: LazyLock<bool> = LazyLock::new(|| {
    env::var("YKD_SWT_CP_BREAK")
        .map(|v| v == "1")
        .unwrap_or(false)
});

// Maps DWARF register numbers to `dynasm` register numbers.
// This function takes a DWARF register number as input and returns the
// corresponding dynasm register number1. The mapping is based on the
// x86_64 architecture, and it's important to note that some registers
// (rsi, rdi, rbp, and rsp) have a slightly different order in dynasm
// compared to their sequential DWARF numbering.
// https://docs.rs/dynasmrt/latest/dynasmrt/x64/enum.Rq.html
pub(crate) fn dwarf_to_dynasm_reg(dwarf_reg_num: u8) -> u8 {
    match dwarf_reg_num {
        0 => 0,   // RAX
        1 => 2,   // RDX
        2 => 1,   // RCX
        3 => 3,   // RBX
        4 => 6,   // RSI
        5 => 7,   // RDI
        6 => 5,   // RBP
        7 => 4,   // RSP
        8 => 8,   // R8
        9 => 9,   // R9
        10 => 10, // R10
        11 => 11, // R11
        12 => 12, // R12
        13 => 13, // R13
        14 => 14, // R14
        15 => 15, // R15
        _ => panic!("Unsupported DWARF register number: {}", dwarf_reg_num),
    }
}

// Mapping of DWARF register numbers to offsets in the __ykrt_control_point stack frame.
pub(crate) static REG_OFFSETS: LazyLock<HashMap<u16, i32>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(0, 0x60); // rax
    // 1 => 8,  // rdx - is not saved
    m.insert(2, 0x58); // rcx
    m.insert(3, 0x50); // rbx
    m.insert(5, 0x48); // rdi
    m.insert(4, 0x40); // rsi
    // 6 => 0x48 - not saved
    // 7 => 0x40 - not saved
    m.insert(8, 0x38); // r8
    m.insert(9, 0x30); // r9
    m.insert(10, 0x28); // r10
    m.insert(11, 0x20); // r11
    m.insert(12, 0x18); // r12
    m.insert(13, 0x10); // r13
    m.insert(14, 0x8); // r14
    m.insert(15, 0x0); // r15
    m
});

pub struct CPTransition {
    // The stack map id that we transition from.
    pub smid: ControlPointStackMapId,
    // The frame address of the caller.
    pub frameaddr: *const c_void,
    // The address of the trace to execute.
    // If the value is 0, it means that there is no trace to execute.
    pub trace_addr: *const c_void,
}

/// Transitions from optimised to unoptimised execution mode at a control point.
///
/// This function generates and executes assembly code to transition from the optimised
/// control point to the unoptimised control point. It copies the live variables and
/// restores the registers.
///
/// # Arguments
///
/// * `frameaddr` - Pointer to the current frame address
/// * `stats` - Statistics tracker for recording the transition
pub(crate) unsafe fn cp_transition_to_unopt(frameaddr: *const c_void, stats: &Stats) {
    // TODO: add cache for the asm code generation
    let buffer = generate_transition_asm(CPTransition {
        smid: ControlPointStackMapId::Opt,
        frameaddr,
        trace_addr: 0 as *const c_void,
    });
    stats.swt_transition_opt_to_unopt();
    unsafe {
        execute_asm_buffer(buffer);
    }
}

/// Transitions from unoptimised to optimised execution mode at a control point.
///
/// This function generates and executes assembly code to transition from the unoptimised
/// control point to the optimised control point. It copies the live variables and
/// restores the registers.
///
/// # Arguments
///
/// * `frameaddr` - Pointer to the current frame address  
/// * `stats` - Statistics tracker for recording the transition
pub(crate) unsafe fn cp_transition_to_opt(frameaddr: *const c_void, stats: &Stats) {
    // TODO: add cache for the asm code generation
    let buffer = generate_transition_asm(CPTransition {
        smid: ControlPointStackMapId::UnOpt,
        frameaddr,
        trace_addr: 0 as *const c_void,
    });
    stats.swt_transition_unopt_to_opt();
    unsafe {
        execute_asm_buffer(buffer);
    }
}

/// Transitions from optimised to unoptimised execution mode at a control point and executes a trace.
///
/// This function generates and executes assembly code to transition from the optimised
/// control point to the unoptimised control point and executes a trace. It copies the live variables and
/// restores the registers.
///
/// # Arguments
///
/// * `frameaddr` - Pointer to the current frame address
/// * `trace_addr` - Pointer to the trace to execute
/// * `stats` - Statistics tracker for recording the transition
pub(crate) unsafe fn cp_transition_to_unopt_and_exec_trace(
    frameaddr: *const c_void,
    trace_addr: *const c_void,
    stats: &Stats,
) {
    // TODO: add cache for the asm code generation
    let buffer = generate_transition_asm(CPTransition {
        smid: ControlPointStackMapId::Opt,
        frameaddr,
        trace_addr,
    });
    stats.swt_transition_opt_to_unopt();
    unsafe {
        execute_asm_buffer(buffer);
    }
}

/// Execute an assembled buffer with optional verbose assembly dumping
#[unsafe(no_mangle)]
unsafe fn execute_asm_buffer(buffer: ExecutableBuffer) {
    if *YKD_SWT_VERBOSE_ASM {
        let cs = Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .build()
            .unwrap();

        let instructions = cs
            .disasm_all(
                unsafe {
                    std::slice::from_raw_parts(
                        buffer.ptr(dynasmrt::AssemblyOffset(0)) as *const u8,
                        buffer.len(),
                    )
                },
                0,
            )
            .unwrap();

        println!("ASM DUMP:");
        for i in instructions.iter() {
            println!(
                "  {:x}: {} {}",
                i.address(),
                i.mnemonic().unwrap(),
                i.op_str().unwrap()
            );
        }
    }
    unsafe {
        let func: unsafe fn() = std::mem::transmute(buffer.as_ptr());
        func();
    }
}

fn generate_transition_asm(transition: CPTransition) -> ExecutableBuffer {
    let frameaddr = transition.frameaddr as usize;
    let mut asm = Assembler::new().unwrap();

    let src_smid: ControlPointStackMapId;
    let dst_smid: ControlPointStackMapId;
    
    if transition.smid == ControlPointStackMapId::Opt {
        src_smid = ControlPointStackMapId::Opt;
        dst_smid = ControlPointStackMapId::UnOpt;
    } else {
        src_smid = ControlPointStackMapId::UnOpt;
        dst_smid = ControlPointStackMapId::Opt;
    }

    let (src_rec, src_pinfo) = AOT_STACKMAPS.as_ref().unwrap().get(src_smid as usize);
    let (dst_rec, dst_pinfo) = AOT_STACKMAPS.as_ref().unwrap().get(dst_smid as usize);

    let mut src_frame_size: u64 = src_rec.size;
    if src_pinfo.hasfp {
        src_frame_size -= REG64_BYTESIZE;
    }
    let mut dst_frame_size: u64 = dst_rec.size;
    if dst_pinfo.hasfp {
        dst_frame_size -= REG64_BYTESIZE;
    }
    if *YKD_SWT_CP_BREAK {
        dynasm!(asm
            ; .arch x64
            ; int3
        );
    }

    // Set RBP and RSP
    dynasm!(asm
        ; .arch x64
        ; mov rbp, QWORD frameaddr as i64
        ; mov rsp, QWORD frameaddr as i64
        ; sub rsp, (dst_frame_size).try_into().unwrap() // adjust rsp
    );

    // Calculate the offset from the RBP to the RSP where __ykrt_control_point_real stored the registers.
    // Example: r15 address = rbp - rbp_offset_reg_store
    let rbp_offset_reg_store = src_frame_size as i64 + (14 * REG64_BYTESIZE) as i64;

    let temp_live_vars_buffer = copy_live_vars_to_temp_buffer(&mut asm, src_rec, transition.smid);
    if *YKD_SWT_VERBOSE {
        eprintln!(
            "Transition from {:?} to {:?}, trace: {:?}",
            src_smid, dst_smid, transition.trace_addr
        );
        eprintln!(
            "src_rbp: 0x{:x}, reg_store: 0x{:x}, src_frame_size: 0x{:x}, dst_frame_size: 0x{:x}, rbp_offset_reg_store: 0x{:x}",
            frameaddr as i64,
            frameaddr as i64 - rbp_offset_reg_store,
            src_frame_size,
            dst_frame_size,
            rbp_offset_reg_store
        );
    }

    // Set destination live vars
    let used_registers = set_destination_live_vars(
        &mut asm,
        src_rec,
        dst_rec,
        rbp_offset_reg_store,
        temp_live_vars_buffer.clone(),
    );

    assert_eq!(
        (frameaddr as i64 - dst_frame_size as i64) % 16,
        0,
        "RSP is not aligned to 16 bytes"
    );
    // Restore unused registers.
    restore_registers(&mut asm, used_registers, rbp_offset_reg_store as i32);

    // If there is a trace to execute, jump to the trace.
    if transition.trace_addr != 0 as *const c_void {
        dynasm!(asm
            ; .arch x64
            ; mov rdx, QWORD transition.trace_addr as i64
            ; jmp rdx
        );
    } else {
        let call_offset = calc_after_cp_offset(dst_rec.offset).unwrap();
        let dst_target_addr = i64::try_from(dst_rec.offset).unwrap() + call_offset;
        dynasm!(asm
            ; .arch x64
            ; sub rsp, 0x10 // Allocate 16 bytes on the stack
            ; mov [rsp], rax // Save the original rsp at 0x0
            ; mov rax, QWORD dst_target_addr
            ; mov [rsp + 0x8], rax // Store the target address into 0x8
            ; pop rax // Restore the original rax
            ; ret
        );
    }

    asm.finalize().unwrap()
}

// Restores the registers from the rbp offset.
fn restore_registers(
    asm: &mut Assembler,
    exclude_registers: HashMap<u16, u16>,
    rbp_offset_reg_store: i32,
) {
    let mut sorted_offsets: Vec<(&u16, &i32)> = REG_OFFSETS.iter().collect();
    sorted_offsets.sort_by(|a, b| b.1.cmp(a.1)); // Sort descending by value

    for (dwarf_reg_num, _) in sorted_offsets.iter() {
        if !exclude_registers.contains_key(dwarf_reg_num) {
            restore_register(
                asm,
                (**dwarf_reg_num).try_into().unwrap(),
                rbp_offset_reg_store,
            );
        }
    }
}

fn restore_register(asm: &mut Assembler, dwarf_reg_num: u16, rbp_offset_reg_store: i32) {
    let reg_offset = REG_OFFSETS.get(&dwarf_reg_num).unwrap();
    let reg_val_rbp_offset = i32::try_from(rbp_offset_reg_store - reg_offset).unwrap();
    let dynasm_reg = dwarf_to_dynasm_reg(dwarf_reg_num.try_into().unwrap());
    dynasm!(asm
        ; mov Rq(dynasm_reg), QWORD [rbp - reg_val_rbp_offset]
    );
}

// Calculates the offset of the call instruction after the control point.
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
#[cfg(swt_modclone)]
mod cp_tests {
    use super::*;
    use crate::trace::swt::asm::{disassemble, verify_instruction_sequence};
    use dynasmrt::x64::Assembler;

    #[test]
    fn test_restore_registers_rbx() {
        let mut asm = Assembler::new().unwrap();
        let mut used_regs = HashMap::new();
        used_regs.insert(0, 8);
        // used_regs.insert(1, 8); // not used:
        used_regs.insert(2, 8);
        // used_regs.insert(3, 8); // used
        used_regs.insert(4, 8);
        used_regs.insert(5, 8);
        // used_regs.insert(6, 8); // not used:
        // used_regs.insert(7, 8); // not used:
        used_regs.insert(8, 8);
        used_regs.insert(9, 8);
        used_regs.insert(10, 8);
        used_regs.insert(11, 8);
        used_regs.insert(12, 8);
        used_regs.insert(13, 8);
        used_regs.insert(14, 8);
        used_regs.insert(15, 8);

        restore_registers(&mut asm, used_regs, 0);
        let buffer: dynasmrt::ExecutableBuffer = asm.finalize().unwrap();
        let instructions = disassemble(&buffer);
        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0], "mov rbx, qword ptr [rbp + 0x50]");
    }

    #[test]
    fn test_restore_registers_no_instructions() {
        let mut asm = Assembler::new().unwrap();
        let mut used_regs = HashMap::new();
        used_regs.insert(0, 8);
        // used_regs.insert(1, 8); // not used:
        used_regs.insert(2, 8);
        used_regs.insert(3, 8);
        used_regs.insert(4, 8);
        used_regs.insert(5, 8);
        // used_regs.insert(6, 8); // not used:
        // used_regs.insert(7, 8); // not used:
        used_regs.insert(8, 8);
        used_regs.insert(9, 8);
        used_regs.insert(10, 8);
        used_regs.insert(11, 8);
        used_regs.insert(12, 8);
        used_regs.insert(13, 8);
        used_regs.insert(14, 8);
        used_regs.insert(15, 8);

        restore_registers(&mut asm, used_regs, 0);
        let buffer: dynasmrt::ExecutableBuffer = asm.finalize().unwrap();
        let instructions = disassemble(&buffer);
        assert_eq!(instructions.len(), 0);
    }

    #[test]
    fn test_restore_registers_partial() {
        let mut asm = Assembler::new().unwrap();
        let mut used_regs = HashMap::new();
        used_regs.insert(0, 8);
        // used_regs.insert(1, 8); // not used:
        used_regs.insert(2, 8);
        used_regs.insert(3, 8);
        used_regs.insert(4, 8);
        used_regs.insert(5, 8);
        // used_regs.insert(6, 8); // not used
        // used_regs.insert(7, 8); // not used
        used_regs.insert(8, 8);
        used_regs.insert(9, 8);
        // used_regs.insert(10, 8); // not used
        used_regs.insert(11, 8);
        used_regs.insert(12, 8);
        used_regs.insert(13, 8);
        // used_regs.insert(14, 8); // not used
        used_regs.insert(15, 8);

        restore_registers(&mut asm, used_regs, 0);
        let buffer = asm.finalize().unwrap();
        let instructions = disassemble(&buffer);

        let expected_instructions = [
            "mov r10, qword ptr [rbp + 0x28]",
            "mov r14, qword ptr [rbp + 8]",
        ];

        verify_instruction_sequence(&instructions, &expected_instructions);
    }

    #[test]
    fn test_restore_registers_empty_restore() {
        let mut asm = Assembler::new().unwrap();
        let used_regs = HashMap::new();
        restore_registers(&mut asm, used_regs, 0);
        let buffer: dynasmrt::ExecutableBuffer = asm.finalize().unwrap();
        let instructions = disassemble(&buffer);

        let expected_instructions = [
            "mov rax, qword ptr [rbp + 0x60]",
            "mov rcx, qword ptr [rbp + 0x58]",
            "mov rbx, qword ptr [rbp + 0x50]",
            "mov rdi, qword ptr [rbp + 0x48]",
            "mov rsi, qword ptr [rbp + 0x40]",
            "mov r8, qword ptr [rbp + 0x38]",
            "mov r9, qword ptr [rbp + 0x30]",
            "mov r10, qword ptr [rbp + 0x28]",
            "mov r11, qword ptr [rbp + 0x20]",
            "mov r12, qword ptr [rbp + 0x18]",
            "mov r13, qword ptr [rbp + 0x10]",
            "mov r14, qword ptr [rbp + 8]",
            "mov r15, qword ptr [rbp]",
        ];

        verify_instruction_sequence(&instructions, &expected_instructions);
    }

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
        let offset = calc_after_cp_offset(code_ptr)?;
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
        let offset = calc_after_cp_offset(code_ptr)?;
        assert_eq!(offset, 11, "The call offset should be 11 bytes");
        Ok(())
    }

    // Tests moved from cfg.rs
    #[test]
    fn test_dwarf_to_dynasm_reg_all_valid_registers() {
        // Test all valid DWARF register numbers and their expected DynASM mappings
        let test_cases = [
            (0, 0),   // RAX -> RAX
            (1, 2),   // RDX -> RDX (note: different order)
            (2, 1),   // RCX -> RCX (note: different order)
            (3, 3),   // RBX -> RBX
            (4, 6),   // RSI -> RSI (note: different order)
            (5, 7),   // RDI -> RDI (note: different order)
            (6, 5),   // RBP -> RBP (note: different order)
            (7, 4),   // RSP -> RSP (note: different order)
            (8, 8),   // R8 -> R8
            (9, 9),   // R9 -> R9
            (10, 10), // R10 -> R10
            (11, 11), // R11 -> R11
            (12, 12), // R12 -> R12
            (13, 13), // R13 -> R13
            (14, 14), // R14 -> R14
            (15, 15), // R15 -> R15
        ];

        for (dwarf_reg, expected_dynasm_reg) in test_cases.iter() {
            assert_eq!(
                dwarf_to_dynasm_reg(*dwarf_reg),
                *expected_dynasm_reg,
                "DWARF register {} should map to DynASM register {}",
                dwarf_reg,
                expected_dynasm_reg
            );
        }
    }

    #[test]
    #[should_panic(expected = "Unsupported DWARF register number: 16")]
    fn test_dwarf_to_dynasm_reg_invalid_16() {
        let _ = dwarf_to_dynasm_reg(16);
    }

    #[test]
    #[should_panic(expected = "Unsupported DWARF register number: 255")]
    fn test_dwarf_to_dynasm_reg_invalid_255() {
        let _ = dwarf_to_dynasm_reg(255);
    }

    #[test]
    fn test_reg_offsets_contains_expected_registers() {
        let expected_registers = [0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15];

        for reg in expected_registers.iter() {
            assert!(
                REG_OFFSETS.contains_key(reg),
                "Register {} should have an offset mapping",
                reg
            );
        }

        // Verify that unsaved registers are not in the mapping
        let unsaved_registers = [1, 6, 7]; // rdx, rbp, rsp
        for reg in unsaved_registers.iter() {
            assert!(
                !REG_OFFSETS.contains_key(reg),
                "Register {} should not have an offset mapping (it's not saved)",
                reg
            );
        }
    }

    #[test]
    fn test_reg_offsets_are_properly_ordered() {
        // Verify that register offsets follow expected stack layout
        // Higher registers should have lower (closer to stack top) offsets
        assert!(REG_OFFSETS[&15] < REG_OFFSETS[&14]); // r15 < r14
        assert!(REG_OFFSETS[&14] < REG_OFFSETS[&13]); // r14 < r13
        assert!(REG_OFFSETS[&0] > REG_OFFSETS[&2]); // rax > rcx (rax is saved last)
    }
}
