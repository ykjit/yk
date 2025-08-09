//! # Live Variable Transfer for Control Point Transitions
//!
//! This module handles the task of transferring live variables between different
//! function variants during control point transitions.
//!
//! ## Purpose
//!
//! When transitioning between optimised and unoptimised control points we need to maintain
//! consistent state of the program. However, the same logical variable values may be
//! stored in different locations. The purpose of this module is to copy these values from
//! source locations to destination locations.
//!
//! The live variable transfer process operates in two main phases:
//!
//! ### Phase 1: Temporary Storage ([`copy_live_vars_to_temp_buffer`])
//! - Copies `Indirect` and `Direct` variables from stack locations to a temporary buffer
//!
//! ### Phase 2: Destination Placement ([`set_destination_live_vars`])
//! - Moves variables from temporary storage and registers to their destination locations
//!   including additional locations.
//! - Manages temporary register conflicts by deferring conflicting assignments.
//!
//! ## Assembly Generation
//! This module generates straight-line assembly code tailored to each specific transfer pattern.

use dynasmrt::{DynasmApi, dynasm, x64::Assembler};
use smallvec::SmallVec;
use std::alloc::Layout;
use std::collections::HashMap;
use yksmp::Location::{Direct, Indirect, Register};
use yksmp::Record;

use crate::trace::swt::buffer::LiveVarsBuffer;
use crate::trace::swt::cp::{
    ControlPointStackMapId, REG_OFFSETS, YKD_SWT_VERBOSE, dwarf_to_dynasm_reg,
};

// Primary temporary register - used in buffer copy and destination live vars copy.
static TEMP_REG_PRIMARY_DWARF: u8 = 0; // RAX

// Dwarf register number for RCX
static TEMP_REG_SECONDARY_DWARF: u8 = 2;

#[derive(Debug, Clone)]
struct RestoreTempRegisters<'a> {
    src_location: &'a yksmp::Location,
    dst_location: &'a yksmp::Location,
    src_var_indirect_index: i32,
}

struct MemToRegParams {
    src_ptr: i64,
    src_offset: i32,
    dst_reg: u8,
    size: u16,
}

struct MemToMemParams {
    src_ptr: i64,
    src_offset: i32,
    dst_offset: i32,
    size: u16,
}

struct RbpToRegParams {
    rbp_offset: i32,
    dst_reg: u8,
    size: u16,
}

struct RegToRbpParams {
    src_reg: u8,
    rbp_offset: i32,
    size: u16,
}

/// Emits asm instructions to load a value from an memory address into a register.
fn emit_mem_to_reg(asm: &mut Assembler, params: MemToRegParams) {
    match params.size {
        1 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rb(params.dst_reg), BYTE [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]),
        2 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rw(params.dst_reg), WORD [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]),
        4 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rd(params.dst_reg), DWORD [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]),
        8 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rq(params.dst_reg), QWORD [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]),
        _ => panic!("Unsupported value size: {}", params.size),
    }
}

/// Emits asm instructions to copy a value from a memory address into an RBP-relative stack slot.
fn emit_mem_to_mem(asm: &mut Assembler, params: MemToMemParams) {
    match params.size {
        1 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rb(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), BYTE [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]
            ; mov BYTE [rbp + params.dst_offset], Rb(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
        ),
        2 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rw(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), WORD [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]
            ; mov WORD [rbp + params.dst_offset], Rw(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
        ),
        4 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rd(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), DWORD [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]
            ; mov DWORD [rbp + params.dst_offset], Rd(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
        ),
        8 => dynasm!(asm
            ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD params.src_ptr
            ; mov Rq(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), QWORD [Rq(TEMP_REG_PRIMARY_DWARF) + params.src_offset]
            ; mov QWORD [rbp + params.dst_offset], Rq(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
        ),
        _ => panic!("Unsupported value size: {}", params.size),
    }
}

/// Emits asm instructions to load a value from an RBP-relative stack slot into a register.
fn emit_rbp_to_reg(asm: &mut Assembler, params: RbpToRegParams) {
    match params.size {
        1 => dynasm!(asm; mov Rb(params.dst_reg), BYTE [rbp - params.rbp_offset]),
        2 => dynasm!(asm; mov Rw(params.dst_reg), WORD [rbp - params.rbp_offset]),
        4 => dynasm!(asm; mov Rd(params.dst_reg), DWORD [rbp - params.rbp_offset]),
        8 => dynasm!(asm; mov Rq(params.dst_reg), QWORD [rbp - params.rbp_offset]),
        _ => panic!("Unsupported value size: {}", params.size),
    }
}

/// Emits asm instructions to store a register into an RBP-relative stack slot.
fn emit_reg_to_rbp(asm: &mut Assembler, params: RegToRbpParams) {
    match params.size {
        1 => dynasm!(asm; mov BYTE [rbp + params.rbp_offset], Rb(params.src_reg)),
        2 => dynasm!(asm; mov WORD [rbp + params.rbp_offset], Rw(params.src_reg)),
        4 => dynasm!(asm; mov DWORD [rbp + params.rbp_offset], Rd(params.src_reg)),
        8 => dynasm!(asm; mov QWORD [rbp + params.rbp_offset], Rq(params.src_reg)),
        _ => panic!("Unsupported value size: {}", params.size),
    }
}

/// Emits additional destination copies for a register-to-register transfer.
fn handle_register_to_register_additional_locations(
    asm: &mut dynasmrt::Assembler<dynasmrt::x64::X64Relocation>,
    reg_store_rbp_offset: i32,
    dst_add_locs: &SmallVec<[i16; 1]>,
    src_val_size: &u16,
    dest_reg_nums: &mut HashMap<u16, u16>,
) {
    // Handle additional locations for the destination register
    for add_loc in dst_add_locs {
        if *add_loc >= 0 {
            // Additional location is a register
            let dst_reg = dwarf_to_dynasm_reg((*add_loc).try_into().unwrap());
            emit_rbp_to_reg(
                asm,
                RbpToRegParams {
                    rbp_offset: reg_store_rbp_offset,
                    dst_reg,
                    size: *src_val_size,
                },
            );
            dest_reg_nums.insert((*add_loc).try_into().unwrap(), *src_val_size);
        } else {
            // Additional location is a stack offset - CRITICAL FIX: write value to stack location
            // First load the register value to a temporary register
            emit_rbp_to_reg(
                asm,
                RbpToRegParams {
                    rbp_offset: reg_store_rbp_offset,
                    dst_reg: TEMP_REG_PRIMARY_DWARF,
                    size: *src_val_size,
                },
            );
            // Then store the value to the additional stack location
            emit_reg_to_rbp(
                asm,
                RegToRbpParams {
                    src_reg: TEMP_REG_PRIMARY_DWARF,
                    rbp_offset: i32::from(*add_loc),
                    size: *src_val_size,
                },
            );
        }
    }
}

/// Emits additional location copies for an indirect-to-register transfer.
fn handle_indirect_to_register_additional_locations(
    asm: &mut dynasmrt::Assembler<dynasmrt::x64::X64Relocation>,
    dst_add_locs: &SmallVec<[i16; 1]>,
    src_val_size: &u16,
    temp_buffer_offset: i32,
    live_vars_buffer: &LiveVarsBuffer,
    dest_reg_nums: &mut HashMap<u16, u16>,
) {
    for location in dst_add_locs {
        // Write any additional locations that were tracked for this variable.
        // Numbers greater or equal to zero are registers in Dwarf notation.
        // Negative numbers are offsets relative to RBP.
        if *location >= 0 {
            dest_reg_nums.insert(*location as u16, *src_val_size);
            let dst_reg = dwarf_to_dynasm_reg((*location).try_into().unwrap());
            emit_mem_to_reg(
                asm,
                MemToRegParams {
                    src_ptr: live_vars_buffer.ptr as i64,
                    src_offset: temp_buffer_offset,
                    dst_reg,
                    size: *src_val_size,
                },
            );
        } else {
            emit_mem_to_mem(
                asm,
                MemToMemParams {
                    src_ptr: live_vars_buffer.ptr as i64,
                    src_offset: temp_buffer_offset,
                    dst_offset: i32::try_from(*location).unwrap(),
                    size: *src_val_size,
                },
            );
        }
    }
}

/// Copies live variables values from source record to destination record.
/// Returns a map of destination DWARF register numbers to the byte size 
/// of the value written.
pub(crate) fn set_destination_live_vars(
    asm: &mut Assembler,
    src_rec: &Record,
    dst_rec: &Record,
    rbp_offset_reg_store: i64,
    live_vars_buffer: LiveVarsBuffer,
) -> HashMap<u16, u16> {
    // Map of destination register numbers to their value sizes.
    let mut dest_reg_nums = HashMap::new();
    // List of temporary registers to restore.
    let mut used_temp_reg_dist = Vec::new();
    // Index of the source live variable in the temporary buffer.
    let mut src_var_indirect_index = 0;

    if *YKD_SWT_VERBOSE {
        eprintln!(
            "Copying live vars: src={}, dst={}",
            src_rec.live_vals.len(),
            dst_rec.live_vals.len()
        );
        for (index, src_var) in src_rec.live_vals.iter().enumerate() {
            let dst_var = &dst_rec.live_vals[index];
            if src_var.len() > 1 || dst_var.len() > 1 {
                todo!("Deal with multi register locations");
            }
            let src_location = &src_var.get(0).unwrap();
            let dst_location = &dst_var.get(0).unwrap();
            eprintln!(
                "Copying live src: {:?}, dst: {:?}",
                src_location, dst_location
            );
        }
    }
    // Ensure we have matching live variables
    assert!(
        src_rec.live_vals.len() == dst_rec.live_vals.len(),
        "Source and destination live variable counts don't match: src={}, dst={}",
        src_rec.live_vals.len(),
        dst_rec.live_vals.len()
    );

    for (index, src_var) in src_rec.live_vals.iter().enumerate() {
        let dst_var = &dst_rec.live_vals[index];
        if src_var.len() > 1 || dst_var.len() > 1 {
            todo!("Deal with multi register locations");
        }

        let src_location = &src_var.get(0).unwrap();
        let dst_location = &dst_var.get(0).unwrap();
        match src_location {
            Register(src_reg_num, src_val_size, _src_add_locs) => {
                let reg_store_offset = REG_OFFSETS.get(src_reg_num).unwrap();
                let reg_store_rbp_offset =
                    i32::try_from(rbp_offset_reg_store - *reg_store_offset as i64).unwrap();
                match dst_location {
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        if *dst_reg_num == TEMP_REG_PRIMARY_DWARF.into()
                            || *dst_reg_num == TEMP_REG_SECONDARY_DWARF.into()
                        {
                            used_temp_reg_dist.push(RestoreTempRegisters {
                                src_location: src_location,
                                dst_location: dst_location,
                                src_var_indirect_index: src_var_indirect_index,
                            });
                        } else {
                            handle_register_to_register_additional_locations(
                                asm,
                                reg_store_rbp_offset,
                                dst_add_locs,
                                src_val_size,
                                &mut dest_reg_nums,
                            );

                            assert!(
                                dst_val_size == src_val_size,
                                "Register2Register - src and dst val size must match. Got src: {} and dst: {}",
                                src_val_size,
                                dst_val_size
                            );
                            dest_reg_nums.insert(*dst_reg_num, *dst_val_size);
                            let dst_reg = dwarf_to_dynasm_reg((*dst_reg_num).try_into().unwrap());
                            emit_rbp_to_reg(
                                asm,
                                RbpToRegParams {
                                    rbp_offset: reg_store_rbp_offset,
                                    dst_reg,
                                    size: *src_val_size,
                                },
                            );
                        }
                    }
                    Indirect(_dst_reg_num, dst_off, dst_val_size) => {
                        emit_rbp_to_reg(
                            asm,
                            RbpToRegParams {
                                rbp_offset: reg_store_rbp_offset,
                                dst_reg: TEMP_REG_PRIMARY_DWARF,
                                size: *src_val_size,
                            },
                        );
                        emit_reg_to_rbp(
                            asm,
                            RegToRbpParams {
                                src_reg: TEMP_REG_PRIMARY_DWARF,
                                rbp_offset: i32::try_from(*dst_off).unwrap(),
                                size: *dst_val_size,
                            },
                        );
                    }
                    _ => panic!(
                        "Unexpected target for Register source location - src: {:?}, dst: {:?}",
                        src_location, dst_location
                    ),
                }
            }
            Indirect(src_reg_num, _src_off, src_val_size)
            | Direct(src_reg_num, _src_off, src_val_size) => {
                assert!(!live_vars_buffer.ptr.is_null(), "Live vars buffer is null");
                let temp_buffer_offset = src_var_indirect_index * (*src_val_size as i32);
                match dst_location {
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        if *dst_reg_num == TEMP_REG_PRIMARY_DWARF.into()
                            || *dst_reg_num == TEMP_REG_SECONDARY_DWARF.into()
                        {
                            used_temp_reg_dist.push(RestoreTempRegisters {
                                src_location: src_location,
                                dst_location: dst_location,
                                src_var_indirect_index: src_var_indirect_index,
                            });
                        } else {
                            assert!(*src_reg_num == 6, "Indirect register is expected to be rbp");
                            let dst_reg = dwarf_to_dynasm_reg((*dst_reg_num).try_into().unwrap());

                            handle_indirect_to_register_additional_locations(
                                asm,
                                dst_add_locs,
                                src_val_size,
                                temp_buffer_offset,
                                &live_vars_buffer,
                                &mut dest_reg_nums,
                            );
                            dest_reg_nums.insert(*dst_reg_num, *dst_val_size);

                            emit_mem_to_reg(
                                asm,
                                MemToRegParams {
                                    src_ptr: live_vars_buffer.ptr as i64,
                                    src_offset: temp_buffer_offset,
                                    dst_reg,
                                    size: *src_val_size,
                                },
                            );
                        }
                    }
                    Indirect(_, dst_off, dst_val_size) | Direct(_, dst_off, dst_val_size) => {
                        emit_mem_to_mem(
                            asm,
                            MemToMemParams {
                                src_ptr: live_vars_buffer.ptr as i64,
                                src_offset: temp_buffer_offset,
                                dst_offset: i32::try_from(*dst_off).unwrap(),
                                size: *dst_val_size,
                            },
                        );
                    }
                    _ => panic!(
                        "Unexpected target for Indirect source location - src: {:?}, dst: {:?}",
                        src_location, dst_location
                    ),
                }
                src_var_indirect_index += 1;
            }
            _ => panic!("Unexpected source location: {:?}", src_location),
        }
    }

    // Handle restoration of used temporary registers.
    for temp_reg in used_temp_reg_dist {
        match temp_reg.src_location {
            Register(src_reg_num, src_val_size, _src_add_locs) => {
                let reg_store_offset = REG_OFFSETS.get(src_reg_num).unwrap();
                let reg_store_rbp_offset =
                    i32::try_from(rbp_offset_reg_store - *reg_store_offset as i64).unwrap();

                match temp_reg.dst_location {
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        handle_register_to_register_additional_locations(
                            asm,
                            reg_store_rbp_offset,
                            dst_add_locs,
                            src_val_size,
                            &mut dest_reg_nums,
                        );

                        assert!(
                            dst_val_size == src_val_size,
                            "Register2Register - src and dst val size must match. Got src: {} and dst: {}",
                            src_val_size,
                            dst_val_size
                        );
                        dest_reg_nums.insert(*dst_reg_num, *dst_val_size);
                        let dst_reg = dwarf_to_dynasm_reg((*dst_reg_num).try_into().unwrap());
                        emit_rbp_to_reg(
                            asm,
                            RbpToRegParams {
                                rbp_offset: reg_store_rbp_offset,
                                dst_reg,
                                size: *src_val_size,
                            },
                        );
                    }
                    _ => panic!(
                        "Unexpected destination for temporary register restoration: {:?}",
                        temp_reg.dst_location
                    ),
                }
            }
            Indirect(_, _, src_val_size) | Direct(_, _, src_val_size) => {
                assert!(!live_vars_buffer.ptr.is_null(), "Live vars buffer is null");
                let temp_buffer_offset =
                    live_vars_buffer.variables[&temp_reg.src_var_indirect_index];

                match temp_reg.dst_location {
                    Register(dst_reg_num, dst_val_size, dst_add_locs) => {
                        let dst_reg = dwarf_to_dynasm_reg((*dst_reg_num).try_into().unwrap());
                        handle_indirect_to_register_additional_locations(
                            asm,
                            dst_add_locs,
                            src_val_size,
                            temp_buffer_offset,
                            &live_vars_buffer,
                            &mut dest_reg_nums,
                        );
                        dest_reg_nums.insert(*dst_reg_num, *dst_val_size);

                        emit_mem_to_reg(
                            asm,
                            MemToRegParams {
                                src_ptr: live_vars_buffer.ptr as i64,
                                src_offset: temp_buffer_offset,
                                dst_reg,
                                size: *src_val_size,
                            },
                        );
                    }
                    _ => panic!(
                        "Unexpected destination for temporary register restoration: {:?}",
                        temp_reg.dst_location
                    ),
                }
            }
            _ => panic!(
                "Unexpected source for temporary register restoration: {:?}",
                temp_reg.src_location
            ),
        }
    }

    dest_reg_nums
}

/// Copies Indirect and Direct live variables from RBP-relative stack slots into a
/// temporary buffer.
/// If no stack-based variables are present for `src_rec`, an empty buffer descriptor is
/// returned with a null `ptr` and zero size.
pub(crate) fn copy_live_vars_to_temp_buffer(
    asm: &mut Assembler,
    src_rec: &Record,
    smid: ControlPointStackMapId,
) -> LiveVarsBuffer {
    let (ptr, layout, size) = LiveVarsBuffer::get_or_create(src_rec, smid);
    if ptr.is_null() {
        return LiveVarsBuffer {
            ptr: std::ptr::null_mut(),
            layout: Layout::new::<u8>(),
            variables: HashMap::new(),
            size: 0,
        };
    }
    if *YKD_SWT_VERBOSE {
        println!("Using buffer at {:p} for smid {:?}", ptr, smid);
    }

    let mut src_var_indirect_index = 0;
    let mut variables = HashMap::new();
    let mut current_buffer_offset = 0i32;

    dynasm!(asm
        ; mov Rq(TEMP_REG_PRIMARY_DWARF), QWORD ptr as i64
    );

    for (_, src_var) in src_rec.live_vals.iter().enumerate() {
        let src_location = src_var.get(0).unwrap();
        match src_location {
            Indirect(_, src_off, src_val_size) | Direct(_, src_off, src_val_size) => {
                let src_rbp_offset = i32::try_from(*src_off).unwrap();
                // Different handling based on size
                match *src_val_size {
                    1 => dynasm!(asm
                        ; mov Rb(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), BYTE [rbp + src_rbp_offset]
                        ; mov BYTE [Rq(TEMP_REG_PRIMARY_DWARF) + current_buffer_offset], Rb(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
                    ),
                    2 => dynasm!(asm
                        ; mov Rw(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), WORD [rbp + src_rbp_offset]
                        ; mov WORD [Rq(TEMP_REG_PRIMARY_DWARF) + current_buffer_offset], Rw(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
                    ),
                    4 => dynasm!(asm
                        ; mov Rd(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), DWORD [rbp + src_rbp_offset]
                        ; mov DWORD [Rq(TEMP_REG_PRIMARY_DWARF) + current_buffer_offset], Rd(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
                    ),
                    8 => dynasm!(asm
                        ; mov Rq(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF)), QWORD [rbp + src_rbp_offset]
                        ; mov QWORD [Rq(TEMP_REG_PRIMARY_DWARF) + current_buffer_offset], Rq(dwarf_to_dynasm_reg(TEMP_REG_SECONDARY_DWARF))
                    ),
                    _ => panic!("Unsupported value size in temporary copy: {}", src_val_size),
                }

                variables.insert(src_var_indirect_index, current_buffer_offset);
                current_buffer_offset += *src_val_size as i32; // Move to next position
                src_var_indirect_index += 1;
            }
            Register(_, _, _) => { /* DO NOTHING */ }
            _ => panic!(
                "Unsupported source location in temporary copy: {:?}",
                src_location /* DO NOTHING */
            ),
        }
    }

    if *YKD_SWT_VERBOSE {
        eprintln!(
            "Buffer population complete. Variables mapping: {:?}",
            variables
        );
    }

    LiveVarsBuffer::new(ptr, layout, size, variables)
}

#[cfg(test)]
#[cfg(swt_modclone)]
mod tests {
    use super::*;
    use crate::trace::swt::asm::{
        disassemble, verify_instruction_at_position, verify_instruction_sequence,
    };
    use crate::trace::swt::cp::{REG_OFFSETS, REG64_BYTESIZE};
    use dynasmrt::x64::Assembler;
    use yksmp::{Location, Record};

    /// Tests for assembly emission helper functions
    mod helper_functions {
        use super::*;

        /// Tests for memory-to-register operations
        mod emit_operations {
            use super::*;

            #[test]
            fn test_emit_mem_to_reg() {
                for size in [1, 2, 4, 8].iter() {
                    let mut asm = Assembler::new().unwrap();
                    let test_ptr = 0x1234567890ABCDEF;
                    let test_offset = 42;
                    let test_dst_reg = 15; // r15

                    emit_mem_to_reg(
                        &mut asm,
                        MemToRegParams {
                            src_ptr: test_ptr,
                            src_offset: test_offset,
                            dst_reg: test_dst_reg,
                            size: *size,
                        },
                    );

                    let buffer = asm.finalize().unwrap();
                    let instructions = disassemble(&buffer);

                    assert_eq!(
                        instructions.len(),
                        2,
                        "Should have exactly 2 instructions for size {}",
                        size
                    );

                    assert_eq!(
                        instructions[0],
                        format!("mov rax, 0x{:x}", test_ptr),
                        "First instruction should load the pointer for size {}",
                        size
                    );

                    let expected_second = match size {
                        1 => format!("mov r15b, byte ptr [rax + 0x{:x}]", test_offset),
                        2 => format!("mov r15w, word ptr [rax + 0x{:x}]", test_offset),
                        4 => format!("mov r15d, dword ptr [rax + 0x{:x}]", test_offset),
                        8 => format!("mov r15, qword ptr [rax + 0x{:x}]", test_offset),
                        _ => unreachable!(),
                    };

                    assert_eq!(
                        instructions[1], expected_second,
                        "Second instruction should load value of size {} into register",
                        size
                    );
                }
            }

            #[test]
            fn test_emit_mem_to_mem() {
                for size in [1, 2, 4, 8].iter() {
                    let mut asm = Assembler::new().unwrap();
                    let test_ptr = 0x1234567890ABCDEF;
                    let test_src_offset = 42;
                    let test_dst_offset = 24;

                    emit_mem_to_mem(
                        &mut asm,
                        MemToMemParams {
                            src_ptr: test_ptr,
                            src_offset: test_src_offset,
                            dst_offset: test_dst_offset,
                            size: *size,
                        },
                    );

                    let buffer = asm.finalize().unwrap();
                    let instructions = disassemble(&buffer);

                    let expected_instructions = match size {
                        1 => [
                            format!("mov rax, 0x{:x}", test_ptr),
                            format!("mov cl, byte ptr [rax + 0x{:x}]", test_src_offset),
                            format!("mov byte ptr [rbp + 0x{:x}], cl", test_dst_offset),
                        ],
                        2 => [
                            format!("mov rax, 0x{:x}", test_ptr),
                            format!("mov cx, word ptr [rax + 0x{:x}]", test_src_offset),
                            format!("mov word ptr [rbp + 0x{:x}], cx", test_dst_offset),
                        ],
                        4 => [
                            format!("mov rax, 0x{:x}", test_ptr),
                            format!("mov ecx, dword ptr [rax + 0x{:x}]", test_src_offset),
                            format!("mov dword ptr [rbp + 0x{:x}], ecx", test_dst_offset),
                        ],
                        8 => [
                            format!("mov rax, 0x{:x}", test_ptr),
                            format!("mov rcx, qword ptr [rax + 0x{:x}]", test_src_offset),
                            format!("mov qword ptr [rbp + 0x{:x}], rcx", test_dst_offset),
                        ],
                        _ => unreachable!(),
                    };

                    let expected_str_refs: Vec<&str> =
                        expected_instructions.iter().map(|s| s.as_str()).collect();
                    verify_instruction_sequence(&instructions, &expected_str_refs);
                }
            }

            #[test]
            fn test_emit_rbp_to_reg() {
                for size in [1, 2, 4, 8].iter() {
                    let mut asm = Assembler::new().unwrap();
                    let test_rbp_offset = 64;
                    let test_dst_reg = 15; // r15

                    emit_rbp_to_reg(
                        &mut asm,
                        RbpToRegParams {
                            rbp_offset: test_rbp_offset,
                            dst_reg: test_dst_reg,
                            size: *size,
                        },
                    );

                    let buffer = asm.finalize().unwrap();
                    let instructions = disassemble(&buffer);

                    assert_eq!(
                        instructions.len(),
                        1,
                        "Should have exactly 1 instruction for size {}",
                        size
                    );

                    let expected = match size {
                        1 => format!("mov r15b, byte ptr [rbp - 0x{:x}]", test_rbp_offset),
                        2 => format!("mov r15w, word ptr [rbp - 0x{:x}]", test_rbp_offset),
                        4 => format!("mov r15d, dword ptr [rbp - 0x{:x}]", test_rbp_offset),
                        8 => format!("mov r15, qword ptr [rbp - 0x{:x}]", test_rbp_offset),
                        _ => unreachable!(),
                    };

                    verify_instruction_at_position(&instructions, 0, &expected);
                }
            }

            #[test]
            fn test_emit_reg_to_rbp() {
                for size in [1, 2, 4, 8].iter() {
                    let mut asm = Assembler::new().unwrap();
                    let test_rbp_offset = 64;
                    let test_src_reg = 15; // r15

                    emit_reg_to_rbp(
                        &mut asm,
                        RegToRbpParams {
                            src_reg: test_src_reg,
                            rbp_offset: test_rbp_offset,
                            size: *size,
                        },
                    );

                    let buffer = asm.finalize().unwrap();
                    let instructions = disassemble(&buffer);

                    assert_eq!(
                        instructions.len(),
                        1,
                        "Should have exactly 1 instruction for size {}",
                        size
                    );

                    let expected = match size {
                        1 => format!("mov byte ptr [rbp + 0x{:x}], r15b", test_rbp_offset),
                        2 => format!("mov word ptr [rbp + 0x{:x}], r15w", test_rbp_offset),
                        4 => format!("mov dword ptr [rbp + 0x{:x}], r15d", test_rbp_offset),
                        8 => format!("mov qword ptr [rbp + 0x{:x}], r15", test_rbp_offset),
                        _ => unreachable!(),
                    };

                    verify_instruction_at_position(&instructions, 0, &expected);
                }
            }

            #[test]
            fn test_helper_functions_with_different_registers() {
                // Test with different registers (not just r15)
                let test_registers = [0, 1, 2, 3, 7, 8, 12]; // rax, rcx, rdx, rbx, rdi, r8, r12

                for reg in test_registers.iter() {
                    let mut asm = Assembler::new().unwrap();

                    // Test rbp_to_reg with this register
                    emit_rbp_to_reg(
                        &mut asm,
                        RbpToRegParams {
                            rbp_offset: 32,
                            dst_reg: *reg,
                            size: 8,
                        },
                    );

                    let buffer = asm.finalize().unwrap();
                    let instructions = disassemble(&buffer);

                    let reg_name = match reg {
                        0 => "rax",
                        1 => "rcx",
                        2 => "rdx",
                        3 => "rbx",
                        7 => "rdi",
                        8 => "r8",
                        12 => "r12",
                        _ => panic!("Test register not handled"),
                    };

                    let expected = format!("mov {}, qword ptr [rbp - 0x20]", reg_name);
                    verify_instruction_at_position(&instructions, 0, &expected);
                }
            }

            #[test]
            fn test_mem_to_reg_mem_edge_cases() {
                // Test with zero offset
                let mut asm = Assembler::new().unwrap();
                emit_mem_to_reg(
                    &mut asm,
                    MemToRegParams {
                        src_ptr: 0x1234,
                        src_offset: 0, // Zero offset
                        dst_reg: 15,
                        size: 8,
                    },
                );

                let buffer = asm.finalize().unwrap();
                let instructions = disassemble(&buffer);

                verify_instruction_at_position(
                    &instructions,
                    1,
                    "mov r15, qword ptr [rax]", // Should have no explicit offset
                );

                // Test with negative offset
                let mut asm = Assembler::new().unwrap();
                emit_mem_to_mem(
                    &mut asm,
                    MemToMemParams {
                        src_ptr: 0x1234,
                        src_offset: -8, // Negative offset
                        dst_offset: 16,
                        size: 4,
                    },
                );

                let buffer = asm.finalize().unwrap();
                let instructions = disassemble(&buffer);

                verify_instruction_at_position(
                    &instructions,
                    1,
                    "mov ecx, dword ptr [rax - 8]", // Should have negative offset
                );
            }
        }

        /// Tests for invalid input handling in helper functions
        mod error_cases {
            use super::*;

            #[test]
            #[should_panic(expected = "Unsupported value size")]
            fn test_emit_mem_to_reg_invalid_size() {
                let mut asm = Assembler::new().unwrap();
                emit_mem_to_reg(
                    &mut asm,
                    MemToRegParams {
                        src_ptr: 0x1234,
                        src_offset: 0,
                        dst_reg: 15,
                        size: 3, // Invalid size
                    },
                );
            }

            #[test]
            #[should_panic(expected = "Unsupported value size")]
            fn test_emit_mem_to_mem_invalid_size() {
                let mut asm = Assembler::new().unwrap();
                emit_mem_to_mem(
                    &mut asm,
                    MemToMemParams {
                        src_ptr: 0x1234,
                        src_offset: 0,
                        dst_offset: 0,
                        size: 16, // Invalid size
                    },
                );
            }

            #[test]
            #[should_panic(expected = "Unsupported value size")]
            fn test_emit_rbp_to_reg_invalid_size() {
                let mut asm = Assembler::new().unwrap();
                emit_rbp_to_reg(
                    &mut asm,
                    RbpToRegParams {
                        rbp_offset: 64,
                        dst_reg: 15,
                        size: 3, // Invalid size
                    },
                );
            }

            #[test]
            #[should_panic(expected = "Unsupported value size")]
            fn test_emit_reg_to_rbp_invalid_size() {
                let mut asm = Assembler::new().unwrap();
                emit_reg_to_rbp(
                    &mut asm,
                    RegToRbpParams {
                        src_reg: 15,
                        rbp_offset: 64,
                        size: 16, // Invalid size
                    },
                );
            }
        }
    }

    /// Tests for buffer-related operations
    mod buffer_operations {
        use super::*;
        #[test]
        fn test_copy_live_vars_to_temp_buffer() {
            let src_rec = Record {
                offset: 0,
                size: 0,
                id: 0,
                live_vals: vec![
                    vec![Location::Indirect(6, 56, 8)].into(),
                    vec![Location::Indirect(6, 72, 8)].into(),
                    vec![Location::Indirect(6, 172, 8)].into(),
                ],
            };

            let mut asm = Assembler::new().unwrap();
            let lvb =
                copy_live_vars_to_temp_buffer(&mut asm, &src_rec, ControlPointStackMapId::UnOpt);
            assert_eq!(24, lvb.size);
            assert_eq!(3, lvb.variables.len());

            // Finalise and disassemble the code.
            let buffer = asm.finalize().unwrap();
            let instructions = disassemble(&buffer);

            let expected_instructions = [
                &format!("mov rax, 0x{:x}", lvb.ptr as i64),
                "mov rcx, qword ptr [rbp + 0x38]",
                "mov qword ptr [rax], rcx",
                "mov rcx, qword ptr [rbp + 0x48]",
                "mov qword ptr [rax + 8], rcx",
                "mov rcx, qword ptr [rbp + 0xac]",
                "mov qword ptr [rax + 0x10], rcx",
            ];
            verify_instruction_sequence(&instructions, &expected_instructions);
        }
    }

    /// Tests for live variable operations
    mod live_variable_operations {
        use super::*;

        /// Tests for register-to-register operations
        mod register_to_register {
            use super::*;

            #[test]
            fn test_basic_register_to_register() {
                let src_rec = Record {
                    offset: 0,
                    size: 0,
                    id: 0,
                    live_vals: vec![vec![Location::Register(15, 8, vec![].into())].into()],
                };
                let dst_rec = Record {
                    offset: 0,
                    size: 0,
                    id: 0,
                    live_vals: vec![vec![Location::Register(1, 8, vec![].into())].into()],
                };

                let mut asm = Assembler::new().unwrap();
                let temp_live_vars_buffer = LiveVarsBuffer {
                    ptr: 0 as *mut u8,
                    layout: Layout::new::<u8>(),
                    variables: HashMap::new(),
                    size: 0,
                };
                let dest_reg_nums = set_destination_live_vars(
                    &mut asm,
                    &src_rec,
                    &dst_rec,
                    0x10,
                    temp_live_vars_buffer,
                );
                let buffer = asm.finalize().unwrap();
                let instructions = disassemble(&buffer);

                verify_instruction_at_position(&instructions, 0, "mov rdx, qword ptr [rbp - 0x10]");
                assert_eq!(
                    dest_reg_nums.get(&1),
                    Some(&8),
                    "The destination register (rcx) should be recorded with its size"
                );
            }

            #[test]
            fn test_register_to_register_with_additional_locations() {
                let src_rec = Record {
                    offset: 0,
                    size: 0,
                    id: 0,
                    live_vals: vec![
                        vec![Location::Register(14, 8, vec![].into())].into(),
                        vec![Location::Register(13, 8, vec![-80, -200].into())].into(),
                        vec![Location::Register(15, 8, vec![-72].into())].into(),
                        vec![Location::Register(12, 8, vec![-56].into())].into(),
                        vec![Location::Register(0, 8, vec![8, -16, -88].into())].into(),
                        vec![Location::Register(3, 8, vec![-64].into())].into(),
                    ],
                };

                let dst_rec = Record {
                    offset: 0,
                    size: 0,
                    id: 0,
                    live_vals: vec![
                        vec![Location::Register(13, 8, vec![].into())].into(),
                        vec![Location::Register(14, 8, vec![-80].into())].into(),
                        vec![Location::Register(12, 8, vec![-64].into())].into(),
                        vec![Location::Register(15, 8, vec![-72].into())].into(),
                        vec![Location::Register(0, 8, vec![-16].into())].into(),
                        vec![Location::Register(3, 8, vec![-88, -8].into())].into(),
                    ],
                };

                let mut asm = Assembler::new().unwrap();
                let temp_live_vars_buffer = LiveVarsBuffer {
                    ptr: std::ptr::null_mut(),
                    layout: Layout::new::<u8>(),
                    variables: HashMap::new(),
                    size: 0,
                };

                let rbp_offset_reg_store: i32 = 200;
                set_destination_live_vars(
                    &mut asm,
                    &src_rec,
                    &dst_rec,
                    rbp_offset_reg_store as i64,
                    temp_live_vars_buffer,
                );
                let buffer = asm.finalize().unwrap();
                let instructions = disassemble(&buffer);
                assert_eq!(instructions.len(), 18);

                // Verify key instructions using the helper
                verify_instruction_at_position(
                    &instructions,
                    0,
                    &format!(
                        "mov r13, qword ptr [rbp - 0x{:x}]",
                        rbp_offset_reg_store - REG_OFFSETS.get(&14).unwrap()
                    ),
                );
            }
        }

        /// Tests for indirect-to-register operations
        mod indirect_to_register {
            use super::*;

            #[test]
            fn test_basic_indirect_to_register() {
                let src_rec = Record {
                    offset: 0,
                    size: 0,
                    id: 0,
                    live_vals: vec![vec![Location::Indirect(6, 0, 8)].into()],
                };
                let dst_rec = Record {
                    offset: 0,
                    size: 0,
                    id: 0,
                    live_vals: vec![vec![Location::Register(15, 8, vec![].into())].into()],
                };

                let mut asm = Assembler::new().unwrap();
                let (ptr, layout, _size) =
                    LiveVarsBuffer::get_or_create(&src_rec, ControlPointStackMapId::UnOpt);

                let mut variables = HashMap::new();
                variables.insert(0 as i32, REG64_BYTESIZE as i32);
                let temp_live_vars_buffer = LiveVarsBuffer {
                    ptr,
                    layout,
                    variables,
                    size: 8 as i32,
                };

                let dest_reg_nums = set_destination_live_vars(
                    &mut asm,
                    &src_rec,
                    &dst_rec,
                    0x10,
                    temp_live_vars_buffer,
                );
                let buffer = asm.finalize().unwrap();
                let instructions = disassemble(&buffer);

                let expected_instructions = [
                    &format!("mov rax, 0x{:x}", ptr as i64),
                    "mov r15, qword ptr [rax]",
                ];

                verify_instruction_sequence(&instructions, &expected_instructions);
                assert_eq!(
                    dest_reg_nums.get(&15),
                    Some(&8),
                    "The destination register (r15) should be recorded with its size"
                );
            }
        }
    }
}
