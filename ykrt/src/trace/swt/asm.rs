//! # Assembly Disassembly and Debugging Utilities
//!
//! This module provides assembly disassembly and formatting utilities specifically tailored
//! for debugging and testing the dynamically generated assembly code used in SWT control
//! point transition.
//!
//! ## Purpose
//!
//! The Software Trace (SWT) system generates complex assembly sequences at runtime using
//! `dynasm`. This module provides essential tools for:
//!
//! 1. **Runtime Debugging**: Inspecting generated assembly when `YKD_SWT_VERBOSE_ASM` is
//!    enabled.
//! 2. **Test Verification**: Ensuring generated code matches expected instruction sequences.
//! 3. **Development Support**: Understanding what assembly is produced for different
//!    transition patterns.
//! 4. **Performance Analysis**: Examining instruction efficiency and optimisation opportunities.
//!
//! ### Disassembly Functions
//! - [`print_disassembly`]: Runtime output for debugging (with addresses)
//! - [`disassemble`]: Test-oriented output (instruction strings only)
//!
//! ## Testing Support
//!
//! The module provides specialised testing utilities:
//! - [`verify_instruction_sequence`]: Compares entire instruction sequences with detailed
//!   error reporting
//! - [`verify_instruction_at_position`]: Checks specific instructions at given positions

use dynasmrt::ExecutableBuffer;
use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter};

/// Configures an IntelFormatter to produce output that matches GDB's Intel syntax.
///
/// This function sets up the formatter with specific options to ensure consistency
/// with GDB's disassembly output, making it easier for developers to correlate
/// generated assembly with familiar debugging tools.
///
/// # Configuration Applied
///
/// - Operand separators have spaces after them (e.g., `mov rax, rbx` not `mov rax,rbx`)
/// - Hexadecimal values use lowercase with `0x` prefix (e.g., `0x1234` not `1234h`)
/// - Memory operations always show size indicators (`qword ptr [rax]`)
/// - Memory arithmetic operations have proper spacing (`[rax + 8]` not `[rax+8]`)
pub(crate) fn configure_gdb_intel_formatter(formatter: &mut IntelFormatter) {
    formatter
        .options_mut()
        .set_space_after_operand_separator(true);
    formatter.options_mut().set_hex_prefix("0x");
    formatter.options_mut().set_hex_suffix("");
    formatter.options_mut().set_uppercase_hex(false);
    formatter.options_mut().set_uppercase_keywords(false);
    formatter
        .options_mut()
        .set_space_between_memory_add_operators(true);
    formatter
        .options_mut()
        .set_space_between_memory_mul_operators(true);
    formatter
        .options_mut()
        .set_memory_size_options(iced_x86::MemorySizeOptions::Always);
    formatter.options_mut().set_rip_relative_addresses(true);
}

/// Prints a human-readable disassembly of an executable buffer with memory addresses.
///
/// This function is used for runtime debugging when `YKD_SWT_VERBOSE_ASM` is enabled.
/// It outputs each instruction with its memory address, making it easy to correlate
/// with debugger output or crash dumps.
///
/// The output format matches GDB's Intel syntax for consistency with debugging tools.
/// Each instruction is prefixed with its memory address in hexadecimal format.
///
/// # Arguments
///
/// * `buffer` - The executable buffer containing machine code to disassemble
///
/// # Output Format
///
/// ```text
///   7f1565029000: mov rbp, 0x7ffe9c46df40
///   7f1565029007: mov rsp, 0x7ffe9c46df40
///   7f156502900e: sub rsp, 0x40
/// ```
///
/// # Safety
///
/// This function uses unsafe code internally to access the raw machine code buffer,
/// but the ExecutableBuffer type ensures the memory is valid and executable.
pub(crate) fn print_disassembly(buffer: &ExecutableBuffer) {
    let code_ptr = buffer.ptr(dynasmrt::AssemblyOffset(0)) as *const u8;
    let code_slice = unsafe { std::slice::from_raw_parts(code_ptr, buffer.len()) };

    let mut decoder = Decoder::with_ip(64, code_slice, code_ptr as u64, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    configure_gdb_intel_formatter(&mut formatter);
    let mut instruction = Instruction::default();
    let mut output = String::new();
    while decoder.can_decode() {
        decoder.decode_out(&mut instruction);
        output.clear();
        formatter.format(&instruction, &mut output);
        eprintln!("  {:x}: {}", instruction.ip(), output);
    }
}

/// Disassembles an executable buffer into a vector of instruction strings for testing.
///
/// This function is specifically designed for test verification, returning instruction
/// strings without memory addresses. This makes it easier to write deterministic tests
/// that verify the correctness of generated assembly sequences.
///
/// Unlike [`print_disassembly`], this function returns structured data rather than
/// printing to stdout, making it suitable for automated testing.
///
/// # Arguments
///
/// * `buffer` - The executable buffer containing machine code to disassemble
///
/// # Returns
///
/// A vector of instruction strings in GDB Intel syntax format. Returns an empty
/// vector if the buffer is empty.
///
/// # Example
///
/// ```rust,ignore
/// let instructions = disassemble(&buffer);
/// assert_eq!(instructions[0], "mov rax, 0x1234");
/// assert_eq!(instructions[1], "mov rbx, qword ptr [rax]");
/// ```
///
/// # Safety
///
/// This function uses unsafe code internally to access the raw machine code buffer,
/// but the ExecutableBuffer type ensures the memory is valid and executable.
#[cfg(test)]
pub(crate) fn disassemble(buffer: &ExecutableBuffer) -> Vec<String> {
    if buffer.len() == 0 {
        return vec![];
    }

    let code_ptr = buffer.ptr(dynasmrt::AssemblyOffset(0)) as *const u8;
    let code_size = buffer.len();

    // Use iced-x86 to disassemble and check the generated instructions
    let code_slice = unsafe { std::slice::from_raw_parts(code_ptr, code_size) };
    let mut decoder = Decoder::with_ip(64, code_slice, code_ptr as u64, DecoderOptions::NONE);

    let mut formatter = IntelFormatter::new();
    configure_gdb_intel_formatter(&mut formatter);

    let mut instructions = Vec::new();
    let mut instruction = Instruction::default();
    let mut output = String::new();

    while decoder.can_decode() {
        decoder.decode_out(&mut instruction);
        output.clear();
        formatter.format(&instruction, &mut output);
        instructions.push(output.clone());
    }

    instructions
}

/// Verifies that an actual instruction sequence matches the expected sequence exactly.
///
/// This function provides comprehensive error reporting when instruction sequences don't
/// match, including the position of the first mismatch and the full context of both
/// sequences. This makes it much easier to debug test failures in generated assembly.
///
/// The verification checks both the instruction count and each individual instruction
/// string for exact matches.
///
/// # Arguments
///
/// * `actual` - The actual instruction sequence from disassembly
/// * `expected` - The expected instruction sequence as string literals
///
/// # Panics
///
/// This function will panic with detailed error information if:
/// - The number of instructions doesn't match
/// - Any instruction doesn't match exactly (case-sensitive)
///
/// # Example
///
/// ```rust,ignore
/// let instructions = disassemble(&buffer);
/// let expected = [
///     "mov rax, 0x1234",
///     "mov rbx, qword ptr [rax]",
/// ];
/// verify_instruction_sequence(&instructions, &expected);
/// ```
#[cfg(test)]
pub(crate) fn verify_instruction_sequence(actual: &[String], expected: &[&str]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Instruction count mismatch. Expected {} instructions, got {}.\nActual: {:?}\nExpected: {:?}",
        expected.len(),
        actual.len(),
        actual,
        expected
    );

    for (i, (actual_inst, expected_inst)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_inst, expected_inst,
            "Instruction {} mismatch.\nActual:   '{}'\nExpected: '{}'\nFull sequence:\nActual: {:?}\nExpected: {:?}",
            i, actual_inst, expected_inst, actual, expected
        );
    }
}

/// Verifies that a specific instruction appears at the given position in the sequence.
///
/// This function is useful when you only need to verify one or a few specific
/// instructions in a larger sequence, rather than the entire sequence. It provides
/// detailed error reporting including the full instruction sequence for context.
///
/// # Arguments
///
/// * `instructions` - The instruction sequence to check
/// * `position` - The zero-based position of the instruction to verify
/// * `expected` - The expected instruction string at that position
///
/// # Panics
///
/// This function will panic with detailed error information if:
/// - The position is out of bounds for the instruction sequence
/// - The instruction at the position doesn't match exactly (case-sensitive)
///
/// # Example
///
/// ```rust,ignore
/// let instructions = disassemble(&buffer);
/// // Verify that the third instruction is a specific move operation
/// verify_instruction_at_position(&instructions, 2, "mov rcx, qword ptr [rbp - 0x10]");
/// ```
#[cfg(test)]
pub(crate) fn verify_instruction_at_position(
    instructions: &[String],
    position: usize,
    expected: &str,
) {
    assert!(
        position < instructions.len(),
        "Position {} is out of bounds for instruction sequence of length {}",
        position,
        instructions.len()
    );

    assert_eq!(
        instructions[position], expected,
        "Instruction at position {} mismatch.\nActual:   '{}'\nExpected: '{}'\nFull sequence: {:?}",
        position, instructions[position], expected, instructions
    );
}
