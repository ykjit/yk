use dynasmrt::ExecutableBuffer;
use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, IntelFormatter};

/// Configure IntelFormatter to match GDB's Intel syntax
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

/// Print disassembly of an executable buffer with addresses
pub(crate) fn print_disassembly(buffer: &ExecutableBuffer) {
    let code_ptr = buffer.ptr(dynasmrt::AssemblyOffset(0)) as *const u8;
    let code_slice = unsafe { std::slice::from_raw_parts(code_ptr, buffer.len()) };

    let mut decoder = Decoder::with_ip(64, code_slice, code_ptr as u64, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    configure_gdb_intel_formatter(&mut formatter);
    let mut instruction = Instruction::default();
    let mut output = String::new();

    println!("ASM DUMP:");
    while decoder.can_decode() {
        decoder.decode_out(&mut instruction);
        output.clear();
        formatter.format(&instruction, &mut output);
        println!("  {:x}: {}", instruction.ip(), output);
    }
}

/// Disassemble an executable buffer into a vector of instruction strings
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


/// Verify that the actual instruction sequence matches the expected sequence
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

/// Verify that a specific instruction appears at the given position
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
