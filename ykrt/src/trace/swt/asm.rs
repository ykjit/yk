use capstone::prelude::*;
use dynasmrt::ExecutableBuffer;

/// Disassemble an executable buffer into a vector of instruction strings
pub(crate) fn disassemble(buffer: &ExecutableBuffer) -> Vec<String> {
    if buffer.len() == 0 {
        return vec![];
    }

    let code_ptr = buffer.ptr(dynasmrt::AssemblyOffset(0)) as *const u8;
    let code_size = buffer.len();

    // Use Capstone to disassemble and check the generated instructions
    let capstone = Capstone::new()
        .x86()
        .mode(arch::x86::ArchMode::Mode64)
        .build()
        .expect("Failed to create Capstone disassembler");

    let instructions = capstone
        .disasm_all(
            unsafe { std::slice::from_raw_parts(code_ptr, code_size) },
            code_ptr as u64,
        )
        .expect("Failed to disassemble code");

    instructions
        .iter()
        .map(|inst| {
            format!(
                "{} {}",
                inst.mnemonic().unwrap_or(""),
                inst.op_str().unwrap_or("")
            )
        })
        .collect()
}

/// Verify that the actual instruction sequence matches the expected sequence
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

/// Assert macros for common testing patterns
#[cfg(test)]
#[cfg(swt_modclone)]
#[macro_export]
macro_rules! assert_instruction_eq {
    ($actual:expr, $expected:expr) => {
        assert_eq!(
            $actual, $expected,
            "Instruction mismatch.\nActual:   '{}'\nExpected: '{}'",
            $actual, $expected
        );
    };
}

#[cfg(test)]
#[cfg(swt_modclone)]
#[macro_export]
macro_rules! assert_instruction_sequence {
    ($actual:expr, $expected:expr) => {
        $crate::trace::swt::asm::verify_instruction_sequence($actual, $expected);
    };
}

#[cfg(test)]
#[cfg(swt_modclone)]
#[macro_export]
macro_rules! assert_register_offset_instruction {
    ($actual:expr, $base:expr, $reg:expr, $offset:expr) => {
        let expected = format!("{} {}, qword ptr [rbp - 0x{:x}]", $base, $reg, $offset);
        assert_instruction_eq!($actual, &expected);
    };
}
