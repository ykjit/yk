//! Helpers for generating assembler code with dynasm.

/// Emits a 'mem <- reg'  assembler instruction using the desired size qualifier.
macro_rules! asm_mem_reg {
    ($dasm: expr, $size: expr, $op: expr, $mem: expr, $reg: expr) => {
        match $size {
            1 => {
                dynasm!($dasm
                    ; $op BYTE $mem, Rb($reg)
                );
            }
            2 => {
                dynasm!($dasm
                    ; $op WORD $mem, Rw($reg)
                );
            },
            4 => {
                dynasm!($dasm
                    ; $op DWORD $mem, Rd($reg)
                );
            },
            8 => {
                dynasm!($dasm
                    ; $op QWORD $mem, Rq($reg)
                );
            }
            _ => panic!("Invalid size operand: {}", $size),
        }
    }
}
