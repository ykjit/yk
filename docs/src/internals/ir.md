# Yk Intermediate IR

**This is a WIP document. We are currently in the process of switching from
using LLVM IR to our own IR. This document describes the new IR**

## On-disk serialisation format.

### Module encoding.

```
Module {
    // header
    magic: u32 = 0xedd5f00d
    version: u32    // format version (currently 0)
    // functions
    num_funcs: usize
    funcs: Function[num_funcs]
    // types
    num_types: usize
    types: Type[num_types]
    // constants
    num_constants: usize
    constants: Constant[num_constants]
}

Function {
    name: null_term_str
    num_blocks: usize
    blocks: Block[num_blocks]
}

Block {
    num_instrs: usize
    instrs: Instruction[num_instrs]
}

Instruction {
    type_index: usize // type of value generated
    opcode: u8
    num_operands: u32
    operands: Operand[num_operands]
}

Constant {
    type_index: usize,
    num_bytes: usize,
    bytes: u8[num_bytes]
}
```


### Type encoding.

`Type` is encoded as one of the following:

IntegerType {
    type_kind: u8 = 0
    num_bits: u32
}

### Operand encoding.

`Operand` is encoded as one of the following:

```
ConstantOperand {
    operand_kind: u8 = 0
    constant_index: usize
}

LocalVariableOperand {
    operand_kind: u8 = 1
    bb_idx: usize       // This field and the following one identify which
    inst_idx: usize     // instruction defines the referenced local variable.
}

StringOperand { // Used as a catch-all for unimplemented operands.
    operand_kind: u8 = 2
    str: null_term_str
}
```
