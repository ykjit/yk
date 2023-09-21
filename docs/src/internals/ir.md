# Yk Intermediate IR

**This is a WIP document. We are currently in the process of switching from
using LLVM IR to our own IR. This document describes the new IR**

## On-disk serialisation format.

```
Module {
    // header
    magic: u32      // hard-coded `0xedd5f00d`
    version: u32    // format version (currently 0)
    // functions
    num_funcs: usize
    funcs: Function[num_funcs]
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
    opcode: u8
}
```
