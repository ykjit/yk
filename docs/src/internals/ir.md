# Yk Intermediate IR

**This is a WIP document. We are currently in the process of switching from
using LLVM IR to our own IR. This document describes the new IR**

## On-disk serialisation format.

(syntax loosely based on
https://www.llvm.org/docs/StackMaps.html#stack-map-format)

```
header {
    magic: usize    // hard-coded `0xedd5f00d`
    version: usize  // format version (currently 0)
}
toc {
    funcs {
        num_funcs: usize
        func_offs[num_funcs] {      // indices of this array server as function IDs.
            func_data_off: usize    // byte offset into `func_data`.
        }
        func_data_size: usize       // in bytes.
                                    // allows deserialiser to quickly skip over
                                    // `func_data`.
    }
    globals {
        num_globals: usize
        global_offs[num_globals] {  // indices of this array serve as global IDs.
            global_data_off: usize  // byte offset into `global_data`.
        }
        global_data_size: usize     // in bytes.
                                    // allows deserialiser to quickly skip over
                                    // `global_data`.
    }
    types {
        num_types: usize
        type_offs[num_types] {      // indices of this array serve as type IDs.
            type_data_off: usize    // byte offset into `type_data`
        }
        type_data_size: usize       // in bytes.
                                    // allows deserialiser to quickly skip over
                                    // `type_data`.
    }
    constants {
        num_constants: usize
        type_offs[num_types] {
            constant_data_off: usize    // byte offset into `constant_data`
        }
    }
}
func_data[num_funcs] {
    name: null_term_str
    func_sig {
        num_args: usize
        arg_tys[num_args] {
            ty_index: usize
        }
    }
    func_index {
        num_blocks: usize           // value of 0 means this is a declaration.
        block_offs[num_blocks] {
            block_data_off: usize   // byte offset into `block_data`.
        }
    }
    block_data[num_blocks] {
        num_instrs: u32
        instrs[num_instrs] {
            opcode: byte
            // The payload is opcode dependent, but "regular" operands are
            // encoded using the `operand` format outline below.
            payload: byte[]
        }
    }
}
global_data[num_globals] {
    name: null_term_str
    type_index: usize
    // TODO: initialiser encoding
}
type_data[num_types] {
    name: null_term_str
    size: usize             // in bytes.
    // TODO: encode information about all the various types that can arise.
    // integers, floats, structs, arrays, pointers, ...
}
constant_data[num_constants] {
    name: null_term_str
    type_index: usize
    data: byte[]
}
```

### Opcodes

TODO: very similar to LLVM IR. Fill in as we implement them.

### Instruction operands

The exact operands encoding is specified per-instruction, but those that are
"regular" (reference a constant, local variable, or a global variable etc.)
could be encoded like:

```
constant_operand {
    magic: byte = 0
    const_idx: usize
}

local_operand {
    magic: byte = 1
    block_idx: usize        // index of basic block (from the same function as
                            // the operand's use).
    instr_idx: usize        // index of the instruction that generates the
                            // value (in the block identified by `block_idx`).
}

local_operand {
    magic: byte = 2
    global_idx: usize
}
...
```

### Lazy loading

The format is designed such that it can be loaded lazily to avoid pre-loading a
large payload (slow) to only use a small portion of it (the code that was
actually traced).

Initially, a deserialiser need only read the `header` and the `toc`. From this
it can compute the base addresses of the `*_data` fields, and information held
deeper can then be lazily deserialised on-demand.

## Notes

 - `null_term_str` has been selected so that it can be zero-copied into a
   `&'static CStr` in Rust.

 - Although `usize`s could be compressed with uleb128 (or similar), we trade
   space for speed and store them uncompressed.

 - A deserialiser would be expected to cache things that are lazily loaded to
   prevent repeated deserialisation.

  - Branches will not contain edge information for now, as a trace guides us.

  - For now, a call does not terminate a block (inherited from LLVM)

## Questions

  - do we need padding to force aligned reads?

  - Did I go overboard with expressing types so richly? Would a simple bag of
    bytes (i.e. just store how long something is) work better?

