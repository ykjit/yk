# Understanding Traces

yk can print the traces it has created to `stderr` to help with debugging.
However, these traces are often lengthy, and not always easy to understand.
This section briefly explains how to get yk to print its traces, and how
to make them a bit easier to understand.


## Producing a trace

### `YKD_LOG_IR`

`YKD_LOG_IR=[<path>:]<irstage_1>[,...,<irstage_n>]` logs IR from different stages
to `path`. The special value `-` (i.e. a single dash) can be used for `<path>`
to indicate stderr.

The following `ir_stage`s are supported:

 - `aot`: the entire AOT IR for the interpreter.
 - `debugstrs`: if compiled into the interpreter, "debug strings" specific to
    that interpreter (e.g. showing the opcodes executed).
 - `hir`: high-level JIT IR.
 - `jit-asm`: the assembler code of the compiled JIT IR trace.
