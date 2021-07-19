# Environment Variables

There are a number of environment variables which control the behaviour of the
yk system.

Variables prefixed with `YKD_` are intended for debugging.

## `YKD_PRINT_IR`

`YKD_PRINT_IR` accepts a comma-separated list of JIT pipeline stages at which
to print LLVM IR (to stderr).

The following stages are supported:

 - `aot`: the IR embedded in the ahead-of-time compiled binary.
 - `jit-pre-opt`: the IR for the trace before it is optimised by LLVM.
 - `jit-pre-opt-sbs`: the IR for the trace before it is optimised alongside the
   AOT IR from which it was derived. Inlining boundaries are also annotated.
 - `jit-post-opt`: the IR for the trace after LLVM has optimised it. This is
   the IR that will be submitted to the LLVM code generator.
