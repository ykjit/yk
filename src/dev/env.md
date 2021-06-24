# Environment Variables

There are a number of environment variables which control the behaviour of the
yk system.

## `YK_PRINT_IR`

When `YK_PRINT_IR=1`, yk will print (to `stderr`) a textual version of each
trace's LLVM IR prior to code generation.

## `YK_PRINT_IR_SBS`

When `YK_PRINT_IR_SBS=1`, yk will print (to `stderr`) a "side-by-side" listing
of each trace's LLVM IR prior to code generation. The listing shows the
instructions of the trace alongside the instructions from which they were
derived (from the AOT IR stored in the interpreter binary). Inlining boundaries
are also annotated on the listing.
