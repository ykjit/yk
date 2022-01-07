# Environment Variables

There are a number of environment variables which control the behaviour of the
yk system.

Variables prefixed with `YKD_` are intended for debugging.

## Run-time Variables

### `YKD_PRINT_IR`

`YKD_PRINT_IR` accepts a comma-separated list of JIT pipeline stages at which
to print LLVM IR (to stderr).

The following stages are supported:

 - `aot`: the IR embedded in the ahead-of-time compiled binary.
 - `jit-pre-opt`: the IR for the trace before it is optimised by LLVM.
 - `jit-post-opt`: the IR for the trace after LLVM has optimised it. This is
   the IR that will be submitted to the LLVM code generator.

### `YKD_SERIALISE_COMPILATION`

When `YKD_SERIALISE_COMPILATION=1`, calls to `yk_control_point(loc)` will block
while `loc` is being compiled.

This variable is only available when the `c_testing` feature is used, and it is
only intended for use in testing.

## Compile-time Variables

### `YKD_PRINT_JITSTATE`

When defined, `YKD_PRINT_JITSTATE` causes `ykllvm` to emit extra prints (to
stderr) into the generated control point IR. The prints indicate state changes
in the JIT and will be visible at runtime as the interpreter is running.

 * `jit-state: start-tracing` is printed when the system starts tracing.
 * `jit-state: stop-tracing` is printed when the system stops tracing.
 * `jit-state: enter-jit-code` is printed when the system starts executing
   JITted code.
 * `jit-state: exit-jit-code` is printed when the system stops executing
   JITted code.

Note that there are no `start-interpreting` and `stop-interpreting`
notifications: if the system is not currently tracing or executing JITted code,
then it is implicitly interpreting.
