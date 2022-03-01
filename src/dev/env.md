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

This variable is always available, and does not require any Cargo feature to be
enabled.

### `YKD_PRINT_JITSTATE`

When defined, `YKD_PRINT_JITSTATE` causes the system to emit extra information
(to stderr) about JIT transition events:

 * `jit-state: start-tracing` is printed when the system starts tracing.
 * `jit-state: stop-tracing` is printed when the system stops tracing.
 * `jit-state: enter-jit-code` is printed when the system starts executing
   JITted code.
 * `jit-state: exit-jit-code` is printed when the system stops executing
   JITted code.

FIXME: Add stop-gapping states once finalised.

Note that there are no `start-interpreting` and `stop-interpreting`
notifications: if the system is not currently tracing or executing JITted code,
then it is implicitly interpreting.

This variable is only available when building with the `yk_jitstate_debug`
Cargo feature is enabled.

### `YKD_SERIALISE_COMPILATION`

When `YKD_SERIALISE_COMPILATION=1`, calls to `yk_control_point(loc)` will block
while `loc` is being compiled.

This variable is only available when the `c_testing` feature is used, and it is
only intended for use in testing.

This variable is only available when building with the `yk_testing` Cargo
feature is enabled.
