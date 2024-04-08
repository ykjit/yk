# Run-time configuration

There are a number of environment variables which control the run_time
behaviour of the yk system.

Variables prefixed with `YKD_` are intended for debugging only. Most (if not
all) of the debugging variables introduce extra computation that slows down
program execution, sometimes significantly.


## Run-time Variables

### `YKD_NEW_CODEGEN`

When set to `1` forces the JIT to use the new codegen.

This is temporary, and will be removed once the new codegen is production
quality.

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


### `YKD_LOG_JITSTATE`

When defined, `YKD_LOG_JITSTATE` causes the system to emit extra information
(to stderr) about JIT transition events:

 * `jit-state: start-tracing` is printed when the system starts tracing.
 * `jit-state: stop-tracing` is printed when the system stops tracing.
 * `jit-state: enter-jit-code` is printed when the system starts executing
   JITted code.
 * `jit-state: exit-jit-code` is printed when the system stops executing
   JITted code.

Note that there are no `start-interpreting` and `stop-interpreting`
notifications: if the system is not currently tracing or executing JITted code,
then it is implicitly interpreting.

This variable is only available when building `ykrt` with the
`ykd` Cargo feature enabled.


### `YKD_SERIALISE_COMPILATION`

When `YKD_SERIALISE_COMPILATION=1`, calls to `yk_control_point(loc)` will block
while `loc` is being compiled.

This variable is only available when building `ykrt` with the `yk_testing`
Cargo feature enabled.


### `YKD_TRACE_DEBUGINFO`

When `YKD_TRACE_DEBUGINFO=1`, the JIT will add debugging information to JITted
traces, allowing debuggers conforming to the [gdb JIT
interface](https://sourceware.org/gdb/current/onlinedocs/gdb/JIT-Interface.html)
to show higher-level representations of the code in the source view.

This feature relies on the use of temporary files, which (in addition to being
slow to create) are not guaranteed to be cleaned up.
