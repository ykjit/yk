# Understanding Traces

yk can print the traces it has created to `stderr` to help with debugging.
However, these traces are often lengthy, and not always easy to understand.
This section briefly explains how to get yk to print its traces, and how
to make them a bit easier to understand.


## Producing a trace

### `YKD_LOG_IR`

`YKD_LOG_IR` accepts a comma-separated list of JIT pipeline stages at which
to print IR to stderr.

The following stages are supported:

 - `aot`: the entire AOT IR for the interpreter.
 - `jit-pre-opt`: the JIT IR trace before optimisation.
 - `jit-post-opt`: the JIT IR trace after optimisation.


#### `YKD_TRACE_DEBUGINFO`

When `YKD_TRACE_DEBUGINFO=1`, the JIT will add debugging information to JITted
traces, allowing debuggers conforming to the [gdb JIT
interface](https://sourceware.org/gdb/current/onlinedocs/gdb/JIT-Interface.html)
to show higher-level representations of the code in the source view.

This feature relies on the use of temporary files, which (in addition to being
slow to create) are not guaranteed to be cleaned up.


### `YKD_LOG_JITSTATE`

If the `YKD_LOG_JITSTATE=<path>` environment variable is defined, then changes
in the "JIT state" will be appended, as they occur, to the file at `<path>` as
they occur. The special value `-` (i.e. a single dash) can be used for `<path>`
to indicate stderr.

The JIT states written are:

 * `jitstate: start-tracing` is printed when the system starts tracing.
 * `jitstate: stop-tracing` is printed when the system stops tracing.
 * `jitstate: enter-jit-code` is printed when the system starts executing
   JITted code.
 * `jitstate: exit-jit-code` is printed when the system stops executing
   JITted code.

Note that there are no `start-interpreting` and `stop-interpreting`
notifications: if the system is not currently tracing or executing JITted code,
then it is implicitly interpreting.

This variable is only available when building `ykrt` with the `ykd` Cargo
feature enabled.
