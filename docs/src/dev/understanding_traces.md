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


## trace_chewer

`trace_chewer` is a small program included with yk which can help you
understand `YKD_LOG_IR`s output. The general format of `trace_chewer` is:

```
trace_chewer <command> <args>
```


### simplify

`trace_chewer simplify <file-name>` takes as input a trace and converts
it into a straight-line trace. It removes LLVM declarations and guard
failure blocks, neither of which are enlightening. It converts LLVM
branches into simpler `guard_true(%var)` or `guard_false(%var)` statements
(i.e. `guard_true(%3)` means "if the boolean value stored in `%3` is true,
then continue executing the trace, otherwise deoptimise back to the
interpreter).

If you specify `-` as the filename, `trace_chewer simplify` will read from
stdin. This means that you can simplify traces without saving them to disk:

```
YKD_LOG_IR=jit-post-opt lua f.lua 2>&1 | trace_chewer simplify -
```

You can specify a plugin with `trace_chewer simplify -p <plugin.py>`:
`simplify` makes use of the [`DebugInfoProcess`](#debuginfoprocess) API.


### plugins

Some `trace_chewer` commands take a `-p <plugin.py>` plugin argument.
`plugin.py` is a Python file that `trace_chewer` will load as a module called
`plugin`. Different commands will use different parts of the plugin API.

#### `DebugInfoProcess`

This class is used to process debug info lines in traces. The API is:

```python
# This class will be instantiated before trace_chewer
# processes any traces.
class DebugInfoProcess:
    # Called before starting each new trace in the
    # debug output.
    def next_trace(self): ...
    # Called for each debug line in the output:
    #   * `path` is the (possibly incomplete) source
    #      path (string)
    #   * `line` is the line number (int)
    #   * `col` is the column number (int)
    #   * `fn` is the function that `line` is part
    #      of (string)
    # Return values are:
    #   * A (possibly multi-line) string
    #   * `None`: do not include any debug info at
    #     this point in the trace
    def process(self, path, line, col, fn): ...
```

The default `DebugInfoProcess` in `trace_chewer` is:

```python
class DebugInfoProcess:
  def next_trace(self): pass
  def process(self, path, line, col, fn):
    return f"  ; {path}:{line}:{col} {fn}"
```
