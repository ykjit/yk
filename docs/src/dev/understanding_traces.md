# Understanding Traces

yk can print the traces it has created to `stderr` to help with debugging.
However, these traces are often lengthy, and not always easy to understand.
This section briefly explains how to get yk to print its traces, and how
to make them a bit easier to understand.


## Producing a trace

The `YKD_PRINT_IR` environment variable determines whether yk prints traces to
`stderr` or not. If `jit-pre-opt` is specified, the traces will be printed before
optimisation; if `jit-post-opt` is specified, the traces will be printed after
optimisation. `jit-pre-opt` and `jit-post-opt` can give you different insights,
so it is often worth checking both.


## trace_chewer

`trace_chewer` is a small program included with yk which can help you
understand a trace.


### simplify

`trace_chewer simplify <file-name>` takes as input a trace and converts
it into a straight-line trace. It removes LLVM declarations and guard
failure blocks, neither of which are enlightening. It converts LLVM
branches into simpler `guard_true(%var)` or `guard_false(%var)` statements
(i.e. `guard_true(%3)` means "if the boolean value stored in `%3` is true,
then continue executing the trace, otherwise deoptimise back to the
interpreter).

If you specify `-` as the filename, `trace_chewer` will read from stdin. This
means that you can simplify traces without saving them to disk:

```
YKD_PRINT_IR=jit-post-opt lua f.lua 2>&1 | trace_chewer simplify -
```
