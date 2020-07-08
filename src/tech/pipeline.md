# The JIT pipeline.

There are several trace representations used at different stages in Yorick's
JIT pipeline.

## PT Traces

The JIT pipeline starts when a binary traces a section of its own execution,
giving us a PT trace.

A PT trace is a raw trace supplied to us by the Intel hardware (via the
[hwtracer](https://github.com/softdevteam/hwtracer/) library). This kind of
trace is effectively a list of block addresses. The addresses are virtual
instruction addresses from the relocated `.text` section of the running binary.
This means that (due to ASLR) these addresses are unique to a particular run of
the program.

## SIR Traces

Given a PT trace, Yorick then constructs a SIR trace. A SIR trace is a list of
<symbol-name, block-index> pairs. Each such pair identifies a SIR block that
execution passed through during tracing.

To make a SIR trace from a PT trace, Yorick uses the
[HWTMapper](https://github.com/softdevteam/yk/blob/master/yktrace/src/hwt/mapper.rs).
When the program to be traced is compiled, ykrustc inserts special debugging
labels (DWARF `DILabel`s) into the binary to help map virtual addresses back to
the SIR.

For this mapping to be correct we rely on LLVM not re-ordering blocks, thus for
now, ykrustc compiles programs without optimisations.

SIR traces are "trimmed", to remove unnecessary blocks. These blocks correspond
with the routines used to start and stop tracing, which themselves get
partially traced.

## TIR Traces

Once Yorick has a SIR trace, it converts it to a TIR trace by:

 - Converting control flow terminators into guards.
 - Renaming variables so that variable indices are unique within the trace.

## Executable Code

The TIR trace is then compiled into native code for the target architecture.

The bulk of the heavy lifting is performed by
[dynasm-rs](https://github.com/CensoredUsername/dynasm-rs), but we do have to
implement our own register allocation.

Once this step is complete, we have an executable code buffer which can be
called (we model the trace as a function).


