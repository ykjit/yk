# Intermediate Representations

Yorick uses two additional intermediate representations (IRs) on top of those
already found in rustc:

 * Serialised IR (SIR)
 * Tracing IR (TIR)

## Serialised IR (SIR)

During LLVM codegen, ykrustc generates SIR. SIR exists so that high-level
program information can be reconstructed at runtime without a need for an
instance of the compiler (and its `tcx` struct). SIR is serialised using serde
and linked into special ELF sections in the resulting binary (one section per
crate, whose names are prefixed `.yksir_`).

SIR is not in SSA form, but this may change in the future.

The SIR data structures are in an
[externally maintained crate](https://github.com/softdevteam/yk/tree/master/ykpack)
so that they can be shared by the compiler and the JIT runtime.

The
[SirCx](https://github.com/softdevteam/ykrustc/blob/master/src/librustc/sir.rs)
is a per-codegen-unit structure that holds all state related to SIR.

Often we have to lookup SIR constructs (functions, blocks, etc.) identified in
the codegen only by an opaque LLVM pointer. For this reason the `SirCx` has to
hold a cache that maps these pointers back to corresponding Sir structures.

### Why is SIR Implemented in the LLVM Codegen?

Initially SIR was generated from Rust's Middle Intermediate Representation
(MIR). This was much simpler and was backend agnostic, but it meant that we
were unable to resolve monomorphised call targets, as monomorphisation happens
later in the codegen.

## Tracing IR (TIR)

TIR is basically SIR with guards instead of branches. TIR is the basis for a
compiled trace.

TIR [lives in yktrace](https://github.com/softdevteam/yk/tree/master/yktrace).

## Local Variables

MIR, SIR and TIR all have the notion of local variables. Each local variable is
identified by an integer (its *index*). These indices are unique only within
the scope of a function.

At the time of writing, there is a one-to-one relationship between the variable
indices in MIR and SIR, however this may change in the future.

Before a TIR trace is compiled, variables are renamed so that (e.g.) variable
index `1` from function `f()` does not clash with variable index `1` from
function `g()`.
