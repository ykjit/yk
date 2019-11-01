# Intermediate Representations

Yorick uses two additional intermediate representations (IRs) on top of those
already found in rustc:

 * Serialised IR (SIR)
 * Tracing IR (TIR)

## Serialised IR (SIR)

During compilation the Rust's Middle Intermediat Representation (MIR) is
traversed and serialised into a simpler
representation called SIR. SIR is a flow-graph IR very similar to MIR. It
mostly exists so that high-level program information can be reconstructed at
runtime without a need for an instance of the compiler (and its `tcx` struct).

The resulting SIR is serialised using serde and linked into the `.yk_sir` ELF
section of binaries compiled with `ykrustc`. At runtime, the tracer will
collect SIR traces, which can then be mapped back to the serialised SIR
information.

The SIR data structures are in an
[externally maintained crate](https://github.com/softdevteam/yk/tree/master/ykpack)
so that they can be shared by the compiler and the JIT runtime.

SIR lowering
[is performed here](https://github.com/softdevteam/ykrustc/blob/master/src/librustc_yk_sections/emit_sir.rs).

## Tracing IR (TIR)

TIR is basically SIR with guards instead of branches. TIR is the basis for a
compiled trace.

TIR [lives in yktrace](https://github.com/softdevteam/yk/tree/master/yktrace).
