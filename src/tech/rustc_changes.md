# Our Rustc Changes

Ykrustc is a fork of the Rust compiler. This page lists what we have changed.

 - Added the command line flag `-C tracer=<t>`, where `<t>` can be `hw`, `sw`
   or `off` (hardware tracing, software tracing, no tracing).

 - Added the `tracermode` configuration macro so that we can do conditional
   compilation based on the value of `-C tracer` (e.g. `#[cfg(tracermode =
   hw)]`).

 - Added the `yk-sir` option to `--emit` allowing dumping of SIR to file.

 - When `-C tracer` is `hw`, optimisations are disabled to prevent LLVM from
   re-ordering blocks, as this would interfere with the way that we map Intel
   Processor Trace (PT) addresses back to our IR.

 - When `-C tracer` is `hw`, we insert DWARF `DILabel`s whose names encode
   block locations within our IR. These labels are used to map a PT trace to a
   SIR trace.
    - The DWARF labels are post-processed after linkage to put them into a
      faster to read ELF section.

 - In `rustc_codegen_llvm` we lower each monomorphised MIR body (and its
   associated types) into our own IR (SIR). Each codegen unit encodes one ELF
   section containing such IR.

 - We pass `--export-dynamic` to the linker so that we can look up symbols from
   the main text section at runtime.

 - Added our own continuous integration configuration and associated scripts,
   and remove upstream's.

 - Added several new attributes:
   - `do_not_trace`: prevents a function from being inlined in a trace.
   - `interp_step`: marks the user's bytecode dispatch function.
   - `trace_debug`: Inserts a comment into the TIR trace.

 - Added tests.
