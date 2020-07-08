# Software and Hardware Tracing

Yorick has two tracing modes:

 * Software tracing.
 * Hardware tracing.

## Software Tracing

Note that for now we have stopped development on software tracing, as we are
focussing on hardware tracing.

We hope to re-visit software tracing later.

## Hardware Tracing

In hardware tracing mode, we use
[Intel Processor Trace](https://software.intel.com/en-us/blogs/2013/09/18/processor-tracing)
to do trace collection. The chip gives us a trace of virtual addresses which we
then map back to SIR locations using DWARF labels (`DILabel`).

### Further Reading

* The LLVM code for the insertion of the Yorick debug labels can be found
  [here](https://github.com/softdevteam/ykrustc/blob/master/src/rustllvm/RustWrapper.cpp).
  Those functions can be accessed from within rustc's code generator using
  helper functions in the
  [codegen builder](https://github.com/softdevteam/ykrustc/blob/master/src/librustc_codegen_llvm/builder.rs).

* The actual label generation happens during the code generation of
  [blocks](https://github.com/softdevteam/ykrustc/blob/master/src/librustc_codegen_ssa/mir/block.rs).
  Labels are inserted at the beginning of each block, as well as when returning
  from function calls.

## Selecting a Tracing Mode

When you build a Rust program that you want to trace, both `ykrustc` (for the
standard library), and your code must be built for a specific tracing backend.

To choose a backend, you pass `-C tracer=T` to `rustc`, where `T` is one of
`hw`, `sw, or `off`. Passing `off` is the same as omitting the option
altogether.

If you are using `cargo`, you will need to add this flag to the `RUSTFLAGS`
environment.

`-C tracer` is a tracked flag: changing it will trigger a rebuild (but bear in
mind that your standard library will not be rebuilt. See below).

Note that hardware tracing currently doesn't work together with optimisations,
and is thus automatically disabled whenever optimisations are enabled.

## Considerations when Building ykrustc Itself.

When you build ykrustc using `x.py` you will need to decide what tracing
support the standard library should be built with. You must set
`STD_TRACER_MODE` to `hw`, `sw`, or `off`. If you fail to set this variable,
the bootstrap will refuse to run.
