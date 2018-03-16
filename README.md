# hwtracer

A small Rust/C library to trace a program using CPU tracing technology.

This library only supports Intel Processor Trace, but in the future we hope to
support alternatives such as Arm's CoreSight.

**This is experimental code.**

## Notes

When running `cargo`, you can set `IPT_PATH=...` to specify a path to a system
libipt to use. If this variable is absent, Cargo will download and build libipt
for you.

If you let Cargo download a libipt, you will have to set your `LD_LIBRARY_PATH`
to `c_deps/inst/lib` before using hwtracer. This is due to a Cargo limitation.
