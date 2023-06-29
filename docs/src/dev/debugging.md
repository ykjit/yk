# Debugging.

## Debugging JITted code.

Often you will find the need to inspect JITted code with a debugger. If the
problem trace comes from a C test (i.e. one of the test cases under `tests/c`),
then you can use the `gdb_c_test` tool.

The tool automates the compilation and invocation of the resulting binary
under GDB.

The simplest invocation of `gdb_c_test` (from the top-level of the `yk` repo)
would look like:

```
bin/gdb_c_test simple.c
```

This will automatically compile and run the `tests/c/simple.c` test under GDB.
This would be ideal if you have a crashing trace, as it will dump you into a
GDB shell at the time of the crash.

The tool has some other switches which are useful for other situations, e.g.:

```
bin/gdb_c_test -j -s -b10 simple.c
```

compiles and runs `tests/c/simple.c` test under GDB with [JIT state
debugging](runtime_config.md#ykd_print_jitstate)
enabled, with [compilation
serialised](runtime_config.md#ykd_serialise_compilation), setting a
breakpoint on the first 10 traces compiled.

For a list of all switches available, run:

```
bin/gdb_c_test --help
```

For help on using GDB, see the [GDB
documentation](https://sourceware.org/gdb/documentation/).
