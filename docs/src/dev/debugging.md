# Debugging

## Trace optimisation

Trace optimisation can make it difficult to understand why a yk interpreter has
behaved in the way it does. It is worth trying to run your code with the
optimiser turned off. You can do this with the `YKD_OPT` environment
variable, which takes the following values:

  * 1: turn the optimiser on. Default if not otherwise specified.
  * 0: turn the optimiser off.


## Debugging JITted code

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

To see what else you can do with `gdb_c_test`, run:

```
bin/gdb_c_test --help
```

For help on using GDB, see the [GDB
documentation](https://sourceware.org/gdb/documentation/).

### GDB plugin

Yk comes with a GDB plugin that allows the debugger to show higher-level
information in the source view window.

The plugin is built by default and put in `target/yk_gdb_plugin.so`.

To use it, put this line in `~/.gdbinit`:
```
jit-reader-load /path/to/yk/target/yk_gdb_plugin.so
```

Then when you run GDB, you should see:
```
Yk JIT support loaded.
```

When you are inside JITted code, the source view will show higher-level
debugging information. You can show the assembler and source views on one GDB
screen using the "split" layout. Type:

```
la spl
```
