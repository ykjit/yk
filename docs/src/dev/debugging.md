# Debugging

## Disabling the JIT

Sometimes it is useful to completely disable the JIT to verify that a problem is
JIT-related. You can do this with the `YK_JITC` environment variable (see
[run-time configuration](runtime_config.md) for details).

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
then you can use `bin/rr_c_test` or `bin/gdb_c_test`.

Run these tools with no arguments for help on how to use them.

Note that because these tools don't run tests under `lang_tester`, environment
variables usually set by `env-var` in the test file will not be applied. If
need be, you can set them manually, e.g. `YKD_SERAIALISE_COMPILATION=1
./bin/rr_c_test ...`.

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
