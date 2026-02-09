# Gotchas

Yk has some unusual implications, requirements, and limitations that interpreter
authors should be aware of.

## Trace quality depends upon completeness of interpreter IR.

Yk works by recording traces (i.e. individual control flow paths) of your
interpreter's implementation. Each trace ends up as an ordered list of LLVM IR
blocks which are "stitched together" and compiled. Unless callees are marked
`yk_noinline`, the JIT will seek to inline them into the trace because,
generally speaking, the more the JIT can inline, the more optimal the JITted
code will be.

In order to inline a function call, the JIT needs to have LLVM IR for the
callee. Yk uses fat LTO to collect (and embed into the resulting binary) a
"full-program" IR for your interpreter. `yk-config` provides the relevant clang
flags to make this happen. You should make sure that the build system of your
interpreter uses the relevant flags. Namely `yk-config --cppflgs --cflags` for
compiling C code, and `yk-config --ldflags --libs` for linking.

It follows that shared objects, which at the time of writing cannot take part
in LTO, cannot be inlined into traces. If your interpreter `dlopen()`s shared
objects at runtime (as is common for C extensions) Yk will be unable to trace
the newly loaded code.

## Symbol visibility

The JIT relies upon the use of `dlsym()` at runtime in order to lookup any
given symbol from its virtual address. For this to work all symbols must be
exposed in the dynamic symbol table.

`yk-config` provides flags to put every function's symbol into the dynamic
symbol table. Since distinct symbols of the same name can exist (e.g. `static`
functions in C), but dynamic symbol names must be unique, symbols may be
mangled (mangling is done by the fat LTO module merger). If your interpreter
does its own symbol introspection, Yk may break it.

## Extra sections in your interpreter binary.

`yk-config` will add flags that add the following sections to your binary:

 - `.llvmbc`: LLVM bytecode for your interpreter. Used to construct traces.
   This is a standard LLVM section (but extended by Yk).
 - `.llvm_stackmaps`: Stackmap table. Used to identify the locations of live
   LLVM IR variables. This is a standard LLVM section (but extended by Yk).

## Other interpreter requirements and gotchas

 - Yk can only currently work with "simple interpreter loop"-style interpreters
   and cannot yet handle unstructured interpreter loops (e.g. threaded
   dispatch).

 - Yk currently assumes that no new code is loaded at runtime (e.g.
   `dlopen()`), and that no code is unloaded (e.g. `dlclose()`). Self modifying
   interpreters will also confuse the JIT.

 - Yk currently doesn't handle calls to `pthread_exit()` gracefully ([more
   details](https://github.com/ykjit/yk/issues/525)).

 - Yk currently doesn't handle `setjmp()`/`longjmp()`.

 - You cannot valgrind an interpreter that is using Intel PT for tracing ([more
   details](https://github.com/ykjit/yk/issues/177)).
