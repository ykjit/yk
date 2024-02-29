# Configuring the build

Start by following the [general installation
instructions](../user/install.html#building).

The `yk` repo is a Rust workspace (i.e. a collection of crates). You can build
and test in the usual ways using `cargo`. For example, to build and test the
system, run:

```
cargo test
```


## `YKB_YKLLVM_BIN_DIR`

Under normal circumstances, yk builds a copy of its LLVM fork "ykllvm", which
it also uses it to build interpreters (via the compiler's use of `yk-config`).
You can use your own ykllvm build by specifying the directory where the
executables (e.g. `clang`, `llvm-config`, and so on) are stored with
`YKB_YKLLVM_BIN_DIR`.

yk does not check your installation for compatibility: it is your
responsibility to ensure that your ykllvm build matches that expected by yk.

It is also undefined behaviour to move between defining this variable and not
within a repository using `yk` (including the `yk` repository itself). If you
want to set/unset `YKB_YKLLVM_BIN_DIR` then `cargo clean` any repositories
using `yk` before rebuilding them.

## `YKB_TRACER`

The `YKB_TRACER` environment variable allows building yk with either `hwt` 
(Hardware Tracer) or `swt` (Software Software Tracer).

`hwt` - Relies on Intel PT, suitable only for x86 CPUs supporting it.

`swt` - CPU architecture-independent, but with fewer features compared to 
`hwt`.

## `yk_testing`

yk has an internal Rust feature called `yk_testing`. It is enabled whenever the
`tests` crate is being compiled, so a regular `cargo build` in the root of the
workspace will enable the feature (to build *without* the feature enabled, do
`cargo build -p ykcapi`).

## clangd

The `yk` build system generates compilation command databases for use with
clangd. If you want diagnostics and/or completion in your editor (via an LSP),
you will have to configure the LSP to use `clangd` (the automated build system
puts a `clangd` binary into `target/<debug|release>/ykllvm/bin` that you could
use).
