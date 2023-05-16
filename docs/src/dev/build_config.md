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


## `yk_testing`

yk has an internal Rust feature called `yk_testing`. It is enabled whenever the
`tests` crate is being compiled, so a regular `cargo build` in the root of the
workspace will enable the feature (to build *without* the feature enabled, do
`cargo build -p ykcapi`).
