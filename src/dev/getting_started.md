# Getting Started

This guide describes how to get up and running with yk.

## Repositories

yk is spread over several git repositories, but you will need to download and
build at least the following two in order to have a running system:

 - [ykllvm](https://github.com/ykjit/ykllvm): Our fork of LLVM.
 - [yk](https://github.com/ykjit/yk): The runtime parts of the system.

Since `yk` depends on `ykllvm`, you must build `ykllvm` first.


## Building `ykllvm`

Clone the repository, switch to the `ykjit/13.x` branch and build `ykllvm`:

```
git clone https://github.com/ykjit/ykllvm
cd ykllvm
mkdir build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../inst \
    -DLLVM_INSTALL_UTILS=On \
    -DCMAKE_BUILD_TYPE=release \
    -DLLVM_ENABLE_ASSERTIONS=On \
    -DLLVM_ENABLE_PROJECTS="lld;clang" \
    ../llvm
make -j `nproc` install
```

In order that your subsequent build(s) of `yk` pick up `ykllvm`, you must
ensure that the `ykllvm` compiler binaries are used instead of your system's
default LLVM binaries. For example, prepend the `ykllvm` installation
directory to your `$PATH`:

```
export PATH=/path/to/ykllvm/inst/bin:${PATH}
```


## Working with the `yk` repo.

The `yk` repo is a Rust workspace (i.e. a collection of crates). You can build
and test in the usual ways using `cargo`.

For example, to build and test the system, run:

```
cargo test
```

The only requirement is that the LLVM binaries in your `$PATH` are those from a
compiled `ykllvm` (see the previous section).


### C++ code

The `yk` repo contains some C++ code for interacting with LLVM. `cargo fmt`
will not format C++ code, and you should instead use `cargo xtask cfmt`.
