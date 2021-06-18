# Getting Started

This guide describes how to get up and running with yk.

## Repos

yk is spread over a handful of repos, but the two you are likely to need to clone are:

 - [ykllvm](https://github.com/ykjit/ykllvm): Our fork of LLVM.
 - [yk](https://github.com/ykjit/yk): The runtime parts of the system.

The latter is a monorepo containing a few different Rust crates.

## Building `ykllvm`

First get the sources:
```
git clone https://github.com/ykjit/ykllvm
cd ykrustc
```

To build, run:
```
./x.py build --stage 1
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

Be sure to put your newly built binaries in your `$PATH`. E.g.:
```
export PATH=/path/to/inst/bin:${PATH}
```

## Working with the `yk` repo.

The `yk` repo is a Rust workspace. You can build and test in the usual ways
using `cargo`.

For example, to build and test the system, run:
```
cargo test
```

The only requirement is that the LLVM binaries in your `$PATH` are those from a
compiled `ykllvm` (see the previous section).

### C++ code

The `yk` repo contains some C++ code for interacting with LLVM. `cargo fmt`
will not format C++ code, and you should instead use `cargo xtask cfmt`.
