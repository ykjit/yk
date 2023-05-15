# Getting Started

yk is spread over several git repositories, but in most cases you only need the
[main yk repository](https://github.com/ykjit/yk). First clone it, and then
build it:

```sh
$ git clone --recurse-submodules --depth 1 \
  https://github.com/ykjit/yk/
$ cd yk
$ cargo build --release
```

Note that this will also clone [ykllvm](https://github.com/ykjit/ykllvm) as a
submodule of yk. If you want access to the full git history, either remove
`--depth 1` or run `git fetch --unshallow`.

The `yk` repo is a Rust workspace (i.e. a collection of crates). You can build
and test in the usual ways using `cargo`. For example, to build and test the
system, run:

```
cargo test
```


### C++ code

The `yk` repo contains some C++ code for interacting with LLVM. `cargo fmt`
will not format C++ code, and you should instead use `cargo xtask cfmt`.
