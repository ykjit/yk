# Building and Running Tests

This guide describes how to get up and running with hardware tracing in Yorick.

XXX: software tracing with cranelift.

## Repos

Yorick is spread over a handful of repos, but the two you are likely to need to clone are:

 - [ykrustc](https://github.com/softdevteam/ykrustc): The compiler.
 - [yk](https://github.com/softdevteam/ykrustc): The runtime parts of the system.

The latter is a monorepo containing a few different crates.

## Building and Testing the Compiler

### Building the Compiler

To build the compiler, run:
```
./x.py build --stage 1
```

A stage one build is usually sufficient for development.

If you have made changes to `ykpack` (in the `yk` repo), then edit the
`Cargo.toml` in the top level of `ykrustc` add lines like:

```
[patch."https://github.com/softdevteam/yk"]
ykpack = { path = "../yk/ykpack" }
```

This will override the compiler's dependency to your local version. **Be sure
not to commit this change!**

### Testing the Compiler

The compiler is tested with:
```
./x.py test --stage 1
```

You can do more thorough testing by testing stage 2, but it takes quite a
while. CI will always do a full stage 2 test.

### Build Configuration

You can configure the compiler by placing a config.toml in the top-level of the
`ykrustc` directory. This is not mandatory, but there are scenarios where it can help.

For example, rustc's backtraces are pretty poor by default and cranking the
debug level can sometimes help (at the cost of a much slower build).

`config.toml.example` is upstream's example config. `.buildbot.config.toml` is
the one CI uses.

## Building and Testing in `yk`

To work on `yk` you will need to have built the compiler, as detailed above.
Then the easiest way to get going is to use `rustup` to create a "linked
toolchain" and then override the `yk` repo to use it.

Supposing you have a stage 1 compiler built, you can make a linked toolchain with:
```
rustup toolchain link yk-stage1 /path/to/ykrustc/build/x86_64-unknown-linux-gnu/stage1
```

Then change directory into where you have cloned `yk` and run:
```
rustup override set yk-stage1
```

Now `cargo` will run our compiler (also setting all of the various flags
required) for this directory instead of the default Rust compiler.

To select hardware tracing, you need to set an environment variable:
```
export RUSTFLAGS="-C tracer=hw"
```

Then you can build and test the `yk` repo using `cargo xtask` commands, for
example `cargo xtask test`.

### Gotchas / Tips

 - If you change the ABI of any of the structures in `yk` then you will have to
   re-build the compiler before working on `yk` again. Failure to do this leads
   to weird deserialisation errors.

 - `cargo` will not trigger a rebuild of `yk` if the compiler is rebuilt, so
   you may have to `cargo clean`.

 - If you want to run `rustfmt` in `yk`, it's usually easier to invoke the one
   from the nightly toolchain with `cargo +nightly fmt`. To format the
   compiler, use `./x.py fmt`.

 - Sadly the same trick does not work for `clippy`, as our `-C tracer` flag
   confuses it. To use `clippy` you have to edit `config.toml` to enable
   building it. Even then `cargo clippy` won't work due to a `rustup` bug and
   you have to run the binary manually, setting a `LD_LIBRARY_PATH` as
   necessary.

 - [Continuous Integration Cycles](ci_cycles.md)

 - If you are using rust-analyser (or similar) in vim, make sure to set the
   `RUSTFLAGS` environment the same as you use to build/test, otherwise vim and
   your shell session will keep invalidating
   the incremental build cache and you'll constantly be rebuilding the same
   packages.
