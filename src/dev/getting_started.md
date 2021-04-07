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

First get the compiler sources:
```
git clone https://github.com/softdevteam/ykrustc
cd ykrustc
```

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

First get the yk sources:
```
git clone https://github.com/softdevteam/yk
cd yk
```

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

## Rust Analyzer

[Rust-analyzer](https://github.com/rust-analyzer/rust-analyzer)
is a language server implementation for Rust.

Due to [the workspace model of the `yk` repo](../tech/yk_structure.md) and the
`x.py` build system in `ykrustc`, Rust Analyzer won't work out of the box.
Configuration is dependent on the LSP plugin used, but in general:

 - For the `ykrustc` repo, you need to tell Rust Analyzer to run `x.py` and not
   `cargo`. This is done via the `rust-analyzer.checkOnSave.overrideCommand`
   option.

 - For the `yk` repo, you need to point Rust Analyzer at the extra
   workspaces via the `rust-analyzer.linkedProjects` option.

For example, for [vim-lsp](https://github.com/prabirshrestha/vim-lsp) you would
use a configuration similar to the following:

```
if executable('rust-analyzer')
    if filereadable("x.py")
        " Rust compiler.
        au User lsp_setup call lsp#register_server({
                    \ 'name': 'rust-analyzer',
                    \ 'cmd': {server_info->['rust-analyzer']},
                    \ 'allowlist': ['rust'],
                    \ 'workspace_config': {'rust-analyzer': {'checkOnSave': {'overrideCommand': './x.py check --json-output'}}},
                    \ })
    elseif filereadable("ykshim_client/Cargo.toml")
        " It's the yk repo.
        au User lsp_setup call lsp#register_server({
                    \ 'name': 'rust-analyzer',
                    \ 'cmd': {server_info->['rust-analyzer']},
                    \ 'allowlist': ['rust'],
                    \ 'workspace_config': {'rust-analyzer': {'linkedProjects': ['traced/Cargo.toml', 'untraced/Cargo.toml']}},
                    \ })
    else
        " Normal project.
        "
        " Either install rust-src from rustup or set RUST_SRC_PATH to the
        " `library` subdir of a rust source clone. If you are working on
        " Yorick, then that's the `library` directory in your ykrustc clone.
        au User lsp_setup call lsp#register_server({
                    \ 'name': 'rust-analyzer',
                    \ 'cmd': {server_info->['rust-analyzer']},
                    \ 'allowlist': ['rust'],
                    \ })
    endif
endif
```

Once rust-analyzer is enabled, you should make sure that your editor uses the
same `RUSTFLAGS` as building and testing do. Failure to do so will cause the
incremental build cache to be repeatedly invalidated and you'll constantly be
rebuilding the same packages.

### Gotchas / Tips

 - If you change the ABI of any of the structures in `yk` then you will have to
   re-build the compiler before working on `yk` again. Failure to do this leads
   to weird deserialisation errors.

 - `cargo` will not trigger a rebuild of `yk` if the compiler is rebuilt, so
   you may have to `cargo clean`.

 - `cargo xtask fmt` and `cargo xtask clippy` in the `yk` repo currently
   require a nightly rust toolchain installed via rustup.

 - [Continuous Integration Cycles](ci_cycles.md)
