# The Yorick meta-tracer.

Yorick is a fork of the Rust language which aims to build a meta-tracing system
from Rust. Imagine PyPy, but where your interpreter is written in Rust instead
of RPython.

This repository houses the non-compiler components of the system.

The compiler repo can be found [here](https://github.com/softdevteam/ykrustc).

## Getting Started

To get started, see
[this section in the documentation](https://softdevteam.github.io/ykdocs/dev/getting_started.html).

To work with this repository, instead of running `cargo <target>`, instead
execute `cargo xtask <target>`.

You will also need to add the `-C tracer=<kind>` flag to `RUSTFLAGS` to tell
the build system what kind of tracer you want to use (`hw` or `sw`, although
only `hw` works at the moment).

For example, to test using hardware tracing:
```
$ export RUSTFLAGS="-C tracer=hw"
$ cargo xtask test
```

## Contributors

Yorick is developed by a team of people listed in the
[contributors page](https://github.com/softdevteam/yk/graphs/contributors).
