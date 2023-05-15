# Installation

This section details how to get yk up and running.

## System Requirements

At the time of writing, yk requires the following:

 - A Linux system with a CPU that supports Intel Processor Trace.
   (`grep intel_pt /proc/cpuinfo` to check)

 - Linux perf (for collecting PT traces).

 - A [Yk-enabled programming language interpreter](interps.md).

 - A recent nightly install of [Rust](https://www.rust-lang.org/).

Note that at present, non-root users can only use Yk if
`/proc/sys/kernel/perf_event_paranoid` is set to `-1`.


## Building

Clone the [main yk repository](https://github.com/ykjit/yk) and build
it with `cargo`:

```sh
$ git clone --recurse-submodules --depth 1 \
  https://github.com/ykjit/yk/
$ cd yk
$ cargo build --release
```

Note that this will also clone [ykllvm](https://github.com/ykjit/ykllvm) as a
submodule of yk. If you later want access to the full git history, either
remove `--depth 1` or run `git fetch --unshallow`.
