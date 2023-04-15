# Profiling

This section describes how best to profile yk (and its consumers).

Note that `yk-config --cflags` includes `-Wl,--no-rosegment`, which [gives
better profiling
information](https://github.com/flamegraph-rs/flamegraph#cargo-flamegraph) for
binaries linked with `lld` (as all yk C interpreters are). 

## Perf

One way to view profile data is with perf.

```
$ perf record --call-graph dwarf -g ./interpreter ...args...
$ perf report -G --no-inline
```

`--call-graph=dwarf` makes perf use DWARF debugging information, which gives
better quality profiling data than the default, but requires that you compile
with debugging info.

We recommend passing `--no-inline` since, for all but the smallest runs,
omitting it will lead to really long waits while `addr2line` is run (newer
versions of `perf` [may fix
this](https://eighty-twenty.org/2021/09/09/perf-addr2line-speed-improvement),
but at the time of writing, the `perf` included in Debian is slow).

## Flame graphs

The most convenient way to make a flame graph is to use the Rust
[`flamegraph`](https://github.com/flamegraph-rs/flamegraph) tool.

On Linux, this tool uses `perf` under the hood and already passes
`--call-graph=dwarf` to `perf record`  by default (pass `-v` to see the exact
perf invocation used).

To install:

```
$ cargo install flamegraph
```

Then run the binary you want to profile:

```
$ ~/.cargo/bin/flamegraph --no-inline -- ./interpreter ...args...
```

This will spit out an `svg` file which you can then view in (e.g.) a web browser.
