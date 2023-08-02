# Profiling

This section describes how best to profile yk and interpreters.


## JIT statistics

At the end of an interpreter run, yk can print out some simple statistics about
what happened during execution. If the `YKD_JITSTATS=<path>` environment
variable is defined, then JSON statistics will be written to the file at
`<path>` once the interpreter "drops" the `YkMt` instance. `-` (i.e. a single
dash) can be used in place of path, in which case the statistics will be
written to `stderr`. Note that if the interpreter starts multiple yk instances,
then the contents of `<file>` are undefined (at best the file will be
nondeterministically overwritten as instances are "dropped", but output may
be interleaved, or otherwise bizarre).

Output from `YKD_JITATS` looks as follows:

```
{                                       
    "duration_compiling": 5.5219,                                               
    "duration_deopting": 2.2638,
    "duration_jit_executing": 0.2,
    "duration_outside_yk": 0.142,
    "duration_trace_mapping": 3.3797,                                           
    "traces_collected_err": 0,                                                  
    "traces_collected_ok": 11,                                                  
    "traces_compiled_err": 1,
    "traces_compiled_ok": 10                                                    
}
```

Fields and their meaning are as follows:

 * `duration_compiling`. Float, seconds. How long was spent compiling traces?
 * `duration_deopting`. Float, seconds. How long was spent deoptimising from
   failed guards?
 * `duration_jit_executing`. Float, seconds. How long was spent executing JIT
   compiled code?
 * `duration_outside_yk`. Float, seconds. How long was spent outside yk? This
   is a proxy for "how much time was spent in the interpreter", but is inherently
   an over-approximation because we can't truly know exactly what the system
   outside Yk counts as "interpreting" or not. For example, if an interpreter
   thread puts itself to sleep, we will still count it as time spent
   "outside yk".
 * `duration_trace_mapping`. Float, seconds. How long was spent mapping a "raw"
   trace to compiler-ready IR?
 * `trace_executions`. Unsigned integer. How many times have traces been
   executed? Note that the same trace can count arbitrarily many times to this.
 * `traces_collected_err`. Unsigned integer. How many traces were collected
   unsuccessfully?
 * `traces_collected_ok`. Unsigned integer. How many traces were collected
   successfully?
 * `traces_compiled_err`. Unsigned integer. How many traces were compiled
   unsuccessfully?
 * `traces_compiled_ok`. Unsigned integer. How many traces were compiled
   successfully?


## Perf

Note that `yk-config --cflags` includes `-Wl,--no-rosegment`, which [gives
better profiling
information](https://github.com/flamegraph-rs/flamegraph#cargo-flamegraph) for
binaries linked with `lld` (as all yk C interpreters are). 

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

### Flame graphs

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
