# Profiling

This section describes how best to profile yk and interpreters.


## JIT statistics

At the end of an interpreter run, yk can print out some simple statistics about
what happened during execution. If the `YKD_LOG_STATS=<path>` environment
variable is defined, then JSON statistics will be written to the file at
`<path>` once the interpreter "drops" the `YkMt` instance. The
special value `-` (i.e. a single dash) can be used for `<path>` to indicate stderr.

Note that if the interpreter starts multiple yk instances, then the contents of
`<file>` are undefined (at best the file will be nondeterministically
overwritten as instances are "dropped", but output may be interleaved, or
otherwise bizarre).

Output from `YKD_LOG_STATS` looks as follows:

```
{                                       
    "duration_compiling": 5.5219,                                               
    "duration_deopting": 2.2638,
    "duration_jit_executing": 0.2,
    "duration_outside_yk": 0.142,
    "duration_trace_mapping": 3.3797,
    "duration_tracing": 1.2345,
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
 * `duration_tracing`. Float, seconds. How long was spent tracing?
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

On Linux, `perf` can be used to profile yk. You first need to record an
execution of an interpreter and then separately view the profiling data that
was generated.


### Recording a Profile

To record a profile we first recommend compiling yk with debugging info
embedded. cargo's `debug` profile does this automatically, but because no code
optimisation is performed, the profiles are unrepresentative. We recommend
using yk's provided `release-with-debug` profile, which turns on
`--release`-style code optimisation *and* embeds debugging information:

```
$ cargo build --profile=release-with-debug
```

Ensure that the interpreter you are profiling links to the appropriate version
of yk and then run:

```
$ /path/to/yk/bin/yk_perf_record --call-graph dwarf -g ./interpreter ...args...
```

This uses `--call-graph dwarf` to force perf use DWARF debugging information:
this will only be useful if you have compiled yk with embedded debugging
information, as recommended above.

The `yk_perf_record` wrapper script sets `YKD_TPROF=1` and automates the task
of getting JITted code into the profile.


### Viewing a profile

perf profiles can be visualised in a number of ways. When using `perf report`
or `perf script` we currently recommend passing `--no-inline` to avoid the huge
processing time incurred by indirectly running `addr2line` (note that this
[might change in the
future](https://eighty-twenty.org/2021/09/09/perf-addr2line-speed-improvement)).


#### Terminal

To quickly view a profile in the terminal:

```
$ perf report -g --no-inline
```


#### Firefox profiler

After processing perf's output, you can use [Firefox's
Profiler](https://profiler.firefox.com/) to view the data locally. Note that
this does not upload the data --- all processing happens in your browser! First
process the data:

```
$ perf script -F +pid --no-inline > out.perf
```

Then go to the [Firefox Profiler page](https://profiler.firefox.com/), press
"Load a profile from file" and upload `out.perf`.


#### Flame graphs

You can make a flame graph using the Rust
[`flamegraph`](https://github.com/flamegraph-rs/flamegraph) tool. Install with
`cargo install flamegraph` and then use `flamegraph` to profile and produce a
flamegraph in one go with:

```
$ /path/to/cargo/bin/flamegraph --no-inline -- ./interpreter ...args...
```

Note that `flamegraph` passes `--call-graph=dwarf` to `perf record` by default
(pass `-v` to see the exact perf invocation used).

This will produce an `svg` file which you can then view.
