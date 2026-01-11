# Run-time configuration

There are a number of environment variables which control the run_time
behaviour of the yk system.


## General configuration

Variables prefixed with `YK_` allow the user to control aspects of yk's execution.

The following environment variables are available:

* `YK_HOT_THRESHOLD`: an integer from 0..4294967295 (both inclusive) that
  determines how many executions of a hot loop are needed before it is traced.
  Defaults to 131.
* `YK_JITC`: selects the JIT compiler to use. Set to "none" to disable JIT
  compilation entirely. When disabled, the hot location counter will not
  increment, preventing any tracing or compilation from occurring. When not set,
  defaults to JIT compilation enabled.
* `YK_JOBS`: specifies the number of threads for compilation. Negative values
  will lead to an error; a value of 0 will be treated as a value of 1. Defaults
  to `num_cpus - 1`.
* `YK_SIDETRACE_THRESHOLD`: an integer from 0..4294967295 (both inclusive) that
  determines how many times a guard needs to fail before a sidetrace is created.
  Defaults to 5.


## Debugging

Variables prefixed with `YKD_` are intended to help interpreter authors debug
performance issues. Some are only available in certain compile-time
configurations of yk, either because they increase the binary size, or slow
performance down.

The following environment variables are available (some only in certain configurations of yk):

* `YKD_LOG=[<path>:]<level>` specifies where, and how much, general information
  yk will log during execution.

  If `<path>:` (i.e. a path followed by ":") is specified then output is sent
  to that path. The special value `-` (i.e. a single dash) can be used for
  `<path>` to indicate stderr. If not specified, logs to stderr.

  `<level>` specifies the level of logging, each adding to the previous: level
  0 turns off all yk logging; level 1 shows major errors only; and level 2
  warnings. Levels above 3 are used for internal yk debugging, and their
  precise output, and indeed the maximum level may change without warning.
  Currently: level 3 shows tracing events (e.g. starting/stopping tracing);
  and level 4 shows trace execution and deoptimisation. Note that some
  information, at all levels, may or may not be displayed based on compile-time
  options. Defaults to 1.
* [`YKD_LOG_IR`](understanding_traces.html#ykd_log_ir) [with the `ykd` feature]
* [`YKD_LOG_STATS`](profiling.html#jit-statistics)
* `YKD_TPROF`: When "1" turns on [trace profiling support](profiling.html) (if
  implemented for the current platform).
