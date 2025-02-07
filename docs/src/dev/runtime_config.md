# Run-time configuration

There are a number of environment variables which control the run_time
behaviour of the yk system.


## General configuration

Variables prefixed with `YK_` allow the user to control aspects of yk's execution.

The following environment variables are available:

* `YK_HOT_THRESHOLD`: an integer from 0..4294967295 (both inclusive) that
  determines how many executions of a hot loop are needed before it is traced.
  Defaults to 50.


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
  Currently: level 3 logs transitions of a `Location` transition; and level 4
  JIT events such as starting/stopping tracing. Note that some information, at
  all levels, may or may not be displayed based on compile-time options.
  Defaults to 1.
* [`YKD_LOG_IR`](understanding_traces.html#ykd_log_ir) [with the `ykd` feature]
* [`YKD_LOG_STATS`](profiling.html#jit-statistics)
