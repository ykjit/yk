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

* [`YKD_LOG_IR`](understanding_traces.html#ykd_log_ir) [with the `ykd` feature]
* [`YKD_TRACE_DEBUGINFO`](understanding_traces.html#ykd_trace_debuginfo) [with the `ykd` feature]
* [`YKD_LOG_JITSTATE`](understanding_traces.html#ykd_log_jitstate) [with the `ykd` feature]
* [`YKD_LOG_STATS`](profiling.html#jit-statistics)
