# Run-time configuration

There are a number of environment variables which control the run_time
behaviour of the yk system.

Variables prefixed with `YKD_` are intended to help interpreter authors debug
performance issues. Some are only available in certain compile-time
configurations of yk, either because they increase the binary size, or slow
performance down.

The following environment variables are available (some only in certain configurations of yk):

* [`YKD_LOG_IR`](/dev/understanding_traces.html#ykd_log_ir) [with the `ykd` feature]
* [`YKD_TRACE_DEBUGINFO`](/dev/understanding_traces.html#ykd_trace_debuginfo) [with the `ykd` feature]
* [`YKD_LOG_JITSTATE`](/dev/understanding_traces.html#ykd_log_jitstate) [with the `ykd` feature]
* [`YKD_LOG_STATS`](/dev/profiling.html#jit-statistics)
