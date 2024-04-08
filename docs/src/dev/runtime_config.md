# Run-time configuration

There are a number of environment variables which control the run_time
behaviour of the yk system.

Variables prefixed with `YKD_` are intended for debugging only. Most (if not
all) of the debugging variables introduce extra computation that slows down
program execution, sometimes significantly.

The following environment variables are available (some only in certain configurations of yk):

* [`YKD_PRINT_IR`](/dev/understanding_traces.html#ykd_print_ir)
* [`YKD_TRACE_DEBUGINFO`](/dev/understanding_traces.html#ykd_trace_debuginfo)
* [`YKD_LOG_JITSTATE`](/dev/understanding_traces.html#ykd_log_jitstate)
* [`YKD_STATS`](/dev/profiling.html#jit-statistics)
