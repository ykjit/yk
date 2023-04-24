# Installation

This section details how to get up and running with Yk.

# System Requirements

At the time of writing, running Yk requires the following:

 - A Linux system with a CPU that supports Intel Processor Trace.
   (`grep intel_pt /proc/cpuinfo` to check)

 - Linux perf (for collecting PT traces).

 - A [Yk-enabled programming language interpreter](interps.md).

Note that at present, non-root users can only use Yk if
`/proc/sys/kernel/perf_event_paranoid` is set to `-1`.

# Getting Yk

Eventually we'd like to distribute source and binary releases of Yk, but we are
not there yet.

In the meantime, see the [developer instructions](../dev/getting_started.md) on
how to build from git source.
