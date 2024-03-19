// Functions exported only for testing.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <yk.h>

/// The statistics passed to the `test` function in `ykstats_wait_until`.
// Note: this struct *must* stay in sync with `YkCStats` in `ykstats.rs`.
typedef struct {
  /// How many traces were recorded successfully?
  uint64_t traces_recorded_ok;
  /// How many traces were recorded unsuccessfully?
  uint64_t traces_recorded_err;
  /// How many traces were compiled successfully?
  uint64_t traces_compiled_ok;
  /// How many traces were compiled unsuccessfully?
  uint64_t traces_compiled_err;
  /// How many times have traces been executed? Note that the same trace can
  /// count arbitrarily many times to this.
  uint64_t trace_executions;
} YkCStats;

/// Iff `YKD_STATS` is set, suspend this thread's execution until
/// `test(YkCStats)` returns true. The `test` function will be called one or
/// more times: as soon as `test` returns `true`, `__ykstats_wait_until` itself
/// returns. This allows a test to wait for e.g. a certain number of traces to
/// be recorded/compiled, even if that happens in another thread.
///
/// Note that a lock is held on yk's statistics while `test` is called, so
/// `test` should not perform lengthy calculations (if it does, it may block
/// other threads). Note also that the `YkCStats` struct it is passed only has
/// valid values for the duration of `test`'s execution: those stats may become
/// invalid immediately after `test` returns.
///
/// This function will panic if `YKD_STATS` is not set.
void __ykstats_wait_until(YkMT *mt, bool test(YkCStats));

// This function will only exist if the `hwt` tracer is compiled in to ykrt.
size_t __yktrace_hwt_mapper_blockmap_len();

// Blocks the compiler from optimising the specified value or expression.
//
// This is similar to the non-const variant borrowed from Google benchmark:
// https://github.com/google/benchmark/blob/e451e50e9b8af453f076dec10bd6890847f1624e/include/benchmark/benchmark.h#L350
//
// Our version works on a value, rather than a pointer.
//
// Note that Google Benchmark also defines a variant for constant data. At the
// time of writing, NOOPT_VAL seems to suffice (even for constants), but we may
// need to consider using the const version later.
#ifdef __clang__
#define NOOPT_VAL(X) asm volatile("" : "+r,m"(X) : : "memory");
#else
#error non-clang compilers are not supported.
#endif

// Tries to block optimisations by telling the compiler that all memory
// locations are touched. `NOOPT_VAL` is preferred, but you may not always have
// direct access to the value(s) or expression(s) that you wish to block
// optimisations to.
//
// Borrowed from:
// https://github.com/google/benchmark/blob/ab74ae5e104f72fa957c1712707a06a781a974a6/include/benchmark/benchmark.h#L359
#define CLOBBER_MEM() asm volatile("" : : : "memory");

// Stuff for decoder benchmarks.
void *__hwykpt_start_collector(void);
void *__hwykpt_stop_collector(void *tc);
void __hwykpt_libipt_vs_ykpt(void *trace);
bool __hwykpt_decode_trace(void *trace);
