// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     1
//     yk-tracing: stop-tracing
//     1
//     yk-execution: enter-jit-code {"trid": "0"}
//     1
//     yk-execution: deoptimise {"trid": "0", "gidx": "1"}
//     yk-tracing: start-side-tracing
//     2
//     yk-tracing: stop-tracing
//     2
//     yk-execution: enter-jit-code {"trid": "0"}
//     yk-execution: deoptimise {"trid": "0", "gidx": "3"}
//     yk-tracing: start-side-tracing
//     2
//     yk-tracing: stop-tracing
//     2
//     yk-execution: enter-jit-code {"trid": "0"}
//     2
//     yk-execution: deoptimise {"trid": "2", "gidx": "0"}
//     yk-tracing: start-side-tracing
//     retired 2
//     yk-tracing: stop-tracing
//     3
//     yk-execution: enter-jit-code {"trid": "0"}
//     yk-execution: deoptimise {"trid": "0", "gidx": "3"}
//     yk-tracing: start-side-tracing
//     3
//     yk-tracing: stop-tracing
//     3
//     yk-execution: enter-jit-code {"trid": "0"}
//     3
//     yk-execution: deoptimise {"trid": "4", "gidx": "0"}
//     yk-tracing: start-side-tracing


// Check that when a side-trace is retired, that guard subsequently deoptimises.

#include <inttypes.h>
#include <stdio.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) static void run(YkMT *mt, YkLocation *loc,
                                          uint64_t *values, int start, int end) {
  for (int i = start; i < end; i++) {
    yk_mt_control_point(mt, loc);
    uint64_t id = yk_promote_code_id(values[i]);
    fprintf(stderr, "%" PRIu64 "\n", id);
  }
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();
  uint64_t values[] = {1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3};
  NOOPT_VAL(values);
  run(mt, &loc, values, 0, 8);
  yk_retire_code_id(mt, 2);
  fputs("retired 2\n", stderr);
  run(mt, &loc, values, 8, 12);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
}
