
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     b3
//     8
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     b1
//     7
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     b1
//     9
//     yk-jit-event: tracing-aborted
//     b3
//     8
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     b3
//     7
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     b3
//     6
//     yk-jit-event: enter-jit-code
//     yk-jit-event: execute-side-trace
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     b2
//     5
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     b2
//     4
//     yk-jit-event: enter-jit-code
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     b2
//     3
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     b2
//     2
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     b2
//     1
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     b2
//     0
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     exit

// Check that early return from a recursive interpreter loop aborts tracing,
// but doesn't stop a location being retraced.

#include <assert.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *, int, bool);

void loop(YkMT *mt, YkLocation *loc, int i, bool inner) {
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    if (i == 10) {
      loop(mt, loc, i - 1, true);
      i--;
    } else if (inner && i <= 8) {
      fprintf(stderr, "b1\n");
      i--;
    } else if (!inner && i <= 6) {
      fprintf(stderr, "b2\n");
      i--;
    } else {
      fprintf(stderr, "b3\n");
      i--;
    }
    if (inner && i == 6) {
      return;
    }
    fprintf(stderr, "%d\n", i);
  }
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  NOOPT_VAL(loc);
  loop(mt, &loc, 10, false);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
