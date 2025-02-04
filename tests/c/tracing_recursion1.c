// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YK_LOG=4
//   stderr:
//     6
//     yk-jit-event: start-tracing
//     5
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     4
//     yk-jit-event: enter-jit-code
//     3
//     yk-jit-event: deoptimise
//     yk-jit-event: enter-jit-code
//     2
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     1
//     return
//     yk-jit-event: enter-jit-code
//     4
//     yk-jit-event: execute-side-trace
//     3
//     yk-jit-event: execute-side-trace
//     2
//     yk-jit-event: execute-side-trace
//     1
//     yk-jit-event: execute-side-trace
//     yk-jit-event: deoptimise
//     return
//     exit

// Check that recursive tracing doesn't hit an error.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *, int);

void loop(YkMT *mt, YkLocation *loc, int i) {
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    if (i == 5)
      loop(mt, loc, i - 1);
    i--;
  }
  fprintf(stderr, "return\n");
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 2);
  YkLocation loc = yk_location_new();

  loop(mt, &loc, 6);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
