// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-tracing: start-tracing
//     6
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//       ...
//       header_end ...
//     --- End jit-pre-opt ---
//     5
//     yk-tracing: start-tracing
//     4
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//       ...
//       connector ...
//     --- End jit-pre-opt ---
//     3
//     yk-execution: enter-jit-code
//     2
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//       ...
//       sidetrace_end ...
//     --- End jit-pre-opt ---
//     1
//     exit

// Test coupler traces.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 0);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();

  int i = 6;
  NOOPT_VAL(loc1);
  NOOPT_VAL(loc2);
  NOOPT_VAL(i);
  while (i > 0) {
    YkLocation *loc;
    if (i > 4 || i == 3)
      loc = &loc1;
    else
      loc = &loc2;
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    i--;
  }
  fprintf(stderr, "exit");
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
