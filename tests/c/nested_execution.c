// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     enter
//     yk-jit-event: start-tracing
//     6
//     enter
//     yk-warning: tracing-aborted: tracing went outside of starting frame
//     5
//     yk-jit-event: start-tracing
//     4
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     3
//     yk-jit-event: enter-jit-code
//     2
//     1
//     yk-jit-event: deoptimise
//     return
//     yk-jit-event: start-tracing
//     5
//     enter
//     yk-warning: tracing-aborted: tracing went outside of starting frame
//     4
//     yk-jit-event: enter-jit-code
//     3
//     2
//     1
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     return
//     yk-warning: tracing-aborted: tracing went outside of starting frame
//     4
//     enter
//     yk-jit-event: enter-jit-code
//     3
//     2
//     1
//     yk-jit-event: deoptimise
//     return
//     yk-jit-event: start-tracing
//     c
//     3
//     enter
//     yk-warning: tracing-aborted: tracing went outside of starting frame
//     2
//     yk-jit-event: enter-jit-code
//     1
//     yk-jit-event: deoptimise
//     return
//     yk-jit-event: start-tracing
//     b
//     2
//     enter
//     yk-warning: tracing-aborted: tracing went outside of starting frame
//     1
//     return
//     yk-jit-event: start-tracing
//     a
//     1
//     enter
//     return
//     return

// Check that recursive execution finds the right guards.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void f(YkMT *mt, int who, YkLocation *loc1, YkLocation *loc2, int i) {
  fprintf(stderr, "enter\n");
  while (i > 0) {
    yk_mt_control_point(mt, loc1);
    if (who) {
      if (i == 1) {
        fprintf(stderr, "a\n");
      }
      if (i == 2) {
        fprintf(stderr, "b\n");
      }
      if (i == 3) {
        fprintf(stderr, "c\n");
      }
    }
    fprintf(stderr, "%d\n", i);
    i -= 1;
    if (loc2 != NULL) {
      f(mt, 0, loc2, NULL, i);
    }
  }
  fprintf(stderr, "return\n");
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 2);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();
  int i = 6;
  NOOPT_VAL(loc1);
  NOOPT_VAL(loc2);
  NOOPT_VAL(i);
  f(mt, 1, &loc1, &loc2, i);
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
