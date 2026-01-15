// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     early return
//     6
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     term []
//     ; guard 0
//     ...
//     --- End jit-pre-opt ---
//     ...
//     return
//     exit

// Used to check that early return from a recursive interpreter loop aborts
// tracing, but doesn't stop a location being retraced. Now, simply checks that
// we can trace and compile an early return.

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
    if (i == 7) {
      loop(mt, loc, i - 1);
      i--;
    } else if (i == 6) {
      fprintf(stderr, "early return\n");
      return;
    }
    fprintf(stderr, "%d\n", i);
    i--;
  }
  fprintf(stderr, "return\n");
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  NOOPT_VAL(loc);
  loop(mt, &loc, 7);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
