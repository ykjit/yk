// Run-time:
//   env-var: YKD_LOG_IR=-:jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     3
//     2
//     yk-jit-event: start-tracing
//     1
//     yk-jit-event: stop-tracing-early-return
//     return
//     yk-jit-event: start-tracing
//     3
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     exit

// Check that basic trace compilation works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *);

void loop(YkMT *mt, YkLocation *loc) {
  int res = 9998;
  int i = 3;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    i--;
  }
  yk_mt_early_return(mt);
  fprintf(stderr, "return\n");
  NOOPT_VAL(res);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 2);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  loop(mt, &loc);
  loop(mt, &loc);
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
