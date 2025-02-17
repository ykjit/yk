// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     enter
//     yk-jit-event: start-tracing
//     1
//     exit
//     enter
//     yk-jit-event: stop-tracing
//     yk-warning: trace-compilation-aborted: returned from function that started tracing
//     ...

// Check that returning from the function that started tracing, then
// re-entering it and stopping tracing, causes the trace to be aborted.
//
// This is an interesting case because the frame address of the place we start
// and stop tracing is the same, so mt.rs cannot catch this.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>
#include <stdbool.h>

void f(YkMT *mt, YkLocation *loc, int i) {
  fprintf(stderr, "enter\n");
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    i -= 1;
  }
  fprintf(stderr, "exit\n");
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 1;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  f(mt, &loc, i);
  f(mt, &loc, i);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
