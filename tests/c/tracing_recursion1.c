// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YK_LOG=4
//   stderr:
//     6
//     yk-jit-event: start-tracing
//     5
//     yk-jit-event: tracing-aborted
//     ...

// Check that traces that end in a different frame than they started in are
// rejected.

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
