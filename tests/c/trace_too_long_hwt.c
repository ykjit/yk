// ## This test checks that an overflowing PT buffer is caught at the point
// ## where a trace is stopped, not after trace mapping. It therefore only works
// ## on hwt.
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-warning: stop-tracing-aborted: Trace too long

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();

  int i = 100000;
  NOOPT_VAL(loc1);
  NOOPT_VAL(loc2);
  NOOPT_VAL(i);
  YkLocation *loc = &loc1;
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    if (i == 100000)
      loc = &loc2;
    else if (i == 2)
      loc = &loc1;
    NOOPT_VAL(i);
    i--;
  }
  printf("exit");
  NOOPT_VAL(i);
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
