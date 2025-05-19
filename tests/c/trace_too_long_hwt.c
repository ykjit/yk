// ## This test checks that an overflowing PT buffer is caught at the point
// ## where a trace is stopped, not after trace mapping. It therefore only works
// ## on hwt.
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     yk-warning: stop-tracing-aborted: Trace overflowed recorder's storage

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 10;
  int t = 0;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(t);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (i == 10) {
      int j = 100000000;
      while (j > 0) {
        t++;
        j--;
      }
    }
    NOOPT_VAL(i);
    i--;
  }
  printf("exit");
  NOOPT_VAL(i);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
