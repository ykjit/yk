// Run-time:
//   env-var: YK_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     yk-jit-event: enter-jit-code
//     res=2
//     ...

// Check that conditional checks work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int cond = 1, i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int res = 0;
    NOOPT_VAL(cond);
    if (cond) {
      res = 2;
    } else {
      res = 4;
    }
    assert(res == 2);
    fprintf(stderr, "res=%d\n", res);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
