// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     ...
//     yk-jit-event: enter-jit-code
//     p1==p2: 1, p2==p3: 0
//     p1==p2: 1, p2==p3: 0
//     yk-jit-event: deoptimise
//     ...
//

// Check that pointer comparisons work.

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

  int i = 0, j = 0;
  int *p1 = &i, *p2 = &i, *p3 = &j;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(j);
  NOOPT_VAL(p1);
  NOOPT_VAL(p2);
  NOOPT_VAL(p3);
  while (i < 4) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "p1==p2: %d, p2==p3: %d\n", p1 == p2, p2 == p3);
    i++;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
