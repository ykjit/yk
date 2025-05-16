// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     4: 10 10
//     3: -10 10
//     2: 10 10
//     1: -10 10
//     exit

// Check computing absolute values works.

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  long long j = 10;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(j);
    fprintf(stderr, "%d: %lld %lld\n", i, j, llabs(j));
    i--;
    j = -j;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
