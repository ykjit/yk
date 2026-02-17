// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stderr:
//     10: 2
//     --- Begin jit-pre-opt ---
//     ...
//     %{{4}}: i32 = load %{{_}}
//     %{{5}}: i32 = ctpop %{{4}}
//     ...
//     --- End jit-pre-opt ---
//     9: 2
//     8: 1
//     7: 3
//     6: 2
//     5: 2
//     4: 1
//     3: 2
//     2: 1
//     1: 1
//     exit

// Check ctpop

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

  int i = 10;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = __builtin_popcount(i);
    fprintf(stderr, "%d: %d\n", i, x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
