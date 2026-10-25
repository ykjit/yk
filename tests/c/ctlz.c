// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   stderr:
//     10: 28
//     --- Begin hir ---
//     ...
//     %{{4}}: i32 = load %{{_}}
//     %{{5}}: i32 = ctlz %{{4}}
//     ...
//     --- End hir ---
//     9: 28
//     8: 28
//     7: 29
//     6: 29
//     5: 29
//     4: 29
//     3: 30
//     2: 30
//     1: 31
//     exit

// Check ctlz works.

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
    int x = __builtin_clz(i);
    fprintf(stderr, "%d: %d\n", i, x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
