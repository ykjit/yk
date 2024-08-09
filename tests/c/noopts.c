// Run-time:
//   env-var: YK_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:aot
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     %{{9_1}}: i32 = add %{{9_0}}, 3i32
//     ...
//     --- End aot ---
//     ...

// Check that our NOOPT_VAL macro blocks AOT optimisations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int x, res = 0, i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    x = 2;
    NOOPT_VAL(x);
    res = x + 3; // We don't want constant folding to happen here.
    i--;
  }
  NOOPT_VAL(res);
  assert(res == 5);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
