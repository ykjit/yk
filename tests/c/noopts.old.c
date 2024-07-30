// Run-time:
//   env-var: YKD_LOG_JITSTATE=-
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt,jit-post-opt
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     store i32 2...
//     ...
//     ...add...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     store i32 2...
//     ...
//     ...add...
//     ...
//     --- End jit-pre-opt ---
//     --- Begin jit-post-opt ---
//     ...
//     store i32 2...
//     ...
//     ...add...
//     ...
//     --- End jit-post-opt ---
//     ...

// Check that our NOOPT_VAL macro blocks optimisations.

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
    res = x + 3; // We don't want this operation to be optimised away.
    i--;
  }
  NOOPT_VAL(res);
  assert(res == 5);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
