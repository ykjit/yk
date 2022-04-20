// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     f: 3
//     jit-state: enter-jit-code
//     ...
//     jit-state: enter-stopgap
//     ...
//     f: 2
//     jit-state: exit-stopgap
//     ...

// Check the stop-gap interpreter can call out to AOT-compiled functions.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute((__noinline__)) void f(int i) { fprintf(stderr, "f: %d\n", i); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new();
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (i == 4)
      fprintf(stderr, "main: %d\n", i);
    else
      f(i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
