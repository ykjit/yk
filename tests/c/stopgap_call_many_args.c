// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     f: 3:4:5:6:7
//     jit-state: enter-jit-code
//     jit-state: deoptimise
//     jit-state: exit-jit-code
//     f: 2:3:4:5:6
//     jit-state: enter-jit-code
//     jit-state: deoptimise
//     jit-state: exit-jit-code
//     f: 1:2:3:4:5
//     ...

// Check the stop-gap interpreter can call out to AOT-compiled functions.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute((__noinline__)) void f(int a, int b, int c, int d, int e) {
  fprintf(stderr, "f: %d:%d:%d:%d:%d\n", a, b, c, d, e);
}

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
      f(i, i + 1, i + 2, i + 3, i + 4);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
