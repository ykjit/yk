// Run-time:
//   env-var: YKD_PRINT_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     jit-state: start-tracing
//     ...
//     jit-state: stop-tracing
//     ...
//     jit-state: enter-jit-code
//     ...
//     jit-state: deoptimise
//  stdout:
//     ...
//     i: 5 ret: 9
//     ...
//     i: 2 ret: 6
//     i: 1 ret: 105

// Simple guard failure test.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int a, int b) {
  if (a > 1) {
    return a + b;
  } else {
    return a + b + 100;
  }
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 5;
  int ret = 0;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    ret = f(i, argc + 3);
    printf("i: %d ret: %d\n", i, ret);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
