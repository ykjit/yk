// ignore: optimisation levels >0 need more intrinsics support.
// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     jit-state: start-tracing
//     4:0
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %45 = call i32 @f(...
//     ...
//     --- End jit-pre-opt ---
//     3:0
//     jit-state: enter-jit-code
//     2:0
//     jit-state: exit-jit-code
//     jit-state: enter-jit-code
//     1:0
//     jit-state: stopgap
//     ...

// Check that recursive function calls are not unrolled.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int num) {
  if (num == 0)
    return 0;
  else
    return f(num - 1);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new();
  yk_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_control_point(mt, &loc);
    int x = 3;
    NOOPT_VAL(x);
    fprintf(stderr, "%d:%d\n", i, f(x));
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
