// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jit-state: start-tracing
//     3:5
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2:5
//     jit-state: enter-jit-code
//     1:5
//     jit-state: deoptimise
//     ...

// Check that using an argument (of a non-main() function) in a trace works.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int f(int x) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d:%d\n", i, x);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return 0;
}

int main(int argc, char **argv) {
  f(5);
  return (EXIT_SUCCESS);
}
