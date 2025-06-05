// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     5.100000
//     5.200000
//     yk-tracing: stop-tracing
//     4.100000
//     4.200000
//     yk-execution: enter-jit-code
//     3.100000
//     3.200000
//     2.100000
//     2.200000
//     yk-execution: deoptimise ...

// Check that passing floats to/from a function works correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_outline))
float f_f(float x) {
  return x + 1.1;
}

__attribute__((yk_outline))
double f_d(double x) {
  return x + 1.2;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%f\n", f_f((float) i));
    fprintf(stderr, "%f\n", f_d((double) i));
    i--;
  }
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
