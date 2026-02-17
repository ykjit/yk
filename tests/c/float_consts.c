// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4 -> 3.350000 3.350000 4.350000
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{6}}: float = 3.35
//     ...
//     %{{7}}: double = 3.3499999046325684
//     ...
//     %{{8}}: double = 4.349999904632568
//     ...
//     --- End jit-pre-opt ---
//     3 -> 3.350000 3.350000 4.350000
//     yk-execution: enter-jit-code
//     2 -> 3.350000 3.350000 4.350000
//     1 -> 3.350000 3.350000 4.350000
//     yk-execution: deoptimise ...

// Check 32- and 64-bit float constants work properly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline, yk_outline))
double double_id(double x) {
  return x;
}

__attribute__((noinline, yk_outline))
float float_id(float x) {
  return x;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    float x = float_id(3.35);
    double y = double_id(3.35f);
    double z = double_id(4.35f);
    fprintf(stderr, "%d -> %f %f %f\n", i, x, y, z);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
