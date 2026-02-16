// Compiler:
//   env-var: YKB_EXTRA_LD_FLAGS=-lm
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 4.840000 -> 4.000000
//     4: -3.160000 -> -4.000000
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{15}}: double = floor %{{14}}
//     ...
//     --- End jit-pre-opt ---
//     3: 3.840000 -> 3.000000
//     3: -2.160000 -> -3.000000
//     yk-execution: enter-jit-code
//     2: 2.840000 -> 2.000000
//     2: -1.160000 -> -2.000000
//     1: 1.840000 -> 1.000000
//     1: -0.160000 -> -1.000000
//     yk-execution: deoptimise ...

// Check floating point `floor` works.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  double d = 0.84;
  NOOPT_VAL(loc);
  NOOPT_VAL(d);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %f -> %f\n", i, d + i, floor(d + i));
    fprintf(stderr, "%d: %f -> %f\n", i, d - i, floor(d - i));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
