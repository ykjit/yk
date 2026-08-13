// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4 -> 0.840000
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     %{{_}}: double = copysign ...
//     ...
//     --- End hir ---
//     3 -> -0.840000
//     yk-execution: enter-jit-code {"trid": "0"}
//     2 -> 0.840000
//     1 -> -0.840000
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}


// Check floating point multiplication works.

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
  double c = 0.84;
  double d = -7;
  NOOPT_VAL(loc);
  NOOPT_VAL(c);
  NOOPT_VAL(d);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    d = -d;
    fprintf(stderr, "%d -> %f\n", i, copysign(c, d));
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
