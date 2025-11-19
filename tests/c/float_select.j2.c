// ignore-if: test "$YK_JITC" != "j2"
// Compiler:
//   env-var: YKB_EXTRA_LD_FLAGS=-lm
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     10: 10.840000
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{12}}: double = sitofp %{{_}}
//     %{{13}}: double = fneg %{{_}}
//     %{{14}}: double = select %{{_}}, %{{12}}, %{{13}}
//     ...
//     --- End jit-pre-opt ---
//     9: -8.160000
//     yk-execution: enter-jit-code
//     8: 8.840000
//     7: -6.160000
//     6: 6.840000
//     5: -4.160000
//     4: 4.840000
//     3: -2.160000
//     2: 2.840000
//     1: -0.160000
//     yk-execution: deoptimise ...
//     exit

// Check floating point `select` works.

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

  int i = 10;
  double d = 0.84;
  NOOPT_VAL(loc);
  NOOPT_VAL(d);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %f\n", i, i % 2 == 0 ? d + i : d - i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return (EXIT_SUCCESS);
}
