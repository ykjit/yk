// Compiler:
//   env-var: YKB_EXTRA_LD_FLAGS=-lm
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: x
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//       ...
//       %{{5}}: i8 = 120
//       ...
//       memset %{{_}}, %{{5}}, %{{_}}, false
//       ...
//     --- End jit-pre-opt ---
//     3: x
//     yk-execution: enter-jit-code
//     2: x
//     1: x
//     yk-execution: deoptimise ...
//     exit

// Check `memset` works.

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
  char *c = malloc(1024);
  NOOPT_VAL(loc);
  NOOPT_VAL(c);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    memset(c, 'x', 1024);
    fprintf(stderr, "%d: %c\n", i, c[i]);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit");
  return (EXIT_SUCCESS);
}
