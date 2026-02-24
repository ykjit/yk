// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_OPT=0
//   stderr:
//     yk-tracing: start-tracing
//     i=4, y=7
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{9_4}}: ptr = ptr_add @line, 0 + (%{{9_3}} * 8)
//     %{{9_5}}: ptr = ptr_add %{{9_4}}, 4
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{14}}: ptr = dynptradd %{{_}}, %{{_}}, 8
//     %{{_}}: ptr = ptradd %{{14}}, 4
//     ...
//     --- End jit-pre-opt ---
//     i=3, y=1
//     yk-execution: enter-jit-code
//     i=2, y=3
//     i=1, y=4
//     yk-execution: deoptimise ...

// Check dynamic ptradd instructions work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

struct point {
  uint32_t x;
  uint32_t y;
};

struct point line[] = {
    {5, 1}, {3, 4}, {4, 3}, {1, 1}, {0, 7},
};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d, y=%d\n", i, line[i].y);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
