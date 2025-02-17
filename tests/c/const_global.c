// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     i=4
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{12}}: i32 = add %{{11}}, 2i32
//     ...
//     --- End jit-pre-opt ---
//     i=3
//     yk-jit-event: enter-jit-code
//     i=2
//     i=1
//     yk-jit-event: deoptimise
//     ...

// Check that using a global constant works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

const int add = 2;

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  int res = 0;
  YkLocation loc = yk_location_new();
  int i = 4;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    res += add;
    i--;
  }
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);

  return (EXIT_SUCCESS);
}
