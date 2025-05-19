// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=3
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     switch %{{10_1}}, bb{{bb15}}, [299 -> bb{{bb11}}, 298 -> bb{{bb13}}]...
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{21}}: i1 = eq %{{20}}, 299i32
//     %{{22}}: i1 = eq %{{20}}, 298i32
//     %{{23}}: i1 = or %{{21}}, %{{22}}
//     guard false, %{{23}}, ...
//     ...
//     --- End jit-pre-opt ---
//     i=2
//     yk-execution: enter-jit-code
//     i=1
//     yk-execution: deoptimise
//     ...

// Check that tracing the default arm of a switch works correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 3;
  int j = 300;
  NOOPT_VAL(i);
  NOOPT_VAL(j);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    switch (j) {
    case 299:
      exit(1);
    case 298:
      exit(1);
    default:
      i--;
      break;
    }
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);

  return (EXIT_SUCCESS);
}
