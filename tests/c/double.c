// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4 -> 4.000000
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     %{{9_3}}: double = si_to_fp %{{9_2}}, double
//     ...
//     %{{9_7}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{9_2}}, %{{9_3}})
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{6}}: double = sitofp %{{_}}
//     ...
//     %{{_}}: i32 = call %{{_W}}(%{{_}}, %{{_}}, %{{_}}, %{{6}}) ; @fprintf
//     ...
//     --- End jit-pre-opt ---
//     3 -> 3.000000
//     yk-execution: enter-jit-code
//     2 -> 2.000000
//     1 -> 1.000000
//     yk-execution: deoptimise ...

// Check basic 64-bit float (double) support.

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

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d -> %f\n", i, (double)i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
