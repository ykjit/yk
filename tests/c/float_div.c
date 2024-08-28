// ## yk-config-env: YKB_AOT_OPTLEVEL=1
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4 -> 8.000000 20.000000
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     %{{10_5}}: float = fdiv %{{_}}, %{{_}}
//     %{{10_6}}: double = fp_ext %{{10_5}}, double
//     ...
//     %{{10_9}}: double = fdiv %{{_}}, 0.2double
//     ...
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{10_6}}, %{{10_9}})
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{16}}: float = fdiv %{{_}}, %{{_}}
//     %{{17}}: double = fp_ext %{{16}}
//     ...
//     %{{20}}: double = fdiv %{{_}}, 0.2double
//     ...
//     %{{_}}: i32 = call @fprintf(%{{_}}, %{{_}}, %{{_}}, %{{17}}, %{{20}})
//     ...
//     --- End jit-pre-opt ---
//     3 -> 6.000000 15.000000
//     yk-jit-event: enter-jit-code
//     2 -> 4.000000 10.000000
//     1 -> 2.000000 5.000000
//     yk-jit-event: deoptimise

// Check floating point division works.

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
  float f = 0.5;
  double d = 0.2;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(f);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d -> %f %f\n", i, (float)i / f, (double)i / d);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
