// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4 -> 1.333200 3.360000
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     %{{10_5}}: float = fmul %{{_}}, %{{_}}
//     %{{10_6}}: double = fp_ext %{{10_5}}, double
//     ...
//     %{{10_9}}: double = fmul %{{_}}, 0.84double
//     ...
//     %{{_}}: i32 = call fprintf(%{{_}}, @{{_}}, %{{_}}, %{{10_6}}, %{{10_9}})
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{9}}: float = fmul %{{_}}, %{{_}}
//     %{{10}}: double = fpext %{{9}}
//     ...
//     %{{12}}: double = 0.84
//     %{{13}}: double = fmul %{{_}}, %{{12}}
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{10}}, %{{13}}) ; @fprintf
//     ...
//     --- End jit-pre-opt ---
//     3 -> 0.999900 2.520000
//     yk-execution: enter-jit-code
//     2 -> 0.666600 1.680000
//     1 -> 0.333300 0.840000
//     yk-execution: deoptimise ...

// Check floating point multiplication works.

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
  float f = 0.3333;
  double d = 0.84;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(f);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d -> %f %f\n", i, (float)i * f, (double)i * d);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
