// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     4 -> 3.252033
//     4 -> 3.252033
//     4 -> 3.252033
//     4 -> 3.252033
//     4 -> 3.252033
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     %{{12_1}}: float = si_to_fp %{{_}}, float
//     %{{_}}: float = fdiv %{{12_1}}, 1.23float
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{10}}: float = si_to_fp %{{_}}
//     %{{_}}: float = fdiv %{{10}}, 1.23float
//     ...
//     --- End jit-pre-opt ---
//     3 -> 2.439024
//     3 -> 2.439024
//     3 -> 2.439024
//     3 -> 2.439024
//     3 -> 2.439024
//     yk-jit-event: enter-jit-code
//     2 -> 1.626016
//     2 -> 1.626016
//     2 -> 1.626016
//     2 -> 1.626016
//     2 -> 1.626016
//     1 -> 0.813008
//     1 -> 0.813008
//     1 -> 0.813008
//     1 -> 0.813008
//     1 -> 0.813008
//     yk-jit-event: deoptimise

// Check basic 32-bit float support.

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
  float f1;
  float *f2 = &f1;
  float *f3 = &f1;
  double d1;
  double *d2 = &d1;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(f1);
  NOOPT_VAL(f2);
  NOOPT_VAL(f3);
  NOOPT_VAL(d1);
  NOOPT_VAL(d2);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    float f = i / (float)1.23;
    fprintf(stderr, "%d -> %f\n", i, f);
    *f2 = f;
    fprintf(stderr, "%d -> %f\n", i, f1);
    fprintf(stderr, "%d -> %f\n", i, *f3);

    double d = i / (double)1.23;
    fprintf(stderr, "%d -> %f\n", i, f);
    *d2 = d;
    fprintf(stderr, "%d -> %f\n", i, d1);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
