// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=4
//     f32->int: 5, 4, -1
//     f64->int: 2, 3, -1
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     %{{_}}: i32 = fp_to_si %{{_}}, i32
//     ...
//     %{{_}}: i32 = fp_to_si %{{_}}, i32
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i32 = fptosi %{{_}}
//     ...
//     %{{_}}: i32 = fptosi %{{_}}
//     ...
//     --- End jit-pre-opt ---
//     i=3
//     f32->int: 5, 4, -1
//     f64->int: 2, 3, -1
//     yk-execution: enter-jit-code
//     i=2
//     f32->int: 5, 4, -1
//     f64->int: 2, 3, -1
//     i=1
//     f32->int: 5, 4, -1
//     f64->int: 2, 3, -1
//     yk-execution: deoptimise ...

// Check float to signed integer conversions.

#include <assert.h>
#include <inttypes.h>
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
  float f1 = 5.78, f2 = 4.55, f3 = -1.01;
  double d1 = 2.1, d2 = 3.3, d3 = -1.99;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(f1);
  NOOPT_VAL(f2);
  NOOPT_VAL(f3);
  NOOPT_VAL(d1);
  NOOPT_VAL(d2);
  NOOPT_VAL(d3);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    fprintf(stderr, "f32->int: %d, %d, %d\n", (int)f1, (int)f2, (int)f3);
    fprintf(stderr, "f64->int: %d, %d, %d\n", (int)d1, (int)d2, (int)d3);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
