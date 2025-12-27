// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_OPT=0
//   stderr:
//     yk-tracing: start-tracing
//     -4 -4 -4 -4
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{6}}: i8 = ...
//     %{{7}}: i16 = sext %{{6}}
//     ...
//     %{{9}}: i16 = ...
//     %{{10}}: i32 = sext %{{9}}
//     ...
//     %{{12}}: i32 = ...
//     %{{13}}: i64 = sext %{{12}}
//     ...
//     --- End jit-pre-opt ---
//     -3 -3 -3 -3
//     yk-execution: enter-jit-code
//     -2 -2 -2 -2
//     -1 -1 -1 -1
//     yk-execution: deoptimise ...
//     exit

// Test zero extend.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int8_t i = -4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i < 0) {
    yk_mt_control_point(mt, &loc);
    int16_t x = i;
    int32_t y = x;
    int64_t z = y;
    fprintf(stderr, "%d %d %d %ld\n", i, x, y, z);
    i++;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
