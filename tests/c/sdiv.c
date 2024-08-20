// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     sdiv1 -10922
//     sdiv2 -715827882
//     sdiv3 -3074457345618258602
//     sdiv4 -42
//     sdiv5 -10922
//     sdiv6 -715827882
//     sdiv7 -3074457345618258602
//     sdiv8 -42
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i16 = sdiv %{{_}}, 3i16
//     ...
//     %{{_}}: i32 = sdiv %{{_}}, 3i32
//     ...
//     %{{_}}: i64 = sdiv %{{_}}, 3i64
//     ...
//     %{{_}}: i8 = sdiv %{{_}}, 3i8
//     ...
//     --- End jit-pre-opt ---
//     sdiv1 -10922
//     sdiv2 -715827882
//     sdiv3 -3074457345618258602
//     sdiv4 -42
//     sdiv5 -10922
//     sdiv6 -715827882
//     sdiv7 -3074457345618258602
//     sdiv8 -42
//     yk-jit-event: enter-jit-code
//     sdiv1 -10922
//     sdiv2 -715827882
//     sdiv3 -3074457345618258602
//     sdiv4 -42
//     sdiv5 -10922
//     sdiv6 -715827882
//     sdiv7 -3074457345618258602
//     sdiv8 -42
//     sdiv1 -10922
//     sdiv2 -715827882
//     sdiv3 -3074457345618258602
//     sdiv4 -42
//     sdiv5 -10922
//     sdiv6 -715827882
//     sdiv7 -3074457345618258602
//     sdiv8 -42
//     yk-jit-event: deoptimise
//     exit

// Test signed division.

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
  int16_t num1 = INT16_MIN;
  int32_t num2 = INT32_MIN;
  int64_t num3 = INT64_MIN;
  int8_t num4 = INT8_MIN;
  int16_t num5 = INT16_MAX;
  int32_t num6 = INT32_MAX;
  int64_t num7 = INT64_MAX;
  int8_t num8 = INT8_MAX;
  NOOPT_VAL(num1);
  NOOPT_VAL(num2);
  NOOPT_VAL(num3);
  NOOPT_VAL(num4);
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int16_t sdiv = num1 / 3;
    int32_t sdiv2 = num2 / 3;
    int64_t sdiv3 = num3 / 3;
    int8_t sdiv4 = num4 / 3;
    fprintf(stderr, "sdiv1 %hd\n", sdiv);
    fprintf(stderr, "sdiv2 %d\n", sdiv2);
    fprintf(stderr, "sdiv3 %ld\n", sdiv3);
    fprintf(stderr, "sdiv4 %d\n", sdiv4);
    int16_t sdiv5 = num5 / -3;
    int32_t sdiv6 = num6 / -3;
    int64_t sdiv7 = num7 / -3;
    int8_t sdiv8 = num8 / -3;
    fprintf(stderr, "sdiv5 %hd\n", sdiv5);
    fprintf(stderr, "sdiv6 %d\n", sdiv6);
    fprintf(stderr, "sdiv7 %ld\n", sdiv7);
    fprintf(stderr, "sdiv8 %d\n", sdiv8);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
