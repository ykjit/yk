// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     sdiv1 -10922
//     sdiv2 -715827882
//     sdiv3 -3074457345618258602
//     sdiv4 -42
//     sdiv5 -10922
//     sdiv6 -715827882
//     sdiv7 -3074457345618258602
//     sdiv8 -42
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{12}}: i16 = 3
//     %{{_}}: i16 = sdiv %{{_}}, %{{12}}
//     ...
//     %{{15}}: i32 = 3
//     %{{_}}: i32 = sdiv %{{_}}, %{{15}}
//     ...
//     %{{18}}: i64 = 3
//     %{{_}}: i64 = sdiv %{{_}}, %{{18}}
//     ...
//     %{{21}}: i8 = 3
//     %{{_}}: i8 = sdiv %{{_}}, %{{21}}
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
//     yk-execution: enter-jit-code
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
//     yk-execution: deoptimise ...
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
  NOOPT_VAL(num5);
  NOOPT_VAL(num6);
  NOOPT_VAL(num7);
  NOOPT_VAL(num8);
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int16_t sdiv = num1 / 3;
    int32_t sdiv2 = num2 / 3;
    int64_t sdiv3 = num3 / 3;
    int8_t sdiv4 = num4 / 3;
    fprintf(stderr, "sdiv1 %hd\nsdiv2 %d\nsdiv3 %ld\nsdiv4 %d\n", sdiv, sdiv2,
            sdiv3, sdiv4);
    int16_t sdiv5 = num5 / -3;
    int32_t sdiv6 = num6 / -3;
    int64_t sdiv7 = num7 / -3;
    int8_t sdiv8 = num8 / -3;
    fprintf(stderr, "sdiv5 %hd\nsdiv6 %d\nsdiv7 %ld\nsdiv8 %d\n", sdiv5, sdiv6,
            sdiv7, sdiv8);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
