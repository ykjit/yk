// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     sdiv 10922
//     sdiv2 715827882
//     sdiv3 1431655764
//     sdiv4 *
//     jitstate: stop-tracing
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
//     sdiv 10922
//     sdiv2 715827882
//     sdiv3 1431655764
//     sdiv4 *
//     jitstate: enter-jit-code
//     sdiv 10922
//     sdiv2 715827882
//     sdiv3 1431655764
//     sdiv4 *
//     sdiv 10922
//     sdiv2 715827882
//     sdiv3 1431655764
//     sdiv4 *
//     jitstate: deoptimise
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
  int16_t num1 = 32767;
  int32_t num2 = 2147483647;
  int64_t num3 = 4294967294;
  int8_t num4 = 127;
  NOOPT_VAL(num1);
  NOOPT_VAL(num2);
  NOOPT_VAL(num3);
  NOOPT_VAL(num4);
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    short int sdiv = num1 / 3;
    int sdiv2 = num2 / 3;
    long long sdiv3 = num3 / 3;
    signed char sdiv4 = num4 / 3;
    fprintf(stderr, "sdiv %hd\n", sdiv);
    fprintf(stderr, "sdiv2 %d\n", sdiv2);
    fprintf(stderr, "sdiv3 %lld\n", sdiv3);
    fprintf(stderr, "sdiv4 %c\n", sdiv4);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
