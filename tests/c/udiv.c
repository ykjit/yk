// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     udiv 21845
//     udiv2 715827882
//     udiv3 1431655764
//     udiv4 42
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{8}}: i16 = 3
//     %{{_}}: i16 = udiv %{{_}}, %{{8}}
//     ...
//     %{{11}}: i32 = 3
//     %{{_}}: i32 = udiv %{{_}}, %{{11}}
//     ...
//     %{{14}}: i64 = 3
//     %{{_}}: i64 = udiv %{{_}}, %{{14}}
//     ...
//     %{{17}}: i8 = 3
//     %{{_}}: i8 = udiv %{{_}}, %{{17}}
//     ...
//     --- End jit-pre-opt ---
//     udiv 21845
//     udiv2 715827882
//     udiv3 1431655764
//     udiv4 42
//     yk-execution: enter-jit-code
//     udiv 21845
//     udiv2 715827882
//     udiv3 1431655764
//     udiv4 42
//     udiv 21845
//     udiv2 715827882
//     udiv3 1431655764
//     udiv4 42
//     yk-execution: deoptimise ...
//     exit

// Test unsigned division.

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
  uint16_t num1 = UINT16_MAX;
  uint32_t num2 = 2147483647;
  uint64_t num3 = 4294967294;
  uint8_t num4 = 127;
  NOOPT_VAL(num1);
  NOOPT_VAL(num2);
  NOOPT_VAL(num3);
  NOOPT_VAL(num4);
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    uint16_t udiv = num1 / 3;
    uint32_t udiv2 = num2 / 3;
    uint64_t udiv3 = num3 / 3;
    uint8_t udiv4 = num4 / 3;
    fprintf(stderr, "udiv %hd\n", udiv);
    fprintf(stderr, "udiv2 %d\n", udiv2);
    fprintf(stderr, "udiv3 %ld\n", udiv3);
    fprintf(stderr, "udiv4 %u\n", udiv4);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
