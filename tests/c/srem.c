// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     srem 3
//     srem2 3
//     srem3 2
//     srem4 3
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{_}}: i32 = srem %{{_}}, %{{_}}
//     ...
//     %{{_}}: i32 = srem %{{_}}, %{{_}}
//     ...
//     %{{_}}: i64 = srem %{{_}}, %{{_}}
//     ...
//     %{{_}}: i32 = srem %{{_}}, %{{_}}
//     ...
//     --- End jit-pre-opt ---
//     srem 1
//     srem2 1
//     srem3 2
//     srem4 1
//     yk-jit-event: enter-jit-code
//     srem 1
//     srem2 1
//     srem3 0
//     srem4 1
//     srem 0
//     srem2 0
//     srem3 0
//     srem4 0
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

  int16_t i = 4;
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
    int16_t srem = num1 % (int16_t)i;
    int32_t srem2 = num2 % (int32_t)i;
    int64_t srem3 = num3 % (int64_t)i;
    int8_t srem4 = num4 % i;
    fprintf(stderr, "srem %hd\n", srem);
    fprintf(stderr, "srem2 %d\n", srem2);
    fprintf(stderr, "srem3 %ld\n", srem3);
    fprintf(stderr, "srem4 %hhd\n", srem4);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
