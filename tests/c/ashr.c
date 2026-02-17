// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0 -Xclang -disable-O0-optnone -Xlinker --lto-newpm-passes=instcombine<max-iterations=1;no-use-loop-info;no-verify-fixpoint>
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: -134217720 2047
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{9}}: i32 = ashr %{{8}}, %{{7}}
//     ...
//     --- End jit-pre-opt ---
//     3: -268435440 4095
//     yk-execution: enter-jit-code
//     2: -536870880 8191
//     1: -1073741760 16383
//     yk-execution: deoptimise ...
//     exit

// Test ashr instructions.

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int32_t x = 0x80000081;
  int16_t y = 0x7fff;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(x);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d: %" PRId32 " %" PRId16 "\n", i, x >> i, y >> i);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
