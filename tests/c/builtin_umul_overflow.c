// ignore-if: test "$YK_JITC" = "j2" # not yet implemented in j2
// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     1 0
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{19}}: {0: i32, 32: i1} = call @llvm.umul.with.overflow.i32...
//     ...
//     --- End jit-pre-opt ---
//     1 0
//     yk-execution: enter-jit-code
//     1 0
//     1 0
//     yk-execution: deoptimise ...
//     exit

// Test that we can deal with GEPOperands as in
// `load ptr getelementptr inbounds ([34 x i16], ptr @a, i64 0, i64 1)`

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

const int a[5] = {10, 11, 12, 13, 14};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    uint64_t a = (uint64_t)argc*2;
    uint64_t b = 2147483648ULL;
    unsigned int res;
    bool overflow = __builtin_umul_overflow(a, b, &res);
    fprintf(stderr, "%d %d\n", overflow, res);
    i--;
  }
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
