// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     and 0
//     or 5
//     lshr 2
//     jitstate: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{result}}: i32 = And %{{1}}, 1i32
//     ...
//     --- End jit-pre-opt ---
//     and 1
//     or 3
//     lshr 1
//     jitstate: enter-jit-code
//     and 0
//     or 3
//     lshr 1
//     and 1
//     or 1
//     lshr 0
//     jitstate: deoptimise
//     exit

// Test some binary operations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int i) { return i + 3; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int and = i & 1;
    int or = i | 1;
    int lshr = (uint)i >> 1;
    fprintf(stderr, "and %d\n", and);
    fprintf(stderr, "or %d\n", or);
    fprintf(stderr, "lshr %d\n", lshr);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
