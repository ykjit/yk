// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     and 0
//     or 5
//     lshr 2
//     ashr 2
//     ashr2 -2
//     xor 5
//     xor2 -5
//     shl 8
//     ---
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{result}}: i32 = and %{{1}}, 1i32
//     ...
//     --- End jit-pre-opt ---
//     and 1
//     or 3
//     lshr 1
//     ashr 1
//     ashr2 -2
//     xor 2
//     xor2 -4
//     shl 6
//     ---
//     yk-jit-event: enter-jit-code
//     and 0
//     or 3
//     lshr 1
//     ashr 1
//     ashr2 -1
//     xor 3
//     xor2 -3
//     shl 4
//     ---
//     and 1
//     or 1
//     lshr 0
//     ashr 0
//     ashr2 -1
//     xor 0
//     xor2 -2
//     shl 2
//     ---
//     yk-jit-event: deoptimise
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
    int ashr = i >> 1;
    int ashr2 = -i >> 1;
    int xor = i ^ 1;
    int xor2 = ~i;
    int shl = i << 1;
    fprintf(stderr, "and %d\n", and);
    fprintf(stderr, "or %d\n", or);
    fprintf(stderr, "lshr %d\n", lshr);
    fprintf(stderr, "ashr %d\n", ashr);
    fprintf(stderr, "ashr2 %d\n", ashr2);
    fprintf(stderr, "xor %d\n", xor);
    fprintf(stderr, "xor2 %d\n", xor2);
    fprintf(stderr, "shl %d\n", shl);
    fprintf(stderr, "---\n");
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
