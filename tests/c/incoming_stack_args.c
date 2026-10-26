// ignore-if: test ${YK_ARCH} != "x86_64"
// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     1 2 3 4 5 6 7 8
//     yk-tracing: stop-tracing
//     1 2 3 4 5 6 7 8
//     yk-execution: enter-jit-code {"trid": "0"}
//     1 2 3 4 5 6 7 8
//     1 2 3 4 5 6 7 8
//     yk-execution: deoptimise {"trid": "0", "gidx": ...
//     total: 144

// Stack arguments are platform specific, so this is an x64 only-test that yk
// and ykllvm deal correctly with incoming stack arguments on the stack (i.e.
// doesn't corrupt them).

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

volatile int inputs[] = {1, 2, 3, 4, 5, 6, 7, 8};

__attribute__((noinline)) static int loop(int a, int b, int c, int d, int e,
                                          int f, int g, int h) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 4;
  NOOPT_VAL(i);
  int total = 0;
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d %d %d %d %d %d %d %d\n", a, b, c, d, e, f, g, h);
    total += a + b + c + d + e + f + g + h;
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return total;
}

int main(void) {
  int total = loop(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
              inputs[6], inputs[7]);
  fprintf(stderr, "total: %d\n", total);
  return EXIT_SUCCESS;
}
