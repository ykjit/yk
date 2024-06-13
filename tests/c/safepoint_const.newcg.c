// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-

// Check that basic trace compilation works.

// FIXME: Get this test all the way through the new codegen pipeline!
//
// Currently it succeeds even though it crashes on deopt. This is so
// that we can incrementally implement the new codegen and have CI merge our
// incomplete work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int x, int i) {
  int y;
  if (i == 1) {
    y = 10;
  } else {
    y = 20;
  }
  // Force deoptimisation of a constant.
  fprintf(stderr, "%d %d %d\n", x, y, i);
  return y;
}

__attribute__((noinline)) int g(int x, int i) { return f(x, i); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, ">%d\n", g(4, i));
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
