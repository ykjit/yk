// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4

// Check that a trace builder correctly tracks the previously seen basic block
// on a per-frame basis.
//
// Checks for regressions of:
// https://github.com/ykjit/yk/pull/1944

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline))
int f(int x) {
  return x + 1;
}

__attribute__((noinline))
int g(int x) {
  return x + 77;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = 0;
  NOOPT_VAL(i);
    if (i % 2 == 0) {
      x = f(x);
    } else {
      x = g(x);
    }
    fprintf(stderr, "%d %d\n", i, x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
