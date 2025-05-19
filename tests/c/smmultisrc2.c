// ## not yet implemented: ran out of gp regs
// ignore-if: true
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stdout:
//     1 2 3 4 5 6
//     1 2 3 4 5 6
//     1 2 3 4 5 6
//     1 2 3 4 5 6
//     1 2 3 4 5 6
//     1 2 3 4 5 6
//     exit

// Another test case extracted from the lua source that showed that stackmaps
// need to track multiple locations for the same value or deoptimisation won't
// work.
// The output of this test isn't really relevant. We mostly want to know this
// doesn't segfault.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int lua_geti(void *L, int x, int y) {
  int a = x;
  int b = y;
  int c = 3;
  int d = 4;
  int e = 5;
  int f = 6;
  fprintf(stderr, "geti: %d %d %d %d %d %d\n", a, b, c, d, e, f);
  return a;
}

__attribute__((yk_unroll_safe)) int tunpack(void *L, int argc) {
  unsigned n = 0;
  int a = 1;
  int b = 2;
  int c = 3;
  int d = 4;
  int f = 5;
  int g = 6;
  int i = 0;
  int e = argc;
  for (; i < e; i++) {
    lua_geti(L, i, argc);
  }
  lua_geti(L, 1, e);
  printf("%d %d %d %d %d %d\n", a, b, c, d, f, g);
  return (int)n;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 5;
  void *a = malloc(8);
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i >= 0) {
    yk_mt_control_point(mt, &loc);
    tunpack(a, i);
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
