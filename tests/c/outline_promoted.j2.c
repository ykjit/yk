// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stderr:
//     5 6
//     --- Begin jit-pre-opt ---
//       ...
//       %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}) ; @__yk_opt_f
//       ...
//     --- End jit-pre-opt ---
//     4 5
//     3 4
//     2 3
//     exit

// Check that promotes in indirect callees of outlined functions are consumed
// properly during outlining.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline))
int g(int x, int y) {
  return yk_promote(x) + y;
}

__attribute__((yk_outline))
int f(int x, int y) {
  return g(x, y);
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
    int x1 = f(1, i);
    int x2 = g(2, i);
    fprintf(stderr, "%d %d\n", x1, x2);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return (EXIT_SUCCESS);
}
