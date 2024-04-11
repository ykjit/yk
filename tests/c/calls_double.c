// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     jitstate: enter-jit-code
//     res=3
//     ...

// Check that calling the same function in sequence (but with different
// arguments) works.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int a) { return a; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int a = f(1);
    int b = f(2);
    int res = a + b;
    fprintf(stderr, "res=%d\n", res);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
