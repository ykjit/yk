// Run-time:
//   env-var: YKD_LOG_IR=-:aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     ...
//     yk-jit-event: enter-jit-code
//     x=2
//     ...

// Check that a call followed immediately by an unconditional branch doesn't
// confuse the block mapping due to fallthrough optimisations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) void f(int x) { fprintf(stderr, "x=%d\n", x); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int x = 0;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(x);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    f(i);
    goto more;

  more:
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
