// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=1
//   stderr:
//     ...
//     jit-state: start-tracing
//     3:1
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2:1
//     jit-state: enter-jit-code
//     1:1
//     jit-state: deoptimise
//     ...

// Check that we can handle struct field accesses.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

struct s {
  int x;
};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  struct s s1 = {argc};
  int y1 = 0, i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(s1);
    y1 = s1.x;
    fprintf(stderr, "%d:%d\n", i, s1.x);
    i--;
  }
  assert(y1 == 1);

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
