// Compiler:
// Run-time:
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     yk-jit-event: start-tracing
//     3: 47
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2: 47
//     yk-jit-event: enter-jit-code
//     1: 47
//     yk-jit-event: deoptimise
//     ...

// Check that tracing a cascading "if...else if...else" works.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int x) {
  if (x == 0)
    return 30;
  else if (x == 1)
    return 47;
  else
    return 52;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3, x = 1;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(x);
    fprintf(stderr, "%d: %d\n", i, f(x));
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
