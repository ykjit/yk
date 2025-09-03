// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4
//     don't deopt before this
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     3
//     don't deopt before this
//     yk-execution: enter-jit-code
//     2
//     don't deopt before this
//     1
//     don't deopt before this
//     yk-execution: deoptimise ...
//     exit

// Tests that switch statements where multiple cases all map to the same block,
// don't result in unnecessary guard failures.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>


int switcheroo(int encindex)
{
    switch (encindex) {
      case 0:
      case 1:
      case 2:
        return 1;
      default:
        return 0;
    }
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d\n", i);
    switcheroo(1);
    fprintf(stderr, "don't deopt before this\n");
    i--;
  }
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
