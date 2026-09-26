// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     3: 204
//     yk-tracing: stop-tracing
//     2: 204
//     yk-execution: enter-jit-code {"trid": "0"}
//     1: 204
//     yk-execution: deoptimise {"trid": "0", "gidx": "1"}

// Check that call arguments beyond the 6 integer argument registers are passed
// correctly on the stack.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((yk_outline)) long f(long a, long b, long c, long d, long e,
                                   long f, long g, long h) {
  return a * 1 + b * 2 + c * 3 + d * 4 + e * 5 + f * 6 + g * 7 + h * 8;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  long one = 1;
  int i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(one);
    fprintf(stderr, "%d: %ld\n", i, f(one, 2, 3, 4, 5, 6, 7, 8));
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
