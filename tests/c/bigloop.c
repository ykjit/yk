// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr: ...

// Benchmark a large loop to test trace compilation efficiency.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  // yk_mt_hot_threshold_set(mt, 1000000); // Set a higher threshold for benchmarking
  YkLocation loc = yk_location_new();
  char *endptr;
  long iterations = strtol(argv[1], &endptr, 10);
  NOOPT_VAL(loc)
  NOOPT_VAL(iterations);
  int sum = 0;
  while (sum < iterations) {
    yk_mt_control_point(mt, &loc);
    while (sum < iterations) {
      sum += 1;
    }
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}