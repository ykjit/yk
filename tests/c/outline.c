// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=4, r=10
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     #[yk_outline]
//     func foo(%arg0: i32) -> i32;
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{3}}: i32 = call %{{_}}(%{{4}}) ; @__yk_opt_foo
//     ...
//     --- End jit-pre-opt ---
//     i=3, r=6
//     yk-execution: enter-jit-code
//     i=2, r=3
//     i=1, r=1
//     yk-execution: deoptimise ...
//     0
//     exit

// Check that the yk_outline attribute works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int foo(int i) {
  int sum = 0;
  // the loop ensures this function is outlined.
  while (i > 0) {
    sum = sum + i;
    i--;
  }
  return sum;
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
    int r = foo(i);
    fprintf(stderr, "i=%d, r=%d\n", i, r);
    res += 2;
    i--;
  }
  fprintf(stderr, "%d\n", i);
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
