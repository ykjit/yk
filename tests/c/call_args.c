// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     3: 5
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{20}}: i32 = add %{{13}}, %{{14}}
//     ...
//     --- End jit-pre-opt ---
//     2: 5
//     yk-execution: enter-jit-code
//     1: 5
//     yk-execution: deoptimise
//     ...

// Check that function calls with arguments do the right thing

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int a, int b) {
  int c = a + b;
  return c;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3, two = 2, three = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(two);
    NOOPT_VAL(three);
    fprintf(stderr, "%d: %d\n", i, f(two, three));
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
