// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     add 5
//     sub 3
//     mul 12
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i32 = add %{{2}}, %{{argc}}
//     ...
//     %{{3}}: i32 = sub %{{4}}, %{{argc}}
//     ...
//     --- End jit-pre-opt ---
//     ...
//     add 4
//     sub 2
//     mul 9
//     yk-execution: enter-jit-code
//     add 3
//     sub 1
//     mul 6
//     add 2
//     sub 0
//     mul 3
//     yk-execution: deoptimise
//     exit

// Test some binary operations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int add = i + argc;
    int sub = i - argc;
    int mul = i * argc * 3;
    fprintf(stderr, "add %d\n", add);
    fprintf(stderr, "sub %d\n", sub);
    fprintf(stderr, "mul %d\n", mul);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
