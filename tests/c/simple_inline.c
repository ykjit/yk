// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=5
//   stderr:
//     yk-tracing: start-tracing
//     foo 7
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{result}}: i32 = add %{{1}}, 3i32
//     ...
//     %{{2}}: i32 = call @fprintf(%{{3}}, %{{4}}, %{{result}})
//     ...
//     --- End jit-pre-opt ---
//     foo 6
//     yk-execution: enter-jit-code
//     foo 5
//     foo 4
//     yk-execution: deoptimise
//     exit

// Check that return values of fucntions inlined into the trace are properly
// mapped.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int i) { return i + 3; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = foo(i);
    fprintf(stderr, "foo %d\n", x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
