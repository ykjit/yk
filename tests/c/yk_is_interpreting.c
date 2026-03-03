// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     in interpreter
//     5
//     yk-tracing: start-tracing
//     4
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %8_0: i1 = call yk_is_interpreting()
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     guard ...
//     term ...
//     ; peel
//     ...
//     --- End jit-pre-opt ---
//     in interpreter
//     3
//     yk-execution: enter-jit-code
//     2
//     1
//     yk-execution: deoptimise ...

// Check that yk_is_interpreting executes code in the interpreter but not
// during tracing or in traced code. The jit-pre-opt above checks that the call
// has been entirely optimised away from the trace: there is only one guard in
// the body of the trace, which is the loop condition.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();


  int i = 5;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (yk_is_interpreting()) {
      fprintf(stderr, "in interpreter\n");
    }
    fprintf(stderr, "%d\n", i);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
