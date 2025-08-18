// ## longjmp detection with CFI breaks this.
// ignore-if: true
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     enter
//     yk-tracing: start-tracing
//     6
//     enter
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     call @f...
//     deopt...
//     --- End jit-pre-opt ---
//     5
//     yk-tracing: start-tracing
//     4
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     header_end...
//     --- End jit-pre-opt ---
//     3
//     yk-execution: enter-jit-code
//     2
//     1
//     yk-execution: deoptimise ...
//     return
//     yk-execution: enter-jit-code
//     5
//     enter
//     yk-execution: enter-jit-code
//     4
//     3
//     2
//     1
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     return
//     yk-warning: tracing-aborted: tracing continued into a JIT frame
//     yk-execution: deoptimise ...
//     yk-execution: enter-jit-code
//     4
//     enter
//     yk-execution: enter-jit-code
//     3
//     2
//     1
//     yk-execution: deoptimise ...
//     return
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     yk-tracing: stop-tracing
//     ...
//     c
//     3
//     enter
//     yk-execution: enter-jit-code
//     2
//     1
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     return
//     yk-warning: tracing-aborted: tracing went outside of starting frame
//     b
//     2
//     enter
//     yk-execution: enter-jit-code
//     1
//     yk-execution: deoptimise ...
//     return
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     a
//     1
//     enter
//     return
//     return

// Check that recursive execution finds the right guards.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void f(YkMT *mt, int who, YkLocation *loc1, YkLocation *loc2, int i) {
  fprintf(stderr, "enter\n");
  while (i > 0) {
    yk_mt_control_point(mt, loc1);
    if (who) {
      if (i == 1) {
        fprintf(stderr, "a\n");
      }
      if (i == 2) {
        fprintf(stderr, "b\n");
      }
      if (i == 3) {
        fprintf(stderr, "c\n");
      }
    }
    fprintf(stderr, "%d\n", i);
    i -= 1;
    if (loc2 != NULL) {
      f(mt, 0, loc2, NULL, i);
    }
  }
  fprintf(stderr, "return\n");
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 2);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();
  int i = 6;
  NOOPT_VAL(loc1);
  NOOPT_VAL(loc2);
  NOOPT_VAL(i);
  f(mt, 1, &loc1, &loc2, i);
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
