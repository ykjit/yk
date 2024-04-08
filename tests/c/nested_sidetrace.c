// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_LOG_JITSTATE=-
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     jitstate: start-tracing
//     1
//     jitstate: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2
//     jitstate: enter-jit-code
//     3
//     4
//     5
//     6
//     7
//     8
//     9
//     10
//     jitstate: deoptimise
//     12
//     jitstate: enter-jit-code
//     jitstate: deoptimise
//     14
//     jitstate: enter-jit-code
//     jitstate: deoptimise
//     16
//     jitstate: enter-jit-code
//     jitstate: deoptimise
//     18
//     jitstate: enter-jit-code
//     jitstate: deoptimise
//     jitstate: start-side-tracing
//     20
//     jitstate: stop-side-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     22
//     jitstate: enter-jit-code
//     jitstate: execute-side-trace
//     24
//     jitstate: deoptimise
//     26
//     jitstate: enter-jit-code
//     jitstate: execute-side-trace
//     28
//     jitstate: deoptimise
//     30
//     ...
//     42
//     jitstate: enter-jit-code
//     jitstate: execute-side-trace
//     jitstate: deoptimise
//     jitstate: start-side-tracing
//     45
//     jitstate: stop-side-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     48
//     jitstate: enter-jit-code
//     jitstate: execute-side-trace
//     jitstate: execute-side-trace
//     51
//     jitstate: deoptimise
//     54
//     jitstate: enter-jit-code
//     jitstate: execute-side-trace
//     jitstate: execute-side-trace
//     57
//     jitstate: deoptimise
//     60
//   stdout:
//     exit

// Tests that side-traces can be compiled from within other side-traces. In
// this case a side-trace is first compiled for the `else` branch in foo. That
// side-trace itself contain a guard for the `else { return 3; }` branch, which
// eventually fails often enough to compile a nested side-trace.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int foo(int i) {
  if (i > 20) {
    return 1;
  } else {
    if (i > 10) {
      return 2;
    } else {
      return 3;
    }
  }
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 5);
  YkLocation loc = yk_location_new();

  int res = 0;
  int i = 30;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    res += foo(i);
    fprintf(stderr, "%d\n", res);
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
