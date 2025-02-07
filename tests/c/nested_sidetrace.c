// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-jit-event: start-tracing
//     1
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2
//     yk-jit-event: enter-jit-code
//     3
//     4
//     5
//     6
//     7
//     8
//     9
//     10
//     yk-jit-event: deoptimise
//     12
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     14
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     16
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     18
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     20
//     yk-jit-event: enter-jit-code
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     22
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     24
//     yk-jit-event: enter-jit-code
//     yk-jit-event: execute-side-trace
//     26
//     yk-jit-event: execute-side-trace
//     28
//     yk-jit-event: execute-side-trace
//     30
//     ...
//     36
//     yk-jit-event: enter-jit-code
//     yk-jit-event: execute-side-trace
//     yk-jit-event: deoptimise
//     ...
//     42
//     yk-jit-event: enter-jit-code
//     yk-jit-event: execute-side-trace
//     yk-jit-event: deoptimise
//     yk-jit-event: start-side-tracing
//     45
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     48
//     yk-jit-event: enter-jit-code
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     51
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     54
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     57
//     yk-jit-event: execute-side-trace
//     yk-jit-event: execute-side-trace
//     60
//     yk-jit-event: deoptimise
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
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
