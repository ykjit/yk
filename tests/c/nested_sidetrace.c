// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-tracing: start-tracing
//     1
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     2
//     yk-execution: enter-jit-code
//     3
//     4
//     5
//     6
//     7
//     8
//     9
//     10
//     yk-execution: deoptimise ...
//     12
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     14
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     16
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     18
//     yk-execution: enter-jit-code
//     yk-execution: deoptimise ...
//     yk-tracing: start-side-tracing
//     20
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     22
//     yk-execution: enter-jit-code
//     24
//     26
//     28
//     30
//     yk-execution: deoptimise ...
//     ...
//     yk-tracing: start-side-tracing
//     45
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     48
//     yk-execution: enter-jit-code
//     51
//     54
//     57
//     60
//     yk-execution: deoptimise ...
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
  fprintf(stderr, "exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
