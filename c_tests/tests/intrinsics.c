// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     jit-state: start-tracing
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     define internal void @__yk_compiled_trace_0(%YkCtrlPointVars* %0) {
//        ...
//     }
//     ...
//     --- End jit-pre-opt ---
//     jit-state: enter-jit-code
//     intrinsics: guard-failure

// Check that basic trace compilation works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void _yk_test(int i, int res) {
  if (i == 0)
    assert(res == 2);
}

int main(int argc, char **argv) {
  int res = 0;
  YkLocation loc = yk_location_new();
  int i = 4;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_control_point(&loc);
    memcpy(&res, &i, 4);
    i--;
  }
  NOOPT_VAL(res);
  assert(res == 0);
  yk_location_drop(loc);

  return (EXIT_SUCCESS);
}
