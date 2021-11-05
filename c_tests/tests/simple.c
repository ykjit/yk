// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     ...
//     define internal %YkCtrlPointVars @__yk_compiled_trace_0(%YkCtrlPointVars %0) {
//        ...
//        %%x = extractvalue %YkCtrlPointVars %0, ...
//        ...
//        %%a = add nsw i32 %%b, 2...
//        ...
//        %%z = insertvalue %YkCtrlPointVars %%y, ...
//        ...
//     }
//     ...
//     jit-state: stop-tracing
//     jit-state: enter-jit-code
//     jit-state: exit-jit-code

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
  int loc = 0;
  int i = 3;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_control_point(loc);
    res += 2;
    i--;
  }
  NOOPT_VAL(res);
  assert(res == 8);

  return (EXIT_SUCCESS);
}
