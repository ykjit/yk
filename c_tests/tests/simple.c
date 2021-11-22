// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     ...
//     define internal %YkCtrlPointVars @__yk_compiled_trace_0(%YkCtrlPointVars %0) {
//        ...
//        %{{x}} = extractvalue %YkCtrlPointVars %0, ...
//        ...
//        %{{a}} = add nsw i32 %{{b}}, 2...
//        ...
//        %{{cond}} = icmp sgt i32 %{{val}}, ...
//        br i1 %{{cond}}, label %{{guard-succ-bb}}, label %{{guard-fail-bb}}
//
//     {{guard-fail-bb}}:...
//       ...
//       call void (i32, i8*, ...) @errx(i32 0,...
//       unreachable
//
//     {{guard-succ-bb}}:...
//        ...
//        %{{z}} = insertvalue %YkCtrlPointVars %{{y}}, ...
//        ...
//        ret %YkCtrlPointVars %{{ret}}
//     }
//     ...
//     jit-state: stop-tracing
//     i=4
//     jit-state: enter-jit-code
//     i=3
//     jit-state: exit-jit-code
//     jit-state: enter-jit-code
//     i=2
//     jit-state: exit-jit-code
//     jit-state: enter-jit-code
//     i=1
//     simple: guard-failure

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
  int i = 5;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_control_point(loc);
    fprintf(stderr, "i=%d\n", i);
    res += 2;
    i--;
  }
  abort(); // FIXME: unreachable due to aborting guard failure earlier.
  NOOPT_VAL(res);

  return (EXIT_SUCCESS);
}
