// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt,aot
//   stderr:
//     jit-state: start-tracing
//     ...
//     define internal %YkCtrlPointVars @__yk_compiled_trace_0(%YkCtrlPointVars %0) {
//        ...
//        %{{x}} = extractvalue %YkCtrlPointVars %0, ...
//        ...
//        %{{a}} = add nsw i32 %{{b}}, %{{add}}...
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
//     mutable_global: guard-failure

// Check that using a global works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int add;

int main(int argc, char **argv) {
  int res = 0;
  YkLocation loc = yk_location_new();
  int i = 5;
  add = argc + 1; // 2
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_control_point(&loc);
    fprintf(stderr, "i=%d\n", i);
    res += add;
    i--;
  }
  abort(); // FIXME: unreachable due to aborting guard failure earlier.
  NOOPT_VAL(res);

  return (EXIT_SUCCESS);
}
