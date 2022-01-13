// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     jit-state: start-tracing
//     i=4
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     define internal void @__yk_compiled_trace_0(%YkCtrlPointVars* %0, i64* %1, i64 %2) {
//        ...
//        %{{fptr}} = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %0, i32 0, i32 0...
//        %{{load}} = load...
//        ...
//        %{{a}} = add nsw i32 %{{b}}, %{{add}}...
//        ...
//        %{{cond}} = icmp sgt i32 %{{val}}, ...
//        br i1 %{{cond}}, label %{{guard-succ-bb}}, label %{{guard-fail-bb}}
//
//     {{guard-fail-bb}}:...
//       call void (...) @llvm.experimental.deoptimize.isVoid(i64* %1, i64 %2) ...
//       ret void
//
//     {{guard-succ-bb}}:...
//        ...
//        %{{fptr2}} = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %0, i32 0, i32 0...
//        store...
//        ...
//        ret void
//     }
//     ...
//     --- End jit-pre-opt ---
//     i=3
//     jit-state: enter-jit-code
//     i=2
//     jit-state: exit-jit-code
//     jit-state: enter-jit-code
//     i=1
//     jit-state: stopgap
//     ...

// Check that using a global works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int add;

int main(int argc, char **argv) {
  yk_set_hot_threshold(0);
  int res = 0;
  YkLocation loc = yk_location_new();
  int i = 4;
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
  yk_location_drop(loc);

  return (EXIT_SUCCESS);
}
