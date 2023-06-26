// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     jit-state: start-tracing
//     i=4
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     define {{rtnty}} @__yk_compiled_trace_0(ptr %0, ptr %1, ptr %2) {
//        ...
//        %{{fptr}} = getelementptr %YkCtrlPointVars, ptr %0, i32 0, i32 0...
//        %{{load}} = load...
//        ...
//        br label %{{loopentry}}
//
//     {{loopentry}}:...
//        ...
//        %{{a}} = add nsw i32 %{{b}}, 2...
//        ...
//        %{{cond}} = icmp sgt i32 %{{val}}, ...
//        br i1 %{{cond}}, label %{{guard-succ-bb}}, label %{{guard-fail-bb}}
//
//     {{guard-fail-bb}}:...
//       %{{al}} = alloca...
//       ...
//       %{{cprtn}} = call {{rtnty}} (...) @llvm.experimental.deoptimize.p0(...
//       ret {{rtnty}} %{{cprtn}}
//
//     {{guard-succ-bb}}:...
//        ...
//        br label %{{loopentry}}
//     }
//     ...
//     --- End jit-pre-opt ---
//     i=3
//     jit-state: enter-jit-code
//     i=2
//     i=1
//     jit-state: deoptimise
//   stdout:
//     exit

// Check that basic trace compilation works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    res += 2;
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
