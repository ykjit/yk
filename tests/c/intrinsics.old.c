// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     jitstate: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(ptr %0, ptr %1...
//        ...
//     }
//     ...
//     --- End jit-pre-opt ---
//     jitstate: enter-jit-code
//     ...
//     jitstate: deoptimise
//     ...
//   stdout:
//     998

// Check that inlined intrinsics are handled correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int res = 0;
  int src = 1000;
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 3;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  NOOPT_VAL(src);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    memcpy(&res, &src, 4);
    src--;
    i--;
  }
  NOOPT_VAL(res);
  printf("%d", res);
  yk_location_drop(loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
