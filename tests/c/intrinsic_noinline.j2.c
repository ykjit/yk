// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     memcpy ...
//     ...
//     --- End jit-pre-opt ---
//     yk-execution: enter-jit-code
//     ...
//     yk-execution: deoptimise ...
//     ...
//   stdout:
//     3

// Check non-inlinable intrinsics.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int res[100];
  int src[100];
  // Make the array big enough so that the memcpy won't get inlined by the
  // compiler.
  for (int i = 0; i < 100; i++) {
    src[i] = argc * i;
  }
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 5;
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  NOOPT_VAL(src);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    // Add observable effect to check the trace executes this memcpy.
    src[0] = i * 3;
    memcpy(&res, &src, sizeof(int) * 100);
    i--;
  }
  NOOPT_VAL(res);
  printf("%d", res[0]);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);

  return (EXIT_SUCCESS);
}
