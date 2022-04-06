// ignore: Requires global variable stderr in stopgap.
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     define internal void @__yk_compiled_trace_0(...
//       ...
//       store i32 30, i32* %{{0}}, align 4...
//       ...
//     --- End jit-pre-opt ---
//     2:30
//     jit-state: enter-jit-code
//     1:30
//     ...

// Check that returning a constant value from a traced function works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f() { return 30; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new();
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 3, res = 0;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    res = f();
    NOOPT_VAL(res);
    assert(res == 30);
    fprintf(stderr, "%d:%d\n", i, res);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
