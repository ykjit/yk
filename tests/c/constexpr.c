// ## -O0 doesn't make a constant expression.
// ignore-if: true
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     @.str = ...
//     ...
//     define void @__yk_compiled_trace_0(...
//       ...
//       ...store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0)...
//       ...
//     --- End jit-pre-opt ---
//     2:97
//     yk-jit-event: enter-jit-code
//     1:97
//     ...

// Check that global variables inside constant expressions are handled.
// FIXME: needs porting to Yk IR once we find out how to get a constexpr gep.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

const volatile int global_int = 6;

__attribute__((noinline)) char foo(char *str) {
  NOOPT_VAL(str);
  // At optimisation levels >O0 this makes a constant GEP expression.
  return str[0];
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 0, i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    res = foo("abc");
    assert(res == 97);
    fprintf(stderr, "%d:%d\n", i, res);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
