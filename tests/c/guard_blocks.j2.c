// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     8
//     --- Begin jit-pre-opt ---
//     ...
//     guard ...
//     ...
//     guard ...
//     ...
//     term [%0, %1, %2, %3]
//     ; guard 0
//     %0: ptr = arg
//     %1: ptr = arg
//     %2: ptr = arg
//     %3: ptr = arg
//     %4: i1 = 0
//     term [%0, %1, %2, %3, %4]
//     ; guard 1
//     %0: ptr = arg
//     %1: ptr = arg
//     %2: ptr = arg
//     %3: ptr = arg
//     %4: i1 = 0
//     term [%0, %1, %2, %3, %4]
//     --- End jit-pre-opt ---
//     7
//     6
//     5
//     4
//     3
//     --- Begin jit-pre-opt ---
//     ...
//     guard ...
//     ...
//     term [%0, %1, %2, %3]
//     ; guard 0
//     %0: ptr = arg
//     %1: ptr = arg
//     %2: ptr = arg
//     %3: ptr = arg
//     %4: i1 = 0
//     term [%0, %1, %2, %3, %4]
//     --- End jit-pre-opt ---
//     2
//     1

// Check that if a guard's life variables include the condition operand, that
// is converted to a constant.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 2);
  YkLocation loc = yk_location_new();

  int res = 0;
  int i = 8;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    if (i % 2 == 0)
      res += 1;
    fprintf(stderr, "%d\n", i);
    i--;
  }
  NOOPT_VAL(res);
  printf("%d\n", res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
