// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     func foo(...
//     ...
//     *@shadowstack_0 = %{{1_1}}
//     ...
//     call bar(...
//     ...
//     *@shadowstack_0 = %{{0_1}}
//     ...
//     --- End aot ---
//     ...
//   stdout:
//     2
//     2
//     2
//     2
//     2
//     1
//     2
//     2
//     ...
//     exit

// Check references created inside a trace are correctly read after deoptimisation.
// Essentially tests that shadow stacks are working.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) void bar(int i, int *ptr) {
  int a = i * 2;
  int result;
  if (a == 10) {
    result = 1;
  } else {
    result = 2;
  }
  *ptr = result;
}

__attribute__((noinline)) void foo(int i) {
  int res = 0;
  bar(i, &res);
  printf("%d\n", res);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 0;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i < 20) {
    yk_mt_control_point(mt, &loc);
    foo(i);
    i++;
  }
  printf("exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
