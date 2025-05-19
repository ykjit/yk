// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4:21
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     3:21
//     yk-execution: enter-jit-code
//     2:21
//     1:21
//     yk-execution: deoptimise
//     ...

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int fib(int num) {
  if (num == 0)
    return 0;
  if (num == 1)
    return 1;
  if (num == 2)
    return 1;
  int a = fib(num - 2);
  int b = fib(num - 1);
  int c = a + b;
  return c;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(argc);
    fprintf(stderr, "%d:%d\n", i, fib(argc * 8));
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
