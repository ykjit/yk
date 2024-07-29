// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     ...
//     jitstate: stop-tracing
//     ...
//     jitstate: enter-jit-code
//     ...
//     jitstate: deoptimise
//  stdout:
//     ...
//     i: 5 ret: 12
//     ...
//     i: 2 ret: 9
//     i: 1 ret: 108

// Test deoptimisation inside nested call.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int h(int a, int b) {
  if (a > 1) {
    return a + b;
  } else {
    return a + b + 100;
  }
}

__attribute__((noinline)) int g(int a, int b) {
  int c = b + 2;
  return h(a, c);
}

__attribute__((noinline)) int f(int a, int b) {
  int c = b + 1;
  return g(a, c);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 5;
  int ret = 0;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    ret = f(i, argc + 3);
    printf("i: %d ret: %d\n", i, ret);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
