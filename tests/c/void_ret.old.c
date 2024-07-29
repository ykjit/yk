// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     jitstate: enter-jit-code
//     inside f
//     inside f
//     jitstate: deoptimise
//     ...

// Check that inlining a function with a void return type works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

void __attribute__((noinline)) f() {
  fputs("inside f\n", stderr);
  return;
}

int main(int argc, char **argv) {

  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    f();
    i--;
  }

  NOOPT_VAL(i);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
