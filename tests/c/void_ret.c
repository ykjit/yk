// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-execution: enter-jit-code
//     inside f
//     inside f
//     yk-execution: deoptimise
//     ...

// Check inlining a function into the trace that has a void return type.

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
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
