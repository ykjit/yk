// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     jitstate: enter-jit-code
//     z=4
//     ...

// Test indirect calls where we have IR for the callee.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int a) {
  NOOPT_VAL(a);
  return a + 1;
}

int bar(int (*func)(int)) {
  int a = func(3);
  return a;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int z = 0, i = 4;
  NOOPT_VAL(i);
  NOOPT_VAL(z);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    z = bar(foo);
    assert(z == 4);
    fprintf(stderr, "z=%d\n", z);
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
