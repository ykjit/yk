// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     pepper
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     global_decl @fruits
//     global_decl @.str
//     global_decl @.str.1
//     ...
//     --- End aot ---
//     cucumber
//     yk-execution: enter-jit-code
//     tomato
//     banana
//     yk-execution: deoptimise
//   stdout:
//     exit

// Check that when we clone a `GlobalVariable` into the JITMod, and that global
// references other `GlobalVariables`, the other globals are cloned as well.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

static const char *const fruits[] = {"apple", "banana", "tomato", "cucumber",
                                     "pepper"};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%s\n", fruits[i]);
    res += 2;
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
