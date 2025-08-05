// ## longjmp detection with CFI breaks this.
// ignore-if: true
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-execution: deoptimise...
//     ...
//     yk-execution: enter-jit-code
//     1
//     yk-execution: enter-jit-code
//     1
//     yk-execution: deoptimise...
//     return inner
//     yk-execution: deoptimise...
//     return outer
//     exit

// Test traces recursively calling the interpreter loop which in turn
// executes another trace, effectively leading to nested trace execution.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *, YkLocation *, int, char *);

__attribute__((yk_outline))
void loop(YkMT *mt, YkLocation *loc1, YkLocation *loc2, int i, char* inner) {
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc2);
    fprintf(stderr, "%d\n", i);
    if (strcmp(inner, "outer") == 0) {
      loop(mt, NULL, loc1, 1, "inner");
    }
    i--;
  }
  fprintf(stderr, "return %s\n", inner);
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 3);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();

  // Make sure location 1 is compiled first.
  loop(mt, NULL, &loc1, 3, "inner");

  // Then compile location 2.
  loop(mt, &loc1, &loc2, 3, "outer");
  fprintf(stderr, "exit\n");
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
