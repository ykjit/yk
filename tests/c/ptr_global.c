// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     i=25
//     jitstate: deoptimise
//     ...

// Check that tracing mutation of a global pointer works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

char *p = NULL;

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  int i = 0;
  YkLocation loc = yk_location_new();
  p = argv[0];
  NOOPT_VAL(i);
  while (*p != '\0') {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    i++;
    p++;
  }
  NOOPT_VAL(i);
  NOOPT_VAL(p);
  yk_location_drop(loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
