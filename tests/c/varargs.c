// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     i=6
//     yk-tracing: stop-tracing
//     i=6
//     yk-execution: enter-jit-code
//     i=6
//     i=6
//     yk-execution: deoptimise ...

// Check varargs calls work.

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int varargfunc(int len, ...) {
  int acc = 0;
  va_list argp;
  va_start(argp, len);
  for (int i = 0; i < len; i++) {
    int arg = va_arg(argp, int);
    acc += arg;
  }
  va_end(argp);
  return acc;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int res = varargfunc(3, argc, 2, 3);
    fprintf(stderr, "i=%d\n", res);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
