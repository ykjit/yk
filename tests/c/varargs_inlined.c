// ## trace-compilation-aborted: dlsym("llvm.va_start") returned NULL
// ignore-if: true
// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     i=1
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     call llvm.va_start...
//     ...
//     call llvm.va_end...
//     ...
//     --- End aot ---
//     i=1
//     yk-jit-event: enter-jit-code
//     i=1
//     i=1
//     yk-jit-event: deoptimise

// Check that inlining works when the function is vararg.

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
  int arg = va_arg(argp, int);
  acc += arg;
  va_end(argp);
  return acc;
}

int foo(int argc) { return varargfunc(3, argc, 2, 3); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int res = foo(argc);
    fprintf(stderr, "i=%d\n", res);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
