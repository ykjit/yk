// ## FIXME: Test fails for both hwt and swt.
// ignore-if: test true
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4

// Check that we can reliably deal with "foreign" (not compiled with ykllvm)
// code that calls back into "native code".
//
// FIXME: actually match some IR/output.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int call_callback(int (*callback)(int, int), int x, int y);

__attribute((noinline)) int callback(int x, int y) { return (x + y) / 2; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int x = 0;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(x);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d, x=%d\n", i, x);
    call_callback(&callback, i, i);
    i--;
  }
  NOOPT_VAL(x);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
