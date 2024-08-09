// ## This test breaks in swt tracer as swt tracer is missing unmappable block so it cannot
// ## see calls from unmappable blocks to mappable blocks and vice-versa. Disable this test
// ## for swt until we fix it.
// ## Example of what hwt see:
// ## mappable block :           <---
// ##        unmappable block       |
// ##         mappable block  ------
// ##
// ## Also the new trace builder cannot yet handle callbacks -- hits a todo!
// ignore-if: test "$YKB_TRACER" = "swt" || true
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YK_LOG=4

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
