// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_PRINT_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   status: error
//   stderr:
//     jit-state: start-tracing
//     foo
//     jit-state: stop-tracing
//     --- Begin aot ---
//     ...
//     func main($arg0: i32, $arg1: ptr) -> i32 {
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     foo
//     jit-state: enter-jit-code
//     foo
//     foo
//     deopt

// Check that basic trace compilation works.

// FIXME: Get this test all the way through the new codegen pipeline!
//
// Currently it succeeds even though it crashes on deopt. This is so
// that we can incrementally implement the new codegen and have CI merge our
// incomplete work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

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
    fputs("foo\n", stderr);
    res += 2;
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
