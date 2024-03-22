// # Currently this test breaks CI entirely, so we temporarily ignore it
// # completely.
// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_PRINT_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   status: error
//   stderr:
//     jit-state: start-tracing
//     jit-state: stop-tracing
//     --- Begin aot ---
//     ...
//     func main($arg0: i32, $arg1: ptr) -> i32 {
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     ...
//     jit-state: enter-jit-code
//     ...
//     deopt...
//     ...

// Check that basic trace compilation works.

// FIXME: Get this test all the way through the new codegen pipeline!
//
// Currently it succeeds even though it crashes out on a todo!(). This is so
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
    // FIXME: ideally we'd print to stderr so as to have the output interleaved
    // with jit-state prints, but we can't yet handle the `stderr` constant in
    // `fputs("i", stderr)`.
    puts("i");
    res += 2;
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
